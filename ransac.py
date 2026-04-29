"""
RANSAC (Random Sample Consensus) algorithm for line detection.
"""

import numpy as np


def fit_line_to_inliers(points):
    """Fit a line to a set of points using least squares.

    Returns (a, b, c) such that ax + by + c = 0, normalized so that a^2+b^2=1.
    """
    if len(points) < 2:
        return None

    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    # Use SVD-based total least squares
    centroid = np.array([x.mean(), y.mean()])
    centered = points.astype(float) - centroid
    _, _, vt = np.linalg.svd(centered)
    # Normal vector is the last row of Vt (smallest singular value)
    a, b = vt[-1]
    c = -(a * centroid[0] + b * centroid[1])
    norm = np.sqrt(a ** 2 + b ** 2)
    if norm == 0:
        return None
    return (a / norm, b / norm, c / norm)


def point_line_distances(points, line):
    """Compute perpendicular distances from points to line ax+by+c=0."""
    a, b, c = line
    return np.abs(a * points[:, 0] + b * points[:, 1] + c)


def ransac_line(points, n_iterations=100, threshold=5.0, min_inliers=50):
    """
    Fit a line to 2-D points using RANSAC.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Array of (x, y) edge-pixel coordinates.
    n_iterations : int
        Number of RANSAC iterations.
    threshold : float
        Maximum perpendicular distance (pixels) for a point to be an inlier.
    min_inliers : int
        Minimum number of inliers required to accept a hypothesis.

    Returns
    -------
    best_line : tuple (a, b, c) or None
        Normalised line coefficients (ax + by + c = 0), or None if no line found.
    best_mask : np.ndarray of bool or None
        Boolean inlier mask aligned with *points*, or None if no line found.
    """
    if len(points) < 2:
        return None, None

    n_points = len(points)
    best_mask = None
    best_line = None
    best_n_inliers = 0

    rng = np.random.default_rng()

    for _ in range(n_iterations):
        # Sample 2 random points to form a line hypothesis
        idx = rng.choice(n_points, 2, replace=False)
        p1, p2 = points[idx]

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm < 1e-6:
            continue

        # Line: dy*x - dx*y + (dx*p1[1] - dy*p1[0]) = 0, then normalise
        a = dy / norm
        b = -dx / norm
        c = (dx * p1[1] - dy * p1[0]) / norm

        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        mask = distances < threshold
        n_inliers = int(mask.sum())

        if n_inliers >= min_inliers and n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_mask = mask
            best_line = (a, b, c)

    if best_line is None:
        return None, None

    # Refit using all inliers for a more accurate line
    refined = fit_line_to_inliers(points[best_mask])
    if refined is not None:
        best_line = refined
        distances = point_line_distances(points, best_line)
        best_mask = distances < threshold

    return best_line, best_mask


def detect_lines(points, max_lines=2, n_iterations=100, threshold=5.0, min_inliers=50):
    """
    Detect multiple lines from edge points using iterative RANSAC.

    Each detected line's inliers are removed before the next iteration so that
    distinct lines can be found.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Edge-pixel coordinates.
    max_lines : int
        Maximum number of lines to detect.
    n_iterations : int
        RANSAC iterations per line.
    threshold : float
        Inlier distance threshold (pixels).
    min_inliers : int
        Minimum inliers to accept a line.

    Returns
    -------
    lines : list of tuple (a, b, c)
        Detected line coefficients.
    masks : list of np.ndarray of bool
        Inlier masks (each aligned with the *original* points array).
    """
    lines = []
    masks = []

    remaining = np.ones(len(points), dtype=bool)

    for _ in range(max_lines):
        candidate_points = points[remaining]
        if len(candidate_points) < min_inliers:
            break

        line, local_mask = ransac_line(
            candidate_points,
            n_iterations=n_iterations,
            threshold=threshold,
            min_inliers=min_inliers,
        )
        if line is None:
            break

        # Map local mask back to original index space
        remaining_indices = np.where(remaining)[0]
        global_mask = np.zeros(len(points), dtype=bool)
        global_mask[remaining_indices[local_mask]] = True

        lines.append(line)
        masks.append(global_mask)

        # Remove inliers so the next iteration finds a different line
        remaining[global_mask] = False

    return lines, masks
