"""
Edge detection utilities for road white-line extraction.
"""

import cv2
import numpy as np


def apply_roi_mask(image, roi_top_ratio=0.5):
    """Zero out the upper portion of *image* to focus on the road region.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image.
    roi_top_ratio : float
        Fraction of the image height from the top to mask out (0–1).
        E.g. 0.5 keeps only the bottom half.

    Returns
    -------
    np.ndarray
        Image with the upper portion set to zero.
    """
    mask = np.zeros_like(image)
    h = image.shape[0]
    top = int(h * roi_top_ratio)
    if image.ndim == 3:
        mask[top:, :, :] = image[top:, :, :]
    else:
        mask[top:, :] = image[top:, :]
    return mask


def detect_edges(
    frame,
    blur_kernel=5,
    canny_low=50,
    canny_high=150,
    roi_top_ratio=0.5,
    white_threshold=200,
):
    """Extract edge pixels from a BGR video frame.

    Processing pipeline:
    1. Convert to grayscale.
    2. Optionally isolate bright (white) pixels to focus on white lines.
    3. Apply Gaussian blur to reduce noise.
    4. Run Canny edge detection.
    5. Mask out the upper portion of the frame (ROI).

    Parameters
    ----------
    frame : np.ndarray
        BGR image (single video frame).
    blur_kernel : int
        Side length of the Gaussian blur kernel (must be odd and ≥ 1).
    canny_low : int
        Lower hysteresis threshold for Canny.
    canny_high : int
        Upper hysteresis threshold for Canny.
    roi_top_ratio : float
        Fraction of image height to mask out from the top (0–1).
    white_threshold : int
        Pixel brightness (0–255) above which a pixel is considered "white".
        Set to 0 to disable white-pixel filtering.

    Returns
    -------
    edges : np.ndarray
        Binary edge image (uint8, values 0 or 255).
    edge_points : np.ndarray, shape (N, 2)
        Array of (x, y) coordinates of edge pixels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Isolate bright (white) regions when a threshold is given
    if white_threshold > 0:
        _, white_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
        gray = cv2.bitwise_and(gray, white_mask)

    # Gaussian blur – ensure kernel size is odd and at least 1
    k = max(1, blur_kernel | 1)  # force odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Apply ROI mask
    edges = apply_roi_mask(edges, roi_top_ratio=roi_top_ratio)

    # Collect edge-pixel coordinates as (x, y)
    ys, xs = np.where(edges > 0)
    edge_points = np.column_stack((xs, ys)) if len(xs) > 0 else np.empty((0, 2), dtype=int)

    return edges, edge_points
