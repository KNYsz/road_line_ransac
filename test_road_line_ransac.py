"""
Tests for the road_line_ransac modules.
"""

import sys
import os
import numpy as np
import pytest

# Make sure the package root is importable when running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ransac import fit_line_to_inliers, point_line_distances, ransac_line, detect_lines
from edge_detection import apply_roi_mask, detect_edges
from video_utils import draw_lines


# ---------------------------------------------------------------------------
# ransac.py
# ---------------------------------------------------------------------------

class TestFitLineToInliers:
    def test_horizontal_line(self):
        points = np.array([[0, 5], [1, 5], [2, 5], [3, 5]], dtype=float)
        a, b, c = fit_line_to_inliers(points)
        # Line should be y = 5, i.e. 0*x + 1*y - 5 = 0 (or negated)
        dist = point_line_distances(points, (a, b, c))
        np.testing.assert_allclose(dist, 0, atol=1e-6)

    def test_vertical_line(self):
        points = np.array([[3, 0], [3, 1], [3, 2], [3, 3]], dtype=float)
        a, b, c = fit_line_to_inliers(points)
        dist = point_line_distances(points, (a, b, c))
        np.testing.assert_allclose(dist, 0, atol=1e-6)

    def test_diagonal_line(self):
        # y = x  →  points on y = x
        points = np.column_stack([np.arange(10), np.arange(10)]).astype(float)
        a, b, c = fit_line_to_inliers(points)
        dist = point_line_distances(points, (a, b, c))
        np.testing.assert_allclose(dist, 0, atol=1e-6)

    def test_too_few_points_returns_none(self):
        result = fit_line_to_inliers(np.array([[1, 2]]))
        assert result is None


class TestRansacLine:
    def _make_line_points(self, n=200, slope=1.0, intercept=0.0, noise=0.5, seed=42):
        rng = np.random.default_rng(seed)
        xs = rng.uniform(0, 100, n)
        ys = slope * xs + intercept + rng.normal(0, noise, n)
        return np.column_stack([xs, ys])

    def test_detects_near_horizontal_line(self):
        points = self._make_line_points(slope=0.1, intercept=50, noise=1.0)
        line, mask = ransac_line(points, n_iterations=200, threshold=3.0, min_inliers=50)
        assert line is not None
        assert mask is not None
        # The majority of points should be inliers
        assert mask.sum() > len(points) * 0.8

    def test_detects_diagonal_line(self):
        points = self._make_line_points(slope=2.0, intercept=10, noise=0.5)
        line, mask = ransac_line(points, n_iterations=200, threshold=3.0, min_inliers=50)
        assert line is not None
        assert mask.sum() > len(points) * 0.8

    def test_returns_none_when_insufficient_points(self):
        points = np.array([[0, 0], [1, 1]], dtype=float)
        line, mask = ransac_line(points, n_iterations=10, threshold=2.0, min_inliers=50)
        assert line is None
        assert mask is None

    def test_returns_none_for_empty_input(self):
        points = np.empty((0, 2), dtype=float)
        line, mask = ransac_line(points)
        assert line is None
        assert mask is None

    def test_normalised_coefficients(self):
        points = self._make_line_points(slope=1.0, intercept=0.0, noise=0.3)
        line, _ = ransac_line(points, n_iterations=200, threshold=3.0, min_inliers=50)
        if line is not None:
            a, b, c = line
            np.testing.assert_allclose(a ** 2 + b ** 2, 1.0, atol=1e-6)


class TestDetectLines:
    def _two_line_points(self, n=300):
        """Generate points on two crossing lines with noise."""
        rng = np.random.default_rng(0)
        # Line 1: y = 0.5x + 20
        xs1 = rng.uniform(0, 200, n)
        ys1 = 0.5 * xs1 + 20 + rng.normal(0, 1, n)
        # Line 2: y = -0.5x + 180
        xs2 = rng.uniform(0, 200, n)
        ys2 = -0.5 * xs2 + 180 + rng.normal(0, 1, n)
        return np.vstack([np.column_stack([xs1, ys1]), np.column_stack([xs2, ys2])])

    def test_detects_two_lines(self):
        points = self._two_line_points()
        lines, masks = detect_lines(
            points, max_lines=2, n_iterations=300, threshold=3.0, min_inliers=100
        )
        assert len(lines) == 2
        assert len(masks) == 2

    def test_max_lines_respected(self):
        points = self._two_line_points()
        lines, masks = detect_lines(
            points, max_lines=1, n_iterations=200, threshold=3.0, min_inliers=100
        )
        assert len(lines) <= 1

    def test_empty_input(self):
        lines, masks = detect_lines(np.empty((0, 2), dtype=float))
        assert lines == []
        assert masks == []


# ---------------------------------------------------------------------------
# edge_detection.py
# ---------------------------------------------------------------------------

class TestApplyRoiMask:
    def test_upper_half_zeroed_grayscale(self):
        img = np.ones((100, 100), dtype=np.uint8) * 255
        masked = apply_roi_mask(img, roi_top_ratio=0.5)
        assert np.all(masked[:50, :] == 0)
        assert np.all(masked[50:, :] == 255)

    def test_upper_half_zeroed_bgr(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        masked = apply_roi_mask(img, roi_top_ratio=0.5)
        assert np.all(masked[:50, :, :] == 0)
        assert np.all(masked[50:, :, :] == 255)

    def test_no_masking_at_zero(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        masked = apply_roi_mask(img, roi_top_ratio=0.0)
        np.testing.assert_array_equal(masked, img)


class TestDetectEdges:
    def _make_frame(self, h=200, w=300):
        """Create a synthetic BGR frame with a bright horizontal band."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # White horizontal stripe at y = 150
        frame[148:153, :, :] = 255
        return frame

    def test_returns_edges_and_points(self):
        frame = self._make_frame()
        edges, points = detect_edges(
            frame,
            blur_kernel=3,
            canny_low=50,
            canny_high=150,
            roi_top_ratio=0.5,
            white_threshold=200,
        )
        assert edges.shape == (frame.shape[0], frame.shape[1])
        assert points.ndim == 2
        assert points.shape[1] == 2

    def test_roi_removes_upper_edges(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        # White stripe in the upper half (should be masked out)
        frame[10:15, :, :] = 255
        _, points = detect_edges(
            frame,
            blur_kernel=3,
            canny_low=10,
            canny_high=50,
            roi_top_ratio=0.5,
            white_threshold=0,
        )
        # All detected edge points must be in the lower half (y >= 100)
        if len(points) > 0:
            assert np.all(points[:, 1] >= 100)

    def test_no_edges_in_blank_frame(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        _, points = detect_edges(frame, blur_kernel=3, canny_low=50, canny_high=150)
        assert len(points) == 0


# ---------------------------------------------------------------------------
# video_utils.py
# ---------------------------------------------------------------------------

class TestDrawLines:
    def test_returns_copy(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = draw_lines(frame, [])
        assert result is not frame

    def test_no_lines_unchanged(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = draw_lines(frame, [])
        np.testing.assert_array_equal(result, frame)

    def test_horizontal_line_drawn(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        # y = 50  →  0*x + 1*y - 50 = 0  (normalised: a=0, b=1, c=-50)
        result = draw_lines(frame, [(0, 1, -50)], color=(255, 0, 0), thickness=1)
        # Some pixels on row 50 should be non-zero
        assert result[50, :, 0].sum() > 0

    def test_vertical_line_drawn(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        # x = 100  →  1*x + 0*y - 100 = 0
        result = draw_lines(frame, [(1, 0, -100)], color=(0, 255, 0), thickness=1)
        assert result[:, 100, 1].sum() > 0
