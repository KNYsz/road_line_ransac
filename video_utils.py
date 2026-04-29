"""
Video frame extraction utilities.
"""

import cv2
from pathlib import Path


def iter_frames(video_path, frame_interval=10):
    """Yield (frame_index, frame) tuples from a video file.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video file.
    frame_interval : int
        Yield one frame every *frame_interval* frames.
        E.g. ``frame_interval=10`` yields frames 0, 10, 20, …

    Yields
    ------
    frame_index : int
        0-based index of the frame in the original video.
    frame : np.ndarray
        BGR image array.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    IOError
        If the video cannot be opened by OpenCV.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_interval == 0:
                yield frame_index, frame
            frame_index += 1
    finally:
        cap.release()


def get_video_info(video_path):
    """Return basic metadata for a video file.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.

    Returns
    -------
    dict with keys:
        ``width``, ``height``, ``fps``, ``total_frames``
    """
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def draw_lines(frame, lines, color=(0, 0, 255), thickness=2):
    """Overlay detected lines on a copy of *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to draw on (not modified in-place).
    lines : list of tuple (a, b, c)
        Normalised line coefficients (ax + by + c = 0).
    color : tuple
        BGR colour for drawing.
    thickness : int
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        New BGR image with lines drawn.
    """
    out = frame.copy()
    h, w = frame.shape[:2]

    for a, b, c in lines:
        # Compute two endpoint x-coordinates and solve for y (or vice-versa)
        pts = []
        if abs(b) > 1e-6:
            # y = -(ax + c) / b  →  solve at x=0 and x=w-1
            for x in [0, w - 1]:
                y = int(round(-(a * x + c) / b))
                if 0 <= y < h:
                    pts.append((x, y))
        if abs(a) > 1e-6:
            # x = -(by + c) / a  →  solve at y=0 and y=h-1
            for y in [0, h - 1]:
                x = int(round(-(b * y + c) / a))
                if 0 <= x < w:
                    pts.append((x, y))

        # Keep at most two distinct points
        unique_pts = []
        for p in pts:
            if p not in unique_pts:
                unique_pts.append(p)
            if len(unique_pts) == 2:
                break

        if len(unique_pts) == 2:
            cv2.line(out, unique_pts[0], unique_pts[1], color, thickness)

    return out
