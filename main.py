#!/usr/bin/env python3
"""
Road white-line detection using RANSAC.

Usage example::

    python main.py input.mp4 -o results/ --frame-interval 10 \\
        --canny-low 50 --canny-high 150 --ransac-threshold 5 --display

All parameters are configurable via command-line arguments (see --help).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from edge_detection import detect_edges
from ransac import detect_lines
from video_utils import draw_lines, get_video_info, iter_frames


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Road white-line detection from dashcam video using RANSAC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input / output -------------------------------------------------------
    parser.add_argument("input", help="Input video file path.")
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Directory where annotated frames are saved.",
    )

    # --- Frame extraction -----------------------------------------------------
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        metavar="N",
        help="Process every N-th frame (e.g. 10 → frames 0, 10, 20, …).",
    )

    # --- Edge detection -------------------------------------------------------
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        metavar="K",
        help="Gaussian blur kernel size (odd integer ≥ 1).",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=50,
        metavar="T",
        help="Canny lower hysteresis threshold.",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=150,
        metavar="T",
        help="Canny upper hysteresis threshold.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=200,
        metavar="T",
        help=(
            "Brightness threshold (0–255) for isolating white pixels before "
            "edge detection. Set to 0 to disable."
        ),
    )

    # --- Region of interest ---------------------------------------------------
    parser.add_argument(
        "--roi-top",
        type=float,
        default=0.5,
        metavar="R",
        help=(
            "Fraction of the frame height (0–1) to mask out from the top. "
            "0.5 keeps only the bottom half of each frame."
        ),
    )

    # --- RANSAC ---------------------------------------------------------------
    parser.add_argument(
        "--ransac-iterations",
        type=int,
        default=200,
        metavar="N",
        help="Number of RANSAC iterations per line.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        metavar="D",
        help="Maximum perpendicular distance (pixels) to count as an inlier.",
    )
    parser.add_argument(
        "--ransac-min-inliers",
        type=int,
        default=50,
        metavar="N",
        help="Minimum inlier count required to accept a line hypothesis.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=2,
        metavar="N",
        help="Maximum number of lane lines to detect per frame.",
    )

    # --- Visualisation --------------------------------------------------------
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show each annotated frame in an OpenCV window (press 'q' to quit).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated frames to disk.",
    )

    return parser


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_frame(frame, args):
    """Run the full edge → RANSAC pipeline on a single BGR frame.

    Returns
    -------
    annotated : np.ndarray
        Frame with detected lines drawn in red and detected edges shown in green.
    lines : list of (a, b, c)
        Detected line coefficients.
    """
    # 1. Edge detection
    edges, edge_points = detect_edges(
        frame,
        blur_kernel=args.blur_kernel,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        roi_top_ratio=args.roi_top,
        white_threshold=args.white_threshold,
    )

    # 2. RANSAC line detection
    lines = []
    if len(edge_points) >= args.ransac_min_inliers:
        lines, _ = detect_lines(
            edge_points,
            max_lines=args.max_lines,
            n_iterations=args.ransac_iterations,
            threshold=args.ransac_threshold,
            min_inliers=args.ransac_min_inliers,
        )

    # 3. Annotate
    annotated = frame.copy()

    # Overlay edges in green (single-channel → 3-channel)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    green_mask = np.zeros_like(edge_bgr)
    green_mask[:, :, 1] = edges  # green channel
    annotated = cv2.addWeighted(annotated, 1.0, green_mask, 0.5, 0)

    # Draw detected lines in red
    annotated = draw_lines(annotated, lines, color=(0, 0, 255), thickness=3)

    # Overlay parameter text (small, top-left corner of the ROI area)
    h = frame.shape[0]
    roi_y = int(h * args.roi_top)
    info_lines = [
        f"Frame interval: {args.frame_interval}",
        f"Canny: {args.canny_low}/{args.canny_high}",
        f"RANSAC iters: {args.ransac_iterations}  thr: {args.ransac_threshold}",
        f"Lines detected: {len(lines)}",
    ]
    for i, text in enumerate(info_lines):
        cv2.putText(
            annotated,
            text,
            (10, roi_y + 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Print basic video info
    info = get_video_info(str(input_path))
    print(
        f"Video: {input_path.name}  "
        f"{info['width']}x{info['height']}  "
        f"{info['fps']:.1f} fps  "
        f"{info['total_frames']} frames"
    )
    print(f"Processing every {args.frame_interval}-th frame …")

    processed = 0
    for frame_idx, frame in iter_frames(str(input_path), frame_interval=args.frame_interval):
        annotated, lines = process_frame(frame, args)

        if not args.no_save:
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), annotated)

        if args.display:
            cv2.imshow("Road Line RANSAC", annotated)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("Interrupted by user.")
                break

        processed += 1
        print(
            f"  frame {frame_idx:6d}  →  {len(lines)} line(s) detected",
            flush=True,
        )

    if args.display:
        cv2.destroyAllWindows()

    print(f"\nDone. {processed} frame(s) processed.")
    if not args.no_save:
        print(f"Annotated frames saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
