"""Track tunnel defects and save a video annotated with crossing counts."""

import argparse
import math
from pathlib import Path

if __package__:
    from ._paths import DEFAULT_RUNS, DEFAULT_TRACKER, require_file
else:
    from _paths import DEFAULT_RUNS, DEFAULT_TRACKER, require_file


def track_and_count_objects(
    video_path,
    output_video_path,
    model_path,
    conf=0.25,
    iou=0.7,
    tracker=DEFAULT_TRACKER,
    region_points=None,
    region_color=(0, 0, 255),
    counts_position="top-left",
    counts_direction=False,
    counts_text_color=(0, 0, 255),
    device=None,
):
    """Count each tracked object once when it crosses a line or enters a polygon.

    Paths supplied by the caller are relative to the working directory.
    Region points use pixel coordinates; two points define a line and three
    or more define a polygon. The default retains the original experiment line.
    Returns a dictionary of total and per-class IN/OUT counts.
    """
    video_path = require_file(video_path, "Input video")
    model_path = require_file(model_path, "Model weights")
    tracker = require_file(tracker, "Tracker configuration")
    output_path = Path(output_video_path).expanduser().resolve()
    if output_path == video_path:
        raise ValueError("Input and output videos must have different paths.")
    if output_path.suffix.lower() not in {".mp4", ".avi"}:
        raise ValueError("Output video must use .mp4 or .avi.")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf and iou must be between 0 and 1.")
    if region_points is None:
        region_points = [(719, -10), (719, 1200)]
    if len(region_points) < 2 or any(
        len(point) != 2 or not all(math.isfinite(value) for value in point)
        for point in region_points
    ):
        raise ValueError("A region needs at least two finite (x, y) points.")
    if len(set(tuple(point) for point in region_points)) != len(region_points):
        raise ValueError("Region points must be distinct.")

    import cv2
    from tunnel_rtdetr import solutions

    cap = cv2.VideoCapture(str(video_path))
    video_writer = None
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open input video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input video has invalid dimensions or frame rate.")
        counter = solutions.ObjectCounter(
            show=False,
            region=region_points,
            model=str(model_path),
            conf=conf,
            iou=iou,
            tracker=str(tracker),
            region_color=region_color,
            counts_position=counts_position,
            counts_direction=counts_direction,
            counts_text_color=counts_text_color,
            device=device,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = "mp4v" if output_path.suffix.lower() == ".mp4" else "MJPG"
        video_writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if not video_writer.isOpened():
            raise OSError(f"Cannot create output video: {output_path}")
        frames = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            results = counter(frame)
            video_writer.write(results.plot_im)
            frames += 1
        if frames == 0:
            raise ValueError("Input video contains no decodable frames.")
        return {
            "frames": frames,
            "in_count": counter.in_count,
            "out_count": counter.out_count,
            "classwise_count": {key: dict(value) for key, value in counter.classwise_count.items()},
            "output": str(output_path),
        }
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", required=True, help="Trained .pt weights from this repository")
    parser.add_argument("--output", default=str(DEFAULT_RUNS / "count" / "output.mp4"))
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    parser.add_argument("--conf", type=float, default=0.75)
    parser.add_argument("--iou", type=float, default=0.2)
    parser.add_argument("--device", default=None, help="cpu or CUDA device, e.g. 0")
    parser.add_argument(
        "--region", type=float, nargs="+", default=[719, -10, 719, 1200],
        metavar="COORD", help="x1 y1 x2 y2 [x3 y3 ...] in pixels",
    )
    parser.add_argument("--counts-direction", action="store_true", help="Show separate IN and OUT counts")
    parser.add_argument(
        "--counts-position", choices=("top-left", "top-right", "bottom-left", "bottom-right"),
        default="top-left",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(args.region) < 4 or len(args.region) % 2:
        parser.error("--region requires at least two x y pairs.")
    result = track_and_count_objects(
        args.source, args.output, args.model,
        conf=args.conf, iou=args.iou, tracker=args.tracker,
        region_points=list(zip(args.region[::2], args.region[1::2])),
        counts_position=args.counts_position, counts_direction=args.counts_direction,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
