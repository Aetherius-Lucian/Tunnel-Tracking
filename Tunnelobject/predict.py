"""Save predictions from a trained tunnel defect detector."""

import argparse

if __package__:
    from ._paths import DEFAULT_RUNS, load_model
else:
    from _paths import DEFAULT_RUNS, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="Image, video or image directory")
    parser.add_argument("--conf", type=float, default=0.75)
    parser.add_argument("--iou", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--project", default=str(DEFAULT_RUNS / "predict"))
    args = parser.parse_args()
    return load_model(args.model).predict(
        source=args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, half=args.half, save=True, project=args.project,
    )


if __name__ == "__main__":
    main()
