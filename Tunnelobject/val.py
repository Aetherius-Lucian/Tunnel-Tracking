"""Evaluate a trained tunnel defect detector."""

import argparse

if __package__:
    from ._paths import DEFAULT_DATA, DEFAULT_RUNS, load_model, require_file
else:
    from _paths import DEFAULT_DATA, DEFAULT_RUNS, load_model, require_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true", help="Use FP16 on a supported GPU")
    parser.add_argument("--project", default=str(DEFAULT_RUNS / "val"))
    args = parser.parse_args()
    data = require_file(args.data, "Dataset YAML")
    return load_model(args.model).val(
        data=str(data), imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, half=args.half, project=args.project,
    )


if __name__ == "__main__":
    main()
