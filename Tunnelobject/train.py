"""Train the MSFW RT-DETR model with the original experiment defaults."""

import argparse
import random

if __package__:
    from ._paths import DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_RUNS, load_model, require_file
else:
    from _paths import DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_RUNS, load_model, require_file


def set_seed(seed=42):
    """Seed the same random sources as the original training script."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local model YAML or .pt weights")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=str(DEFAULT_RUNS / "detect" / "paper3"))
    parser.add_argument("--name", default="rtdetr")
    parser.add_argument("--pretrained", action="store_true", help="Allow the trainer to use pretrained weights")
    return parser


def main():
    args = build_parser().parse_args()
    data = require_file(args.data, "Dataset YAML")
    set_seed(args.seed)
    model = load_model(args.model)
    return model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, optimizer=args.optimizer, patience=args.patience,
        seed=args.seed, device=args.device, project=args.project, name=args.name,
        pretrained=args.pretrained, augment=False,
    )


if __name__ == "__main__":
    main()
