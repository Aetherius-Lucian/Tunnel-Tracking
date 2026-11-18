"""Paths and model selection shared by the research entry points."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
# Direct execution in PyCharm must use this modified framework, not a pip copy.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = PROJECT_DIR / "model" / "rtdetr.yaml"
DEFAULT_DATA = PROJECT_DIR / "dataset" / "all.yaml"
DEFAULT_TRACKER = PROJECT_DIR / "trackers" / "bytetrack.yaml"
DEFAULT_RUNS = PROJECT_DIR / "runs"


def require_file(path, description):
    """Resolve an input file without silently downloading a replacement."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_model(path):
    """Load a local RT-DETR model configuration or checkpoint."""
    from tunnel_rtdetr import RTDETR
    return RTDETR(str(require_file(path, "RT-DETR model")))
