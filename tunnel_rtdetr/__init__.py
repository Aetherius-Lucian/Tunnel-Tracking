
__version__ = "8.4.12"

import importlib
import os
from typing import TYPE_CHECKING

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from tunnel_rtdetr.utils import SETTINGS
from tunnel_rtdetr.utils.checks import check_rtdetr as checks
from tunnel_rtdetr.utils.downloads import download

settings = SETTINGS

MODELS = ("RTDETR",)

__all__ = (
    "__version__",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from tunnel_rtdetr.models import RTDETR  # noqa


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("tunnel_rtdetr.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
