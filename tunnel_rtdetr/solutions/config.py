
from __future__ import annotations

from dataclasses import dataclass
from typing import Any



@dataclass
class SolutionConfig:
    """Configuration for RT-DETR tracking and object counting."""

    source: str | None = None
    model: str | None = None
    classes: list[int] | None = None
    show_conf: bool = True
    show_labels: bool = True
    region: list[tuple[int, int]] | None = None
    region_color: tuple[int, int, int] = (104, 0, 123)
    show_in: bool = True
    show_out: bool = True
    counts_position: str = "top-right"
    counts_direction: bool = True
    counts_text_color: tuple[int, int, int] = (104, 31, 17)
    line_width: int = 2
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: str | None = None
    max_det: int = 300
    half: bool = False
    tracker: str = "bytetrack.yaml"
    verbose: bool = True

    def update(self, **kwargs: Any):
        """Update configuration parameters with new values provided as keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                url = "https://docs.ultralytics.com/solutions/#solutions-arguments"
                raise ValueError(f"{key} is not a valid solution argument, see {url}")

        return self
