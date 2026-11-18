"""autobatch. RT-DETR detection workflow."""

from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import torch

from tunnel_rtdetr.utils import DEFAULT_CFG, LOGGER, colorstr
from tunnel_rtdetr.utils.torch_utils import autocast, profile_ops


def check_train_batch_size(
    model: torch.nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    batch: int | float = -1,
    max_num_obj: int = 1,
) -> int:
    """check train batch size. RT-DETR detection workflow."""
    with autocast(enabled=amp):
        return autobatch(
            deepcopy(model).train(), imgsz, fraction=batch if 0.0 < batch < 1.0 else 0.6, max_num_obj=max_num_obj
        )


def autobatch(
    model: torch.nn.Module,
    imgsz: int = 640,
    fraction: float = 0.60,
    batch_size: int = DEFAULT_CFG.batch,
    max_num_obj: int = 1,
) -> int:
    """autobatch. RT-DETR detection workflow."""
    # Check device
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% CUDA memory utilization.")
    device = next(model.parameters()).device  # get model device
    if device.type in {"cpu", "mps"}:
        LOGGER.warning(f"{prefix}intended for CUDA devices, using default batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.warning(f"{prefix}Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]
    ch = model.yaml.get("channels", 3)
    try:
        img = [torch.empty(b, ch, imgsz, imgsz) for b in batch_sizes]
        results = profile_ops(img, model, n=1, device=device, max_num_obj=max_num_obj)

        # Fit a solution
        xy = [
            [x, y[2]]
            for i, (x, y) in enumerate(zip(batch_sizes, results))
            if y  # valid result
            and isinstance(y[2], (int, float))  # is numeric
            and 0 < y[2] < t  # between 0 and GPU limit
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])  # first item or increasing memory
        ]
        fit_x, fit_y = zip(*xy) if xy else ([], [])
        p = np.polyfit(fit_x, fit_y, deg=1)  # first-degree polynomial fit in log space
        b = int((round(f * fraction) - p[1]) / p[0])  # y intercept (optimal batch size)
        if None in results:  # some sizes failed
            i = results.index(None)  # first fail index
            if b >= batch_sizes[i]:  # y intercept above failure point
                b = batch_sizes[max(i - 1, 0)]  # select prior safe point
        if b < 1 or b > 1024:  # b outside of safe range
            LOGGER.warning(f"{prefix}batch={b} outside safe range, using default batch-size {batch_size}.")
            b = batch_size

        fraction = (np.polyval(p, b) + r + a) / t  # predicted fraction
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.warning(f"{prefix}error detected: {e},  using default batch-size {batch_size}.")
        return batch_size
    finally:
        torch.cuda.empty_cache()
