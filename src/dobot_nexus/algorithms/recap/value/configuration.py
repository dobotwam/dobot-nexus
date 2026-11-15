"""RECAP Value job device selection."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def job_device(name: str) -> torch.device:
    """Resolve labeling or local training; distributed training uses value_runtime."""
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This RECAP entry is single-device; launch labeling without torchrun/accelerate. "
            "Use train-value for distributed Value training."
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a GPU environment or device=cpu.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("RECAP value jobs support CPU or one CUDA device.")
    return device
