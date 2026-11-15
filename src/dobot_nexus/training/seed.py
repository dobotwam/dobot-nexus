"""Training RNG policy, independent of model operators and inference sampling."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler


def training_seed(base_seed: int, global_rank: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed < 2**32:
        raise ValueError("exp.seed must be an integer in [0, 2**32).")
    if global_rank < 0:
        raise ValueError("global rank must be non-negative.")
    return (base_seed + global_rank) % 2**32


def seed_all(seed: int) -> None:
    seed = training_seed(seed, 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_sampler(data_loader):
    """Find the underlying sampler through standard/Accelerate batch wrappers."""
    batch_sampler = getattr(data_loader, "batch_sampler", None)
    seen = set()
    while batch_sampler is not None and id(batch_sampler) not in seen:
        seen.add(id(batch_sampler))
        nested = getattr(batch_sampler, "batch_sampler", None)
        if nested is not None:
            batch_sampler = nested
            continue
        sampler = getattr(batch_sampler, "sampler", None)
        if sampler is not None:
            return sampler
        break
    return getattr(data_loader, "sampler", None)


def seed_dataloader(data_loader, seed: int) -> None:
    """Keep worker creation and model sampling from advancing the sample order.

    Existing sampler generators keep their state. Custom/iterable loaders own
    their sampling policy; no repository-specific model hook is required.
    """
    if not isinstance(data_loader, DataLoader):
        return
    sampler = data_sampler(data_loader)
    if isinstance(sampler, RandomSampler) and sampler.generator is None:
        sampler.generator = torch.Generator().manual_seed(seed)
    worker_generator = getattr(data_loader, "generator", None)
    if worker_generator is None:
        data_loader.generator = torch.Generator().manual_seed(seed)
    elif worker_generator is getattr(sampler, "generator", None):
        # A fresh iterator draws a worker base seed, whereas persistent worker
        # resets do not. Sharing this generator makes resume change the shuffle.
        data_loader.generator = torch.Generator().set_state(worker_generator.get_state())


def data_generators(data_loader) -> dict[str, torch.Generator]:
    candidates = {
        "loader": getattr(data_loader, "generator", None),
        "sampler": getattr(data_sampler(data_loader), "generator", None),
        "synchronized": getattr(data_loader, "synchronized_generator", None),
    }
    return {key: value for key, value in candidates.items() if isinstance(value, torch.Generator)}


def set_data_epoch(data_loader, epoch: int) -> None:
    set_epoch = getattr(data_loader, "set_epoch", None)
    if set_epoch is None:
        set_epoch = getattr(data_sampler(data_loader), "set_epoch", None)
    if set_epoch is not None:
        set_epoch(epoch)
