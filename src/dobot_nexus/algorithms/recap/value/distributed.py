"""Distributed execution owned by the offline Pi0.5 Value trainer."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from .configuration import job_device


@dataclass(frozen=True)
class ValueRuntime:
    device: torch.device
    rank: int = 0
    world_size: int = 1

    @property
    def distributed(self):
        return self.world_size > 1

    def sum(self, values):
        result = torch.tensor(list(values), device=self.device, dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(result)
        return result.cpu().tolist()

    def gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.distributed:
            dist.barrier()


@contextmanager
def value_runtime(device_name):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        yield ValueRuntime(job_device(device_name))
        return
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= rank < world_size:
        raise ValueError("RANK must lie within WORLD_SIZE.")
    if device_name == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    elif device_name.startswith("cuda") and torch.cuda.is_available():
        device, backend = torch.device("cuda", local_rank), "nccl"
        torch.cuda.set_device(device)
    else:
        raise ValueError("Distributed Value training requires CPU or available CUDA devices.")
    if dist.is_initialized():
        raise RuntimeError("Value trainer must own its distributed process group.")
    dist.init_process_group(backend, timeout=timedelta(minutes=15))
    try:
        yield ValueRuntime(device, rank, world_size)
    finally:
        dist.destroy_process_group()
