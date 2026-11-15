"""Optional parameter EMA: local shards for resume, full weights for deployment."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import torch


class ParameterEMA:
    def __init__(self, model, decay):
        if isinstance(decay, bool) or not math.isfinite(float(decay)) or not 0 <= decay < 1:
            raise ValueError("train.ema_decay must be a finite number in [0,1), or null.")
        self.decay = float(decay)
        self.parameters = dict(model.named_parameters())
        self.shadow = {k: p.detach().clone() for k, p in self.parameters.items()}
        self.updates = 0

    @torch.no_grad()
    def update(self):
        for name, parameter in self.parameters.items():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)
        self.updates += 1

    @contextmanager
    def apply(self):
        with torch.no_grad():
            original = {k: p.detach().clone() for k, p in self.parameters.items()}
            try:
                for key, parameter in self.parameters.items():
                    parameter.copy_(self.shadow[key])
                yield
            finally:
                for key, parameter in self.parameters.items():
                    parameter.copy_(original[key])

    @staticmethod
    def _local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    def save(self, resume_dir, rank):
        directory = Path(resume_dir) / "ema"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "decay": self.decay,
                "updates": self.updates,
                "shadow": {k: self._local(t).detach().cpu() for k, t in self.shadow.items()},
            },
            directory / f"rank_{rank:05d}.pt",
        )

    @torch.no_grad()
    def load(self, checkpoint_dir, rank):
        payload = torch.load(
            Path(checkpoint_dir) / "resume/ema" / f"rank_{rank:05d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if payload["decay"] != self.decay or set(payload["shadow"]) != set(self.shadow):
            raise ValueError("EMA decay or parameter layout differs from the resume checkpoint.")
        for key, value in payload["shadow"].items():
            destination = self._local(self.shadow[key])
            if destination.shape != value.shape or destination.dtype != value.dtype:
                raise ValueError(f"EMA shard shape/dtype changed for {key}.")
            destination.copy_(value)
        self.updates = int(payload["updates"])
