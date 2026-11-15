"""Dobot Fusion RGB resize and optional min/max state-as-text processing."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.modules.value.distributions import expected_value
from dobot_nexus.modules.value.fusion.processing import (
    FusionProcessor,
    preprocess_image,
    state_tokens,
    validate_state_stats,
)
from dobot_nexus.modules.value.losses import categorical_loss

from ..configuration import CAMERAS
from ..types import FrameKey

__all__ = [
    "ReCapValueProcessor",
    "ValueBatch",
    "ValuePrediction",
    "ValueSample",
    "forward_value",
    "load_state_stats",
    "preprocess_image",
    "state_tokens",
    "validate_state_stats",
]


@dataclass(frozen=True)
class ValueSample:
    key: FrameKey
    images: Mapping[str, np.ndarray]
    prompt: str
    target_value: float
    state: np.ndarray | None = None


@dataclass
class ValueBatch:
    keys: tuple[FrameKey, ...]
    observation: dict
    targets: torch.Tensor

    def pin_memory(self):
        return dataclasses.replace(
            self,
            observation={k: v.pin_memory() for k, v in self.observation.items()},
            targets=self.targets.pin_memory(),
        )


@dataclass
class ValuePrediction:
    values: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor | None


def load_state_stats(dataset_path: Path, state_key: str) -> dict:
    meta = Path(dataset_path) / "meta"
    path = meta / "stats.json"
    if path.is_file():
        stats = json.loads(path.read_text()).get(state_key, {})
    else:
        from dobot_nexus.data.lerobot.metadata import load_jsonl

        path = meta / "episodes_stats.jsonl"
        rows = load_jsonl(path) if path.is_file() else []
        parts = [r.get("stats", {}).get(state_key) for r in rows]
        parts = [p for p in parts if p]
        stats = (
            {
                "min": np.min([p["min"] for p in parts], axis=0).tolist(),
                "max": np.max([p["max"] for p in parts], axis=0).tolist(),
            }
            if parts
            else {}
        )
    if not stats:
        raise ValueError(f"Value state tokens require explicit min/max stats for {state_key}.")
    return validate_state_stats(stats)


class ReCapValueProcessor(FusionProcessor):
    """Bind shared Fusion observations to RECAP frame keys and return targets."""

    def __init__(self, config, *, state_stats=None, tokenizer=None):
        super().__init__(config, camera_names=CAMERAS, state_stats=state_stats, tokenizer=tokenizer)

    def collate(self, samples: Sequence[ValueSample]) -> ValueBatch:
        if not samples:
            raise ValueError("Cannot collate an empty value batch.")
        prompts = []
        for sample in samples:
            self.validate_cameras(sample.images)
            prompts.append(self.format_prompt(sample.prompt, sample.state))
        tokens = self.tokenize(prompts)
        targets = torch.tensor([s.target_value for s in samples], dtype=torch.float32)
        if not torch.isfinite(targets).all() or torch.any((targets < -1) | (targets > 0)):
            raise ValueError("Value targets must be finite normalized returns in [-1,0].")
        return ValueBatch(
            keys=tuple(s.key for s in samples),
            observation={
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                **self.encode_images([s.images for s in samples]),
            },
            targets=targets,
        )

    def to_device(self, batch: ValueBatch, device: torch.device) -> ValueBatch:
        return dataclasses.replace(
            batch,
            observation={k: v.to(device, non_blocking=True) for k, v in batch.observation.items()},
            targets=batch.targets.to(device, non_blocking=True),
        )


def forward_value(model, batch: ValueBatch, *, with_targets: bool) -> ValuePrediction:
    logits = model(batch.observation).float()
    if logits.shape != (len(batch.keys), 201) or not torch.isfinite(logits).all():
        raise ValueError("Fusion must return finite logits[B,201].")
    values = expected_value(logits)
    loss = categorical_loss(logits, batch.targets)[0].mean() if with_targets else None
    return ValuePrediction(values, logits, loss)
