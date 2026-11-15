"""Fusion checkpoints bind weights, task scales and state preprocessing to their data."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.core.config import to_plain_dict
from dobot_nexus.modules.value.fusion import build_value_model
from dobot_nexus.training.resume import capture_rng_state, restore_rng_state

from ..annotations import file_digest, load_annotation_manifest
from ..configuration import RECIPE, SCHEMA_VERSION
from ..types import ReturnScale
from .processing import ReCapValueProcessor


def capture_value_rng_state() -> dict:
    """Encode Nexus RNG state for the existing weights-only value checkpoint format."""
    state = capture_rng_state()
    numpy = state["numpy"]
    state["numpy"] = (numpy[0], list(numpy[1]), *numpy[2:])
    state.setdefault("cuda", [])
    return state


def restore_value_rng_state(state: dict) -> None:
    numpy = state["numpy"]
    restore_rng_state(
        {**state, "numpy": (numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:])}
    )


@dataclass(frozen=True)
class LoadedValueModel:
    model: torch.nn.Module
    processor: ReCapValueProcessor
    scale: ReturnScale
    checkpoint_id: str
    network_config: dict
    dataset_signature: str
    returns_id: str


def processor_config(config, state_stats):
    return {
        "image_fields": dict(config.dataset.image_fields),
        "camera_order": ["base", "left_wrist", "right_wrist"],
        "language_key": config.dataset.language_key,
        "state_key": config.dataset.state_key,
        "state_stats": state_stats,
        "resize": "pillow_bilinear_stretch",
        "augmentation": False,
    }


def save_value_checkpoint(
    path,
    *,
    model,
    config,
    scale,
    step,
    optimizer_state,
    state_stats=None,
    training_state=None,
    replace=False,
):
    path = Path(path)
    manifest = load_annotation_manifest(config.annotation_dir, stage="returns")
    if ReturnScale(**manifest["return_scale"]) != scale:
        raise ValueError("Checkpoint ReturnScale disagrees with training annotations.")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Cannot save non-finite value weights.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recipe": RECIPE,
        "network_config": config.value.architecture(),
        "processor_config": processor_config(config, state_stats),
        "model": state,
        "optimizer": optimizer_state,
        "step": step,
        "return_scale": asdict(scale),
        "config": to_plain_dict(asdict(config)),
        "dataset_signature": manifest["dataset_signature"],
        "returns_id": manifest["files"]["returns.parquet"],
        "training_state": training_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".value-", suffix=".pt", dir=path.parent, delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_value_checkpoint(path, *, device, config) -> LoadedValueModel:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("recipe") != RECIPE:
        raise ValueError(f"Value checkpoint requires schema {SCHEMA_VERSION}, recipe {RECIPE}.")
    if payload.get("network_config") != config.value.architecture():
        raise ValueError("Value checkpoint topology or precision does not match this recipe.")
    stats = payload["processor_config"].get("state_stats")
    if payload["processor_config"] != processor_config(config, stats):
        raise ValueError("Value checkpoint camera/language/state processing differs from this job.")
    if any(
        v.is_floating_point() and not torch.isfinite(v).all() for v in payload["model"].values()
    ):
        raise ValueError("Value checkpoint contains non-finite weights.")
    model = build_value_model(config.value)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device).eval()
    return LoadedValueModel(
        model,
        ReCapValueProcessor(config.value, state_stats=stats),
        ReturnScale(**payload["return_scale"]),
        file_digest(path),
        payload["network_config"],
        payload["dataset_signature"],
        payload["returns_id"],
    )
