"""Memory checkpoint loading and frozen Torch deployment assets."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..training.checkpoint import load_core_state_dict


def is_memory_key(key):
    return key.startswith("memory_encoder.") or ".memory_modulation." in key


def load_memory_checkpoint(core, checkpoint, *, allow_initialization=False):
    path, state = load_core_state_dict(checkpoint)
    reference = core.state_dict()
    missing, extra = set(reference) - set(state), set(state) - set(reference)
    fresh = {key for key in reference if is_memory_key(key)}
    initialize = bool(missing) and allow_initialization and missing == fresh
    if extra or (missing and not initialize):
        raise ValueError(
            f"Incomplete memory checkpoint: missing={sorted(missing)[:8]}, "
            f"extra={sorted(extra)[:8]}"
        )
    for key, value in state.items():
        if value.shape != reference[key].shape:
            raise ValueError(f"Memory checkpoint shape mismatch: {key}")
    if initialize:
        modules = [core.memory_encoder, *(b.memory_modulation for b in core.llm.layers)]
        for module in modules:
            module.to_empty(device="cpu")
            module.reset_parameters()
        initialized = core.state_dict()
        state.update({key: initialized[key] for key in fresh})
    core.load_state_dict(state, strict=True, assign=True)
    return path


def export_memory_artifacts(artifact_dir, *, memory_config):
    """Bundle the frozen history encoder; policy weights are already saved by Nexus."""
    from .feature_store import content_sha256

    root = Path(artifact_dir)
    vision = root / "deployment/memory/vision.npz"
    vision.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=vision.parent) as staging:
        copy = Path(staging) / vision.name
        shutil.copyfile(memory_config.vision_checkpoint, copy)
        if content_sha256(copy) != memory_config.vision_sha256:
            raise ValueError("Fixed memory encoder changed during training.")
        copy.replace(vision)
    return {"memory_vision": vision.relative_to(root).as_posix()}
