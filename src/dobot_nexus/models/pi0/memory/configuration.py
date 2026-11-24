"""Model-owned FrameSample configuration and Artifact identity."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass as dataclass
from dataclasses import fields as fields
from pathlib import Path

from dobot_nexus.core.config import clone_config
from dobot_nexus.modules.memory.types import MemoryConfig

from ..configuration_pi0 import restore_token_limit

MODEL_TARGET = "dobot_nexus.models.pi0.memory.modeling_memory:Pi0MemoryModel"


def resolve_config(config, *, mode):
    if mode not in {"train", "infer"}:
        raise ValueError("Memory mode must be train or infer.")
    resolved = clone_config(config)
    resolved["task"] = mode
    model = resolved.model
    if model.get("artifact"):
        from dobot_nexus.core.artifacts import load_manifest, resolve_artifact
        from dobot_nexus.core.config import load_config

        root = resolve_artifact(str(model.artifact))
        manifest = load_manifest(root)
        saved = load_config(root / "config_resolved.yaml", expected_task="train")
        if manifest["model"].get("target") != MODEL_TARGET or saved.model.target != MODEL_TARGET:
            raise ValueError("Expected a Pi0MemoryModel Artifact.")
        trained = MemoryConfig.from_dict(saved.model.memory)
        if not trained.vision_sha256:
            raise ValueError("Artifact memory.vision_sha256 is required.")
        spec = asdict(trained)
        incoming = dict(model.get("memory", {}))
        for key, value in incoming.items():
            if key != "vision_checkpoint" and (key not in spec or value != spec[key]):
                raise ValueError(f"Cannot override trained memory.{key}.")
        if manifest["model"].get("memory") != spec:
            raise ValueError("Artifact memory metadata disagrees with its configuration.")
        vision = manifest.get("files", {}).get("memory_vision")
        if vision != "deployment/memory/vision.npz":
            raise ValueError("Artifact does not contain its frozen memory vision encoder.")
        spec["vision_checkpoint"] = str((root / vision).resolve())
        model["memory"] = spec
        restore_token_limit(model, saved.model, manifest["model"])
        trained_source = saved.model.get("memory_reference_sha256")
        if trained_source:
            if manifest["model"].get("memory_reference_sha256") != trained_source:
                raise ValueError(
                    "Artifact reference source metadata disagrees with its configuration."
                )
            if model.get("memory_reference_sha256", trained_source) != trained_source:
                raise ValueError("Cannot override the trained memory reference source identity.")
            model["memory_reference_sha256"] = trained_source
    else:
        model.setdefault("memory", {})
        model.setdefault("max_token_len", 48)
    memory = MemoryConfig.from_dict(model.memory)
    if not memory.vision_checkpoint:
        raise ValueError(
            "model.memory.vision_checkpoint must name the fixed SigLIP .npz checkpoint."
        )
    from .feature_store import content_sha256

    digest = content_sha256(Path(memory.vision_checkpoint))
    if memory.vision_sha256 and digest != memory.vision_sha256:
        raise ValueError("Fixed memory vision checkpoint content does not match vision_sha256.")
    model["memory"] = {**asdict(memory), "vision_sha256": digest}
    resolved.setdefault("data", {})
    if resolved["data"].setdefault("normalization", "zscore") != "zscore":
        raise ValueError("Pi0 FrameSample requires mean/std (zscore) normalization.")
    return resolved, MemoryConfig.from_dict(model.memory)
