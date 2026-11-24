"""Model-owned FrameSample configuration and Artifact identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dobot_nexus.core.config import clone_config

from ..configuration_pi0 import restore_token_limit

MODEL_TARGET = "dobot_nexus.models.pi0.memory.modeling_memory:Pi0MemoryModel"


@dataclass(frozen=True)
class MemoryConfig:
    schema_version: int = 1
    budget: int = 512
    window_frames: int | None = None
    max_frames: int = 8192
    camera: str = "base"
    tokens_per_image: int = 16
    image_dim: int = 2048
    position_dim: int = 768
    token_dim: int = 1024
    vision_checkpoint: str = ""
    vision_sha256: str = ""
    preprocessing: str = "openpi_jax_float_resize_v1"

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported memory schema_version.")
        for name in ("budget", "max_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"memory.{name} must be a positive integer.")
        if self.window_frames is not None and (
            type(self.window_frames) is not int or self.window_frames <= 0
        ):
            raise ValueError("memory.window_frames must be null or a positive integer.")
        if (self.tokens_per_image, self.image_dim, self.position_dim, self.token_dim) != (
            16,
            2048,
            768,
            1024,
        ):
            raise ValueError(
                "FrameSample v1 requires 16 tokens, image=2048, position=768, token=1024."
            )
        if self.budget % self.tokens_per_image:
            raise ValueError("memory.budget must be divisible by tokens_per_image.")
        if not self.camera or self.preprocessing != "openpi_jax_float_resize_v1":
            raise ValueError("Specify a memory camera and the supported preprocessing recipe.")
        if self.vision_sha256 and (
            len(self.vision_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.vision_sha256)
        ):
            raise ValueError("memory.vision_sha256 must be a SHA256 content digest.")

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown memory configuration fields: {sorted(unknown)}")
        return cls(**dict(value))

    @property
    def sampled_frames(self):
        return self.budget // self.tokens_per_image


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
