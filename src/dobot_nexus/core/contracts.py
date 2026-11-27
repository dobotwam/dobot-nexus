from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dobot_nexus.core.rl_contracts import EncodedActionBatch as EncodedActionBatch
from dobot_nexus.core.rl_contracts import RLFeatureBatch as RLFeatureBatch
from dobot_nexus.core.rl_contracts import RLFeatureSpec as RLFeatureSpec


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


@dataclass
class ObservationBatch:
    """Canonical runtime observation accepted by every policy."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    prompt: list[str]
    env_ids: list[str] | None = None
    timestamps: np.ndarray | None = None
    # Per-observation boolean control signals; their semantics belong to algorithms.
    control: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: ObservationBatch | dict[str, Any]) -> ObservationBatch:
        if isinstance(value, cls):
            observation = value
        elif isinstance(value, dict):
            raw_prompt = value.get("prompt", value.get("prompts"))
            if isinstance(raw_prompt, str):
                raw_prompt = [raw_prompt]
            observation = cls(
                images={
                    key: _as_numpy(item)
                    for key, item in dict(value.get("images", {})).items()
                },
                state=_as_numpy(value.get("state")),
                prompt=list(raw_prompt or []),
                env_ids=list(value["env_ids"]) if value.get("env_ids") is not None else None,
                timestamps=(
                    _as_numpy(value["timestamps"]) if value.get("timestamps") is not None else None
                ),
                control=dict(value.get("control", {})),
            )
        else:
            raise TypeError(f"Observation must be ObservationBatch or dict, got {type(value)!r}.")
        return observation.validate()

    def validate(self) -> ObservationBatch:
        state = _as_numpy(self.state)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim != 2:
            raise ValueError(f"state must have shape [D] or [B,D], got {state.shape}.")
        if not np.issubdtype(state.dtype, np.floating):
            state = state.astype(np.float32)
        state = np.ascontiguousarray(state, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("state contains non-finite values.")
        batch_size = state.shape[0]

        if len(self.prompt) != batch_size:
            raise ValueError(
                "prompt must contain one string per observation: "
                f"{len(self.prompt)} != {batch_size}."
            )
        if any(not isinstance(item, str) or not item.strip() for item in self.prompt):
            raise ValueError("Every prompt must be a non-empty string.")

        images: dict[str, np.ndarray] = {}
        for name, raw_image in self.images.items():
            image = _as_numpy(raw_image)
            if image.ndim == 3:
                image = image[None, ...]
            if image.ndim != 4 or image.shape[-1] != 3:
                raise ValueError(
                    f"images[{name!r}] must be HWC or BHWC RGB, got {image.shape}."
                )
            if image.shape[0] != batch_size:
                raise ValueError(
                    f"images[{name!r}] batch {image.shape[0]} does not match state "
                    f"batch {batch_size}."
                )
            if image.dtype != np.uint8:
                raise TypeError(f"images[{name!r}] must use uint8, got {image.dtype}.")
            images[name] = np.ascontiguousarray(image)
        if not images:
            raise ValueError("At least one camera image is required.")

        if self.env_ids is not None and len(self.env_ids) != batch_size:
            raise ValueError("env_ids must contain one id per observation.")
        control = {}
        for name, raw in self.control.items():
            signal = _as_numpy(raw)
            if not isinstance(name, str) or not name or signal.shape != (batch_size,) or signal.dtype != np.bool_:
                raise ValueError("Observation control signals require a nonempty name and bool [B] values.")
            control[name] = np.ascontiguousarray(signal).copy()
        self.control = control
        self.images = images
        self.state = state
        return self

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])


@dataclass(frozen=True)
class SamplingOptions:
    seed: int | None = None
    noise: Any | None = None
    num_steps: int | None = None
    deterministic: bool = False
    collect_timing: bool = True


@dataclass
class PolicyOutput:
    actions: np.ndarray
    valid_mask: np.ndarray | None = None
    timing: dict[str, float] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = _as_numpy(self.actions)
        if actions.ndim == 2:
            actions = actions[None, ...]
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B,H,A], got {actions.shape}.")
        self.actions = np.ascontiguousarray(actions, dtype=np.float32)
        if not all(self.actions.shape) or not np.isfinite(self.actions).all():
            raise ValueError("actions must have nonempty dimensions and contain only finite values.")
        if self.valid_mask is not None:
            mask = _as_numpy(self.valid_mask)
            if mask.shape != self.actions.shape[:2] or mask.dtype != np.bool_:
                raise ValueError("valid_mask must be boolean with shape [B,H].")
            self.valid_mask = np.ascontiguousarray(mask)


@dataclass(frozen=True)
class PolicySpec:
    model_name: str
    implementation: str
    camera_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    action_horizon: int
    image_size: tuple[int, int] = (224, 224)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "implementation": self.implementation,
            "camera_names": list(self.camera_names),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "image_size": list(self.image_size),
        }


@dataclass
class TrainOutput:
    loss: Any
    metrics: dict[str, Any] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainResult:
    run_dir: str
    checkpoint: str
    global_step: int
    metrics: dict[str, float]
