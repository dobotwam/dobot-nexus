"""Optional tensor contracts for frozen VLA features and executed commands."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RLFeatureSpec:
    latent_dim: int
    proprio_dim: int
    action_dim: int
    reference_horizon: int
    feature_id: str
    action_codec: str
    state_codec: str
    action_units: str
    control_dt: float
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported RL feature schema.")
        for name in ("latent_dim", "proprio_dim", "action_dim", "reference_horizon"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("feature_id", "action_codec", "state_codec", "action_units"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty.")
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite.")
        for name in ("action_low", "action_high"):
            value = tuple(float(x) for x in getattr(self, name))
            object.__setattr__(self, name, value)
            if len(value) != self.action_dim or not all(map(math.isfinite, value)):
                raise ValueError(f"{name} must contain action_dim finite values.")
        if any(a >= b for a, b in zip(self.action_low, self.action_high, strict=True)):
            raise ValueError("Each action_low must be smaller than action_high.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def _tensor(value, shape, name, *, mask=False):
    import torch

    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must be a tensor with shape {tuple(shape)}.")
    if mask:
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must use bool.")
    elif not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating point values.")


@dataclass
class RLFeatureBatch:
    latent: torch.Tensor
    state: torch.Tensor
    reference_actions: torch.Tensor
    reference_mask: torch.Tensor
    feature_id: str
    action_context: dict[str, Any] = field(default_factory=dict)
    # Providers supply complete single-row contexts when batched features will be split.
    # Arbitrary JSON metadata cannot safely be sliced by guessing which lists are batched.
    action_contexts: list[dict[str, Any]] | None = field(
        default=None, metadata={"digest_omit_none": True}
    )

    def validate(self, spec: RLFeatureSpec, *, chunk_size: int | None = None):
        batch = self.latent.shape[0] if self.latent.ndim else 0
        if batch <= 0 or self.feature_id != spec.feature_id:
            raise ValueError("Empty feature batch or mismatched feature_id.")
        _tensor(self.latent, (batch, spec.latent_dim), "latent")
        _tensor(self.state, (batch, spec.proprio_dim), "state")
        _tensor(
            self.reference_actions,
            (batch, spec.reference_horizon, spec.action_dim),
            "reference_actions",
        )
        _tensor(self.reference_mask, (batch, spec.reference_horizon), "reference_mask", mask=True)
        if any(x.requires_grad for x in (self.latent, self.state, self.reference_actions)):
            raise ValueError("RL features must be detached.")
        if chunk_size is not None:
            if not 0 < chunk_size <= spec.reference_horizon:
                raise ValueError("chunk_size must fit reference_horizon.")
            if not self.reference_mask[:, :chunk_size].all():
                raise ValueError("Reference must be valid for the complete action chunk.")
        if (self.reference_actions[self.reference_mask].abs() > 1.000001).any():
            raise ValueError("Reference actions must use the declared [-1,1] RL codec.")
        json.dumps(self.action_context, allow_nan=False)
        if self.action_contexts is not None:
            if len(self.action_contexts) != batch or any(
                not isinstance(context, dict) for context in self.action_contexts
            ):
                raise ValueError("action_contexts must contain one dictionary per observation.")
            json.dumps(self.action_contexts, allow_nan=False)
        return self

    def row(self, index: int):
        if type(index) is not int or not 0 <= index < self.latent.shape[0]:
            raise ValueError("Feature row index is out of range.")
        if self.action_contexts is None:
            if self.latent.shape[0] != 1:
                raise ValueError(
                    "Batched rollout requires provider action_contexts per observation."
                )
            context = self.action_context
        else:
            context = self.action_contexts[index]
        return RLFeatureBatch(
            self.latent[index : index + 1],
            self.state[index : index + 1],
            self.reference_actions[index : index + 1],
            self.reference_mask[index : index + 1],
            self.feature_id,
            context,
        ).cpu_copy()

    def cpu_copy(self):
        import torch

        return RLFeatureBatch(
            *(
                x.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
                for x in (self.latent, self.state, self.reference_actions)
            ),
            self.reference_mask.detach().cpu().contiguous().clone(),
            self.feature_id,
            copy.deepcopy(self.action_context),
            copy.deepcopy(self.action_contexts),
        )


@dataclass
class EncodedActionBatch:
    actions: torch.Tensor
    executed_mask: torch.Tensor
    encodable_mask: torch.Tensor
    action_codec: str

    def validate(self, spec: RLFeatureSpec):
        if self.actions.ndim != 3 or self.actions.shape[-1] != spec.action_dim:
            raise ValueError("Executed actions must have shape [B,C,A].")
        _tensor(self.actions, self.actions.shape, "actions")
        for name in ("executed_mask", "encodable_mask"):
            _tensor(getattr(self, name), self.actions.shape[:2], name, mask=True)
        if self.actions.requires_grad or self.action_codec != spec.action_codec:
            raise ValueError("Executed actions must be detached and use the matching codec.")
        valid = self.executed_mask & self.encodable_mask
        if (self.actions[valid].abs() > 1.000001).any():
            raise ValueError("Encodable actions must be within [-1,1].")
        return self
