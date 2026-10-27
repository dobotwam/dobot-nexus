from __future__ import annotations

import numpy as np
import torch

from dobot_nexus.core.contracts import EncodedActionBatch, PolicyOutput
from dobot_nexus.models.pi05.data_adapter_pi05 import Pi05DataAdapter


class Pi05RLDataAdapter(Pi05DataAdapter):
    """BC preprocessing with a versioned controller-command RL codec."""

    def __init__(self, config, root_config, observation_type):
        super().__init__(config, root_config, observation_type)
        codec = root_config.model.rltoken["action_codec"]
        self.codec_id = codec.get("name")
        if self.codec_id not in {"absolute_affine_v1", "normalized_joint_delta_v1"}:
            raise ValueError("Unsupported Pi0.5 RLToken action codec.")
        self.low = torch.tensor(codec["low"], dtype=torch.float32)
        self.high = torch.tensor(codec["high"], dtype=torch.float32)
        if self.low.shape != (config.action_dim,) or self.high.shape != self.low.shape:
            raise ValueError("RL action bounds must match public action_dim.")
        if (
            not torch.isfinite(self.low).all()
            or not torch.isfinite(self.high).all()
            or not (self.high > self.low).all()
        ):
            raise ValueError("RL command bounds must be finite and strictly ordered.")
        self.reference_bounds = codec.get("reference_bounds", "reject")
        if self.reference_bounds not in {"reject", "clip"}:
            raise ValueError("action_codec.reference_bounds must be reject or clip.")
        if self.codec_id == "normalized_joint_delta_v1":
            self._validate_joint_delta(codec)
        elif self.reference_bounds != "reject":
            raise ValueError("Absolute-command reference bounds must use reject.")

    def _validate_joint_delta(self, codec):
        if not torch.equal(self.low, -torch.ones_like(self.low)) or not torch.equal(
            self.high, torch.ones_like(self.high)
        ):
            raise ValueError("Normalized joint-delta command bounds must be exactly [-1,1].")
        if codec.get("units") != "normalized_joint_delta_and_gripper":
            raise ValueError("Joint-delta codec requires explicit normalized command units.")
        controller = codec.get("controller", {})
        if (controller.get("mode") != "pd_joint_delta_pos"
                or controller.get("delta_reference") != "current_qpos"):
            raise ValueError("Joint-delta codec requires pd_joint_delta_pos from current_qpos.")
        for low_key, high_key, shape in (
            ("arm_delta_low", "arm_delta_high", (self.config.action_dim - 1,)),
            ("gripper_position_low", "gripper_position_high", ()),
        ):
            try:
                low = np.asarray(controller[low_key], dtype=np.float64)
                high = np.asarray(controller[high_key], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Joint-delta codec requires controller scaling metadata.") from exc
            if (low.shape != shape or high.shape != shape or not np.isfinite(low).all()
                    or not np.isfinite(high).all() or not (high > low).all()):
                raise ValueError("Joint-delta controller bounds have invalid dimensions or values.")
        if self.pipeline.action_transform.kind != "absolute":
            raise ValueError("Already normalized joint deltas must not be differenced again.")

    def decode_vla_actions(self, actions, *, aux):
        output = super().decode_actions(actions, aux=aux)
        if self.reference_bounds == "clip":
            if not np.isfinite(output.actions).all():
                raise ValueError("Cannot clip nonfinite reference actions.")
            return PolicyOutput(
                np.clip(output.actions, -1, 1),
                aux={**output.aux, "reference_bounds": "clip"},
            )
        return output

    def decode_actions(self, actions, *, aux=None):
        value = torch.as_tensor(actions, dtype=torch.float32).detach().cpu()
        if value.ndim != 3 or value.shape[-1] != self.config.action_dim:
            raise ValueError("RL actions must have shape [B,C,action_dim].")
        if not torch.isfinite(value).all() or (value.abs() > 1.000001).any():
            raise ValueError("RL actions must be finite and within [-1,1].")
        commands = self.low + (value + 1) * 0.5 * (self.high - self.low)
        return PolicyOutput(commands.numpy(), aux=dict(aux or {}))

    def encode_executed_actions(self, actions, *, aux=None):
        aux = dict(aux or {})
        if not aux.get("observation_id") or "raw_state" not in aux or "executed_steps" not in aux:
            raise ValueError(
                "RL command encoding requires observation_id, raw_state and executed_steps."
            )
        value = torch.as_tensor(actions, dtype=torch.float32).detach().cpu().clone()
        if value.ndim != 3 or value.shape[-1] != self.config.action_dim:
            raise ValueError("Executed commands must have shape [B,C,action_dim].")
        if np.asarray(aux["raw_state"]).shape != (value.shape[0], self.config.state_dim):
            raise ValueError("Action context raw_state has incorrect dimensions.")
        lengths = torch.as_tensor(aux["executed_steps"])
        if (
            lengths.shape != (value.shape[0],)
            or lengths.is_floating_point()
            or (lengths < 0).any()
            or (lengths > value.shape[1]).any()
        ):
            raise ValueError("executed_steps must contain valid integer lengths.")
        valid = torch.arange(value.shape[1])[None] < lengths[:, None]
        encodable = (
            torch.isfinite(value).all(-1)
            & (value >= self.low).all(-1)
            & (value <= self.high).all(-1)
        )
        encoded = 2 * (value - self.low) / (self.high - self.low) - 1
        # Preserve out-of-domain finite values; do not clip a command into a false valid action.
        encoded = torch.where(torch.isfinite(encoded), encoded, torch.zeros_like(encoded))
        return EncodedActionBatch(encoded, valid, encodable, self.codec_id)
