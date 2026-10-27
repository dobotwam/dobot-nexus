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
        if self.codec_id != "absolute_affine_v1":
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

    def decode_vla_actions(self, actions, *, aux):
        return super().decode_actions(actions, aux=aux)

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
