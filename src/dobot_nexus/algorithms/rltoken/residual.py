"""Pre-clip residual chunk policy and independent twin action-value heads."""

from __future__ import annotations

import torch
from torch import nn

from dobot_nexus.core.contracts import RLFeatureSpec

from .configuration import HeadConfig, ResidualConfig


def _td3_mlp(
    input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, *, layer_norm: bool = False
) -> nn.Sequential:
    layers = []
    for width in hidden_dims:
        layers.append(nn.Linear(input_dim, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width, eps=1e-5))
        layers.append(nn.ReLU())
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


def _normal(shape, like: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(shape, device=like.device, dtype=like.dtype, generator=generator)


class ResidualActor(nn.Module):
    def __init__(self, spec: RLFeatureSpec, config: HeadConfig):
        super().__init__()
        self.spec, self.config = spec, config
        self.settings = ResidualConfig(**config.residual)
        if config.chunk_size > spec.reference_horizon:
            raise ValueError("Residual chunk_size exceeds reference_horizon.")
        self.settings.validate_scale(config.chunk_size, spec.action_dim)
        shape = (1, config.chunk_size, spec.action_dim)
        scale = torch.tensor(self.settings.scale, dtype=torch.float32)
        if scale.numel() == config.chunk_size * spec.action_dim:
            scale = scale.reshape(shape)
        self.register_buffer("scale", scale.expand(shape).clone())
        self.latent_dim = self.settings.latent_dim
        actions = config.chunk_size * spec.action_dim
        self.network = _td3_mlp(
            spec.latent_dim + spec.proprio_dim + actions + (self.latent_dim or 0),
            self.settings.hidden_dims,
            actions,
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward_with_preclip(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        dropout: float = 0.0,
        apply_action_noise: bool | None = None,
        latent_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dropout:
            raise ValueError("Residual policy forbids reference dropout.")
        if reference.ndim != 3:
            raise ValueError("reference must have shape [B,H,A].")
        ref = reference[:, : self.config.chunk_size].detach().float()
        batch = ref.shape[0]
        if (
            latent.shape != (batch, self.spec.latent_dim)
            or state.shape != (batch, self.spec.proprio_dim)
            or ref.shape[1:] != self.scale.shape[1:]
        ):
            raise ValueError(
                "Residual policy inputs disagree with batch, feature or chunk dimensions."
            )
        inputs = [latent.float(), state.float(), ref.flatten(1)]
        if self.latent_dim is not None:
            if apply_action_noise:
                raise ValueError("Latent residual actor forbids external output noise.")
            shape = (batch, self.latent_dim)
            if latent_z is None:
                if deterministic and self.settings.eval_latent_mode == "zero":
                    latent_z = ref.new_zeros(shape)
                else:
                    latent_z = _normal(shape, ref, generator)
            if (
                not isinstance(latent_z, torch.Tensor)
                or latent_z.shape != shape
                or not torch.isfinite(latent_z).all()
            ):
                raise ValueError(f"latent_z must be finite with shape {shape}.")
            inputs.append(latent_z.float())
        elif latent_z is not None:
            raise ValueError("Ordinary residual actor does not accept latent_z.")
        raw = self.network(torch.cat(inputs, dim=-1)).reshape_as(ref)
        u = ref + self.scale.to(raw) * raw.tanh()
        noise_on = not deterministic if apply_action_noise is None else bool(apply_action_noise)
        if self.latent_dim is None and noise_on and self.settings.exploration_std:
            u = u + self.settings.exploration_std * _normal(u.shape, u, generator)
        return u.clamp(-1, 1), u

    def forward(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        dropout: float = 0.0,
        apply_action_noise: bool | None = None,
        latent_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_preclip(
            latent,
            state,
            reference,
            deterministic=deterministic,
            generator=generator,
            dropout=dropout,
            apply_action_noise=apply_action_noise,
            latent_z=latent_z,
        )[0]


class ResidualTwinQ(nn.Module):
    def __init__(self, spec: RLFeatureSpec, config: HeadConfig):
        super().__init__()
        self.settings = ResidualConfig(**config.residual)
        width = spec.latent_dim + spec.proprio_dim + config.chunk_size * spec.action_dim
        self.q1 = _td3_mlp(
            width, self.settings.hidden_dims, 1, layer_norm=self.settings.critic_layer_norm
        )
        self.q2 = _td3_mlp(
            width, self.settings.hidden_dims, 1, layer_norm=self.settings.critic_layer_norm
        )

    def forward(self, latent, state, actions, *, return_logits=False) -> torch.Tensor:
        inputs = torch.cat((latent.float(), state.float(), actions.flatten(1).float()), dim=-1)
        logits = torch.cat((self.q1(inputs), self.q2(inputs)), dim=-1)
        if self.settings.critic_output == "sigmoid" and not return_logits:
            return logits.sigmoid()
        return logits
