from __future__ import annotations

import torch
from torch import nn

from dobot_nexus.core.contracts import RLFeatureSpec

from .configuration import HeadConfig


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class Actor(nn.Module):
    def __init__(self, spec: RLFeatureSpec, config: HeadConfig):
        super().__init__()
        if config.chunk_size > spec.reference_horizon:
            raise ValueError("Actor chunk_size exceeds reference horizon.")
        self.config, self.spec = config, spec
        actions = config.chunk_size * spec.action_dim
        self.network = _mlp(
            spec.latent_dim + spec.proprio_dim + actions, config.hidden_dims, actions
        )

    def forward(
        self, latent, state, reference, *, deterministic=False, generator=None, dropout=0.0
    ):
        ref = reference[:, : self.config.chunk_size]
        if dropout:
            keep = (
                torch.rand((ref.shape[0], 1, 1), device=ref.device, generator=generator) >= dropout
            )
            ref = ref * keep
        inputs = torch.cat((ref.flatten(1), latent, state), -1).float()
        mean = self.network(inputs).reshape(-1, self.config.chunk_size, self.spec.action_dim)
        if deterministic:
            return mean.tanh()
        epsilon = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return (mean + self.config.fixed_std * epsilon).tanh()


class TwinQ(nn.Module):
    def __init__(self, spec: RLFeatureSpec, config: HeadConfig):
        super().__init__()
        width = spec.latent_dim + spec.proprio_dim + config.chunk_size * spec.action_dim
        self.q1 = _mlp(width, config.hidden_dims, 1)
        self.q2 = _mlp(width, config.hidden_dims, 1)

    def forward(self, latent, state, actions):
        inputs = torch.cat((latent, state, actions.flatten(1)), -1).float()
        return torch.cat((self.q1(inputs), self.q2(inputs)), -1)


def build_actor(spec, config):
    if config.residual is not None:
        from .residual import ResidualActor
        return ResidualActor(spec, config)
    return Actor(spec, config)


def build_critic(spec, config):
    if config.residual is not None:
        from .residual import ResidualTwinQ

        return ResidualTwinQ(spec, config)
    return TwinQ(spec, config)
