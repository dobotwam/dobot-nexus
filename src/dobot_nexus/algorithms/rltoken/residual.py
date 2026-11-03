"""Explicit residual policy and Q decomposition (algorithm-owned, model independent)."""
from __future__ import annotations

import torch
from torch import nn

from .configuration import ResidualConfig


def _network(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.ReLU(),
                         nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(),
                         nn.Linear(width, 1))


class ResidualActor(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        self.spec, self.config = spec, config
        self.settings = ResidualConfig(**config.residual)
        if len(self.settings.rho) != spec.action_dim or config.chunk_size > spec.reference_horizon:
            raise ValueError("Residual scales/chunk must match the feature spec.")
        self.register_buffer("rho", torch.tensor(self.settings.rho).reshape(1, 1, -1))
        width = self.settings.actor_width
        self.context = nn.Sequential(nn.Linear(spec.latent_dim + spec.proprio_dim, width),
                                     nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.sequence = nn.LSTM(width + spec.action_dim, width, batch_first=True)
        self.head = nn.Linear(width, spec.action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def project(self, actions, reference):
        ref = reference[:, :self.config.chunk_size]
        return (ref + (actions - ref).clamp(-self.rho, self.rho)).clamp(-1, 1)

    def forward(self, latent, state, reference, *, deterministic=False, generator=None, dropout=0):
        if dropout:
            raise ValueError("Residual policy requires an unmodified reference.")
        ref = reference[:, :self.config.chunk_size].float()
        context = self.context(torch.cat((latent, state), -1).float())
        hidden, _ = self.sequence(torch.cat((context[:, None].expand(-1, ref.shape[1], -1), ref), -1))
        delta = self.rho * self.head(hidden).tanh()
        if not deterministic and self.settings.exploration_std:
            shape = (ref.shape[0], 1 if self.settings.exploration_mode == "chunk" else ref.shape[1], ref.shape[2])
            delta = delta + self.settings.exploration_std * torch.randn(
                shape, device=ref.device, dtype=ref.dtype, generator=generator)
        return self.project(ref + delta, ref)


class ExplicitResidualQ(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        self.settings = ResidualConfig(**config.residual)
        if len(self.settings.rho) != spec.action_dim:
            raise ValueError("Critic residual scales must match action dimensions.")
        self.chunk_size = config.chunk_size
        self.register_buffer("rho", torch.tensor(self.settings.rho).reshape(1, 1, -1))
        inputs = spec.latent_dim + spec.proprio_dim + config.chunk_size * spec.action_dim
        width = self.settings.critic_width
        self.base = nn.ModuleList([_network(inputs, width) for _ in range(2)])
        self.offset = nn.ModuleList([_network(inputs + 1, width) for _ in range(2)])
        self.correction = nn.ModuleList([
            _network(inputs + 1 + config.chunk_size * spec.action_dim, width) for _ in range(2)])
        for network in self.offset:
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def freeze_base(self, frozen=True):
        self.base.requires_grad_(not frozen)
        if frozen:
            self.base.eval()

    def components(self, latent, state, actions, reference):
        ref = reference[:, :self.chunk_size]
        inputs = torch.cat((latent, state, ref.flatten(1)), -1).float()
        delta = ((actions - ref) / self.rho).flatten(1)
        bases, residuals = [], []
        for base, offset, correction in zip(self.base, self.offset, self.correction, strict=True):
            b = base(inputs)
            context = torch.cat((inputs, b), -1)
            h = correction(torch.cat((context, delta), -1))
            h0 = correction(torch.cat((context, torch.zeros_like(delta)), -1))
            raw = offset(context) + h - h0
            bases.append(b)
            residuals.append(self.settings.residual_limit * (raw / self.settings.residual_limit).tanh())
        return torch.cat(bases, -1), torch.cat(residuals, -1)

    def forward(self, latent, state, actions, reference):
        base, residual = self.components(latent, state, actions, reference)
        return base + residual

    def base_values(self, latent, state, reference):
        inputs = torch.cat((latent, state, reference[:, :self.chunk_size].flatten(1)), -1).float()
        return torch.cat([base(inputs) for base in self.base], -1)
