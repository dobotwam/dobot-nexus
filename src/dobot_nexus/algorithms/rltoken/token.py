"""Detached prefix compression and causal reconstruction using ordinary Torch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenConfig:
    input_dim: int
    width: int = 256
    heads: int = 8
    layers: int = 2
    max_tokens: int = 2048
    dropout: float = 0.0
    seed: int = 0

    def __post_init__(self):
        for name in ("input_dim", "width", "heads", "layers", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Token {name} must be a positive integer.")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Token width must divide heads; dropout must be in [0,1).")


class TokenEncoder(nn.Module):
    def __init__(self, config: TokenConfig):
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.projection = nn.Linear(config.input_dim, config.width)
            self.position = nn.Parameter(torch.randn(1, config.max_tokens, config.width) * 0.02)
            self.token = nn.Parameter(torch.randn(1, 1, config.width) * 0.02)
            layer = nn.TransformerEncoderLayer(
                config.width,
                config.heads,
                config.width * 4,
                config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, config.layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(config.width)

    def forward(self, prefix, valid):
        _validate_prefix(prefix, valid, self.config)
        batch, length, _ = prefix.shape
        # Invalid slots may contain NaNs in upstream padding. Never feed those to attention.
        prefix = torch.where(valid[..., None], prefix.detach(), 0).to(self.projection.weight.dtype)
        inputs = self.projection(prefix) + self.position[:, :length]
        inputs = torch.cat((inputs, self.token.expand(batch, -1, -1)), dim=1)
        padding = torch.cat(
            (~valid, torch.zeros(batch, 1, device=valid.device, dtype=torch.bool)), 1
        )
        return self.norm(self.transformer(inputs, src_key_padding_mask=padding)[:, -1])


class TokenDecoder(nn.Module):
    def __init__(self, config: TokenConfig):
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed + 1)
            self.projection = nn.Linear(config.input_dim, config.width)
            self.position = nn.Parameter(torch.randn(1, config.max_tokens, config.width) * 0.02)
            self.start = nn.Parameter(torch.zeros(1, 1, config.width))
            layer = nn.TransformerDecoderLayer(
                config.width,
                config.heads,
                config.width * 4,
                config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerDecoder(layer, config.layers)
            self.output = nn.Linear(config.width, config.input_dim)

    def forward(self, latent, prefix, valid):
        _validate_prefix(prefix, valid, self.config)
        batch, length, _ = prefix.shape
        teacher = torch.where(valid[..., None], prefix.detach(), 0).to(self.projection.weight.dtype)
        shifted = torch.cat((self.start.expand(batch, -1, -1), self.projection(teacher[:, :-1])), 1)
        padding = torch.cat(
            (torch.zeros(batch, 1, device=valid.device, dtype=torch.bool), ~valid[:, :-1]), 1
        )
        causal = torch.ones(length, length, device=valid.device, dtype=torch.bool).triu(1)
        hidden = self.transformer(
            shifted + self.position[:, :length],
            latent[:, None],
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
        )
        return self.output(hidden)


def _validate_prefix(prefix, valid, config):
    if prefix.ndim != 3 or prefix.shape[-1] != config.input_dim:
        raise ValueError("prefix must have shape [B,L,input_dim].")
    if not 0 < prefix.shape[1] <= config.max_tokens:
        raise ValueError("prefix exceeds Token position capacity or is empty.")
    if valid.shape != prefix.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("prefix valid mask must be bool [B,L].")
    if not torch.isfinite(prefix[valid]).all():
        raise ValueError("Valid prefix tokens must be finite.")


def reconstruction_loss(prediction, prefix, valid):
    if (
        prediction.shape != prefix.shape
        or valid.shape != prefix.shape[:2]
        or valid.dtype != torch.bool
    ):
        raise ValueError("Reconstruction shapes or mask disagree.")
    if not valid.any():
        raise ValueError("Cannot reconstruct an entirely empty prefix batch.")
    return (prediction[valid].float() - prefix.detach()[valid].float()).square().mean()
