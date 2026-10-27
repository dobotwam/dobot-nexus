from __future__ import annotations

from contextlib import contextmanager

import torch


@torch.no_grad()
def td_target(rewards, bootstrap, next_values, gamma):
    """next_values contains ONLY the bootstrap rows (terminal rows are never evaluated)."""
    if rewards.ndim != 2 or bootstrap.shape != rewards.shape[:1] or bootstrap.dtype != torch.bool:
        raise ValueError("Expected rewards [B,C] and bool bootstrap [B].")
    if next_values.shape != (int(bootstrap.sum()),):
        raise ValueError("next_values must contain exactly the bootstrap subset.")
    discounts = gamma ** torch.arange(rewards.shape[1], device=rewards.device, dtype=torch.float32)
    result = (rewards.float() * discounts).sum(-1)
    result[bootstrap] += gamma ** rewards.shape[1] * next_values.float()
    return result


def critic_loss(values, target):
    if values.shape != (target.shape[0], 2):
        raise ValueError("Twin Q must return [B,2].")
    return (values.float() - target.detach().float()[:, None]).square().mean()


@contextmanager
def frozen_parameters(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)


def actor_loss(actor, critic, batch, config, *, generator=None):
    actions = actor(
        batch["latent"],
        batch["state"],
        batch["reference_condition"],
        generator=generator,
        dropout=config.reference_dropout,
    )
    with frozen_parameters(critic):
        q = critic(batch["latent"], batch["state"], actions)[:, 0].float().mean()
    bc = (actions.float() - batch["bc_target"].float()).square().mean()
    return -config.q_weight * q + config.bc_weight * bc, {
        "actor_q": q.detach(),
        "actor_bc": bc.detach(),
    }
