"""Terminal success and weak support action ranking in normalized action space."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .configuration import TD3Config


def terminal_rank_loss(
    critic: torch.nn.Module, batch: dict, settings: TD3Config, generator: torch.Generator
) -> tuple[torch.Tensor, dict[str, float]]:
    known = batch["episode_success_known"] & ~batch["truncated"]
    success = batch["episode_success"] & known
    terminal = success & batch["terminated"]
    preterminal = success & ~batch["terminated"]
    support = preterminal | (known & ~batch["episode_success"])
    ref = batch["reference_condition"].clamp(-1, 1)
    valid = batch["valid_mask"][..., None]
    action = torch.where(valid, batch["executed"], ref)

    def score(mask, candidate):
        legal = torch.where(valid[mask], candidate.clamp(-1, 1), ref[mask])
        return critic(batch["latent"][mask], batch["state"][mask], legal)

    def random_actions(like):
        return torch.empty_like(like).uniform_(-1, 1, generator=generator)

    def pair(better, worse):
        return F.softplus(-(better - worse) / settings.rank_temperature).mean()

    terminal_loss = action.new_zeros(())
    support_loss = action.new_zeros(())
    if terminal.any():
        good = action[terminal]
        noise = torch.randn(good.shape, device=good.device, dtype=good.dtype, generator=generator)
        noise = settings.rank_noise_std * noise
        random = random_actions(good)
        if support.any():
            pool = action[support]
            indices = torch.randint(
                len(pool), (len(good),), device=good.device, generator=generator
            )
            mismatch = pool[indices]
        else:
            mismatch = good.roll(1, 0) if len(good) > 1 else random
        recorded, near, far, unrelated, mismatched = [
            score(terminal, candidate)
            for candidate in (good, good + noise, good + 2 * noise, random, mismatch)
        ]
        pairs = [
            pair(recorded, near),
            pair(near, far),
            pair(far, unrelated),
            pair(recorded, mismatched),
        ]
        if preterminal.any():
            pairs.append(pair(recorded[:, None], score(preterminal, action[preterminal])[None]))
        terminal_loss = torch.stack(pairs).mean()
    if settings.rank_failure_weight and support.any():
        support_loss = pair(
            score(support, action[support]), score(support, random_actions(action[support]))
        )
    return terminal_loss + settings.rank_failure_weight * support_loss, {
        "rank_terminal_loss": float(terminal_loss.detach()),
        "rank_support_loss": float(support_loss.detach()),
    }
