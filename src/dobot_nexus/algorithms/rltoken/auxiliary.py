"""Optional expert-gradient and terminal action-ranking objectives."""
import torch
from torch.nn import functional as F


def direction_loss(critic, actor, batch):
    eligible = batch["success"] & batch["supervision_mask"].any(-1) & ~batch["truncated"]
    if not eligible.any():
        return batch["executed"].sum() * 0
    ref = batch["reference_condition"][eligible]
    z, state = batch["latent"][eligible], batch["state"][eligible]
    with torch.no_grad():
        actions = actor(z, state, ref, deterministic=True)
        u0 = (actions - ref) / actor.rho
        expert = torch.where(batch["supervision_mask"][eligible, :, None],
                             batch["supervision"][eligible], actions)
        direction = ((actor.project(expert, ref) - actions) / actor.rho).flatten(1)
        valid = direction.norm(dim=1) > 1e-7
    if not valid.any():
        return actions.sum() * 0
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        u = u0.detach().requires_grad_(True)
        q = critic(z, state, ref + u * actor.rho, ref)
        gradients = [torch.autograd.grad(q[:, i].sum(), u, create_graph=create_graph,
                                         retain_graph=True)[0].flatten(1) for i in range(2)]
        return torch.stack([1 - F.cosine_similarity(g[valid], direction[valid], dim=1, eps=1e-8)
                            for g in gradients]).mean()


def terminal_rank_loss(critic, actor, batch, settings, generator, *, base_only=False):
    ref, action = batch["reference_condition"], actor.project(batch["critic_actions"], batch["reference_condition"])
    terminal = batch["success"] & ~batch["bootstrap"] & ~batch["truncated"]
    before_terminal = batch["success"] & batch["bootstrap"] & ~batch["truncated"]
    support = (before_terminal | batch["failure"]) & ~batch["truncated"]
    zero = action.sum() * 0
    groups = []
    def score(mask, actions):
        q = (critic.base_values(batch["latent"][mask], batch["state"][mask], actions)
             if base_only else critic(batch["latent"][mask], batch["state"][mask], actions, ref[mask]))
        groups.append(q)
        return q
    def random_actions(reference):
        noise = torch.rand(reference.shape, device=ref.device, generator=generator) * 2 - 1
        return actor.project(reference + noise * actor.rho, reference)
    def pair(better, worse):
        gap = better - worse
        return (F.softplus(-gap / settings.rank_temperature) if settings.rank_loss_type == "softplus"
                else (settings.rank_margin - gap).relu()).mean()
    success_loss = zero
    if terminal.any():
        good, reference = action[terminal], ref[terminal]
        epsilon = torch.randn(good.shape, device=good.device, generator=generator) * settings.rank_noise_std
        random = random_actions(reference)
        if support.any():
            indices = torch.randint(int(support.sum()), (len(good),), device=good.device, generator=generator)
            mismatch = reference + (action[support] - ref[support])[indices]
        else:
            mismatch = good.roll(1, 0) if len(good) > 1 else random
        scores = [score(terminal, candidate) for candidate in (good,
            actor.project(good + epsilon, reference), actor.project(good + 2 * epsilon, reference),
            random, actor.project(mismatch, reference))]
        losses = [pair(scores[a], scores[b]) for a,b in ((0,1),(1,2),(2,3),(0,4))]
        if before_terminal.any():
            losses.append(pair(scores[0][:, None], score(before_terminal, action[before_terminal])[None]))
        success_loss = torch.stack(losses).mean()
    support_loss = zero
    if settings.rank_failure_weight and support.any():
        support_loss = pair(score(support, action[support]), score(support, random_actions(ref[support])))
    bound = zero
    if groups and settings.rank_q_bound_weight:
        low, high = settings.target_bounds or (0., 1.)
        bound = torch.stack([((low-q).relu().square() + (q-high).relu().square()).mean() for q in groups]).mean()
    return success_loss + settings.rank_failure_weight * support_loss + settings.rank_q_bound_weight * bound
