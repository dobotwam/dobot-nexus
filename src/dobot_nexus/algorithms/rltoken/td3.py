"""Residual TD3 for joint offline prewarm and online updates."""

from __future__ import annotations

import copy
from collections import deque

import torch
from torch.nn import functional as F

from .actor_critic import build_actor, build_critic
from .auxiliary import terminal_rank_loss
from .configuration import TD3Config, WeightScheduleConfig, recipe_identity, validate_recipe
from .identity import content_digest
from .learner import Learner
from .losses import frozen_parameters, td_target
from .routing import routed_actions
from .transitions import residual_batch


def _validate_executed(actions: torch.Tensor) -> None:
    if not torch.isfinite(actions).all() or (actions.abs() > 1).any():
        raise ValueError("Executed behavior actions must be finite and in [-1,1].")


def legal_success_behavior_target(
    executed: torch.Tensor,
    reference: torch.Tensor,
    success: torch.Tensor,
    known: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.isfinite(reference).all():
        raise ValueError("Reference actions must be finite.")
    mask = valid_mask & (success & known)[:, None]
    _validate_executed(executed[mask])
    return torch.where(mask[..., None], executed, reference.clamp(-1, 1)), mask


class TD3Learner(Learner):
    resume_format = "rltoken_td3_resume_v3"

    def __init__(self, spec, heads, config, replay, *, device="cpu", policy=None):
        validate_recipe("rlt_td3_residual_v1", heads, config)
        if replay.capacity < config.warmup or replay.utd != config.updates_per_new_transition:
            raise ValueError("Replay warmup/UTD differs from the learner.")
        self.spec, self.heads, self.config, self.replay = spec, heads, config, replay
        self.settings = TD3Config(**config.td3)
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.actor = build_actor(spec, heads) if policy is None else policy.actor
            self.critic = build_critic(spec, heads) if policy is None else policy.critic
        self.actor.to(self.device).float()
        self.critic.to(self.device).float()
        self.target = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed + 1)
        self.noise_generator = torch.Generator(device=self.device).manual_seed(config.seed + 2)
        self.actor_updates = self.critic_updates = self.actor_revision = 0
        self.update_step = 0
        self.weight_schedule = WeightScheduleConfig(**self.settings.weight_schedule)
        self.pretrained_ready = False
        self.ready_revision = 0
        self.rollout_rng = None
        self.failed = False
        self.prefetched = deque()
        self.fingerprint = content_digest(
            {
                "spec": spec,
                "heads": heads,
                "learning": config,
                "recipe": recipe_identity("rlt_td3_residual_v1"),
            }
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optimizer, lambda _: 1.0
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lambda _: 1.0
        )

    @property
    def gamma_step(self) -> float:
        return self.config.gamma

    def _optimize(self, loss, optimizer, scheduler):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite TD3 loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [
            p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(params, self.settings.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if any(
            not torch.isfinite(p).all() for group in optimizer.param_groups for p in group["params"]
        ):
            raise FloatingPointError("Nonfinite TD3 parameters.")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _polyak(self) -> None:
        for target, current in ((self.target, self.critic), (self.target_actor, self.actor)):
            for dst, src in zip(target.parameters(), current.parameters(), strict=True):
                dst.lerp_(src, self.config.tau)

    @torch.no_grad()
    def target_values(self, batch, *, generator: torch.Generator | None = None) -> torch.Tensor:
        values = torch.empty(0, device=self.device)
        if batch["bootstrap"].any():
            actions = routed_actions(
                self.target_actor,
                batch["next_latent"],
                batch["next_state"],
                batch["next_reference"][:, : self.heads.chunk_size],
                batch["next_actor_enabled"],
                self.heads.chunk_size,
                deterministic=False,
                generator=self.noise_generator if generator is None else generator,
            )
            values = self.target(batch["next_latent"], batch["next_state"], actions).amin(-1)
            if self.settings.target_q_clip is not None:
                values = values.clamp(*self.settings.target_q_clip)
        target = td_target(batch["rewards"], batch["bootstrap"], values, self.gamma_step)
        if self.settings.critic_objective == "bce" and not (
            torch.isfinite(target).all() and (target >= 0).all() and (target <= 1).all()
        ):
            raise ValueError("BCE critic requires unmodified TD targets in [0,1].")
        return target

    def critic_objective(self, batch, target, *, generator=None):
        actions = batch["critic_actions"]
        if not torch.isfinite(actions).all() or (actions.abs() > 1.000001).any():
            raise ValueError("Critic actions must be finite and in [-1,1].")
        args = (batch["latent"], batch["state"], actions)
        expected = target.detach().float()[:, None]
        if self.settings.critic_objective == "bce":
            logits = self.critic(*args, return_logits=True)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite Critic logits.")
            loss = F.binary_cross_entropy_with_logits(logits.float(), expected.expand_as(logits))
            q = logits.detach().sigmoid()
        else:
            q = self.critic(*args)
            loss = F.mse_loss(q.float(), expected.expand_as(q))
        metrics = {"critic_q": float(q.detach().mean())}
        if self.settings.rank_weight:
            rank, rank_metrics = terminal_rank_loss(
                self.critic,
                batch,
                self.settings,
                self.generator if generator is None else generator,
            )
            loss = loss + self.settings.rank_weight * rank
            metrics.update(rank_metrics, rank_loss=float(rank.detach()))
        return loss, metrics

    def actor_objective(self, batch, *, generator=None):
        ref = batch["reference_condition"]
        executed, valid = batch["executed"], batch["valid_mask"]
        success, known = batch["episode_success"], batch["episode_success_known"]
        effective_success = success
        metrics = {}
        if self.settings.success_anchor == "reference_only":
            effective_success = torch.zeros_like(success)
        elif self.settings.success_anchor == "q_gated_success":
            with torch.no_grad():
                _validate_executed(executed[valid])
                ref_legal = ref.clamp(-1, 1)
                candidate_exec = torch.where(valid[..., None], executed, ref_legal)
                q_exec = self.critic(batch["latent"], batch["state"], candidate_exec).amin(-1)
                q_ref = self.critic(batch["latent"], batch["state"], ref_legal).amin(-1)
                if not (torch.isfinite(q_exec).all() and torch.isfinite(q_ref).all()):
                    raise FloatingPointError("Nonfinite success anchor Q values.")
                effective_success = success & known & (q_exec > q_ref)
                metrics.update(
                    anchor_q_gate_selected_fraction=float(effective_success.float().mean()),
                    anchor_q_gate_executed_minus_reference=float((q_exec - q_ref).mean()),
                )
        target, anchor_mask = legal_success_behavior_target(
            executed, ref, effective_success, known, valid
        )
        pi, u = self.actor.forward_with_preclip(
            batch["latent"],
            batch["state"],
            ref,
            deterministic=False,
            generator=self.noise_generator if generator is None else generator,
            apply_action_noise=self.settings.actor_update_action_noise,
        )
        with frozen_parameters(self.critic):
            q_values = self.critic(batch["latent"], batch["state"], pi)
            q_pi = q_values.amin(-1).float().mean()
        pred = u if self.settings.bc_action_space == "preclip" else pi
        bc = (pred.float() - target.float()).square().mean()
        bc_weight, q_weight, weights = self.weight_schedule.resolve(
            self.update_step, self.config.q_weight, self.config.bc_weight
        )
        weighted_q, weighted_bc = -q_weight * q_pi, bc_weight * bc
        metrics.update(
            weights,
            actor_q=float(q_pi.detach()),
            actor_bc=float(bc.detach()),
            weighted_q=float(weighted_q.detach()),
            weighted_bc=float(weighted_bc.detach()),
            success_anchor_fraction=float(anchor_mask.float().mean()),
            episode_outcome_known_fraction=float(known.float().mean()),
            preclip_clip_fraction=float((u.detach().abs() > 1).float().mean()),
            action_ref_abs_mean=float((pi.detach() - ref).abs().mean()),
            q_value_0=float(q_values[:, 0].detach().mean()),
            q_value_1=float(q_values[:, 1].detach().mean()),
        )
        return weighted_q + weighted_bc, metrics

    def train_rows(self, rows):
        self._healthy()
        batch = residual_batch(rows, self.heads.chunk_size, self.device)
        try:
            loss, metrics = self.critic_objective(batch, self.target_values(batch))
            self._optimize(loss, self.critic_optimizer, self.critic_scheduler)
            self.critic_updates += 1
            metrics["critic_loss"] = float(loss.detach())
            next_step = self.update_step + 1
            if next_step % self.config.critic_actor_ratio == 0:
                loss, values = self.actor_objective(batch)
                self._optimize(loss, self.actor_optimizer, self.actor_scheduler)
                self.actor_updates += 1
                self.actor_revision += 1
                metrics.update(values, actor_loss=float(loss.detach()))
            if next_step % self.config.target_update_frequency == 0:
                self._polyak()
            self.update_step = next_step
        except BaseException:
            self.failed = True
            raise
        return metrics

    def update(self, max_updates=1, *, incoming=()):
        self._healthy()
        if type(max_updates) is not int or max_updates < 0:
            raise ValueError("max_updates must be a nonnegative integer.")
        acks = [self.replay.commit(row) for row in incoming]
        completed, metrics = 0, {}
        for _ in range(max_updates):
            if self.replay.budget <= 0 or len(self.replay.rows) < self.config.warmup:
                break
            rows = (
                self.prefetched.popleft()[1]
                if self.prefetched
                else self.replay._sample_views(self.config.batch_size)
            )
            metrics = self.train_rows(rows)
            self.replay.budget -= 1
            completed += 1
        return dict(
            updates=completed,
            update_step=self.update_step,
            critic_updates=self.critic_updates,
            actor_updates=self.actor_updates,
            actor_revision=self.actor_revision,
            ready_revision=self.ready_revision,
            pretrained_ready=self.pretrained_ready,
            budget=self.replay.budget,
            admitted=self.replay.admitted,
            replay_size=len(self.replay.rows),
            acks=acks,
            metrics={
                **metrics,
                "sample_demo_fraction": self.replay.last_sample_demo_fraction,
                "online_admitted": self.replay.online_admitted,
                "target_utd": self.replay.utd,
                "actual_utd": self.critic_updates / max(1, self.replay.online_admitted),
            },
        )

    def _checkpoint_state_dict(self):
        result = super()._checkpoint_state_dict()
        result.update(
            format=self.resume_format,
            target_actor=copy.deepcopy(self.target_actor.state_dict()),
            noise_generator=self.noise_generator.get_state(),
            update_step=self.update_step,
            actor_revision=self.actor_revision,
            ready_revision=self.ready_revision,
            pretrained_ready=self.pretrained_ready,
            rollout_rng=copy.deepcopy(self.rollout_rng),
        )
        return result

    def _load_checkpoint_state_dict(self, state):
        if (
            state.get("format") != self.resume_format
            or state.get("fingerprint") != self.fingerprint
        ):
            raise ValueError("TD3 resume recipe/feature identity mismatch.")
        for name in ("update_step", "actor_revision", "ready_revision"):
            if type(state.get(name)) is not int or state[name] < 0:
                raise ValueError(f"Invalid TD3 resume {name}.")
        if (
            type(state["pretrained_ready"]) is not bool
            or state["ready_revision"] > state["actor_revision"]
            or state["pretrained_ready"] != (state["ready_revision"] > 0)
        ):
            raise ValueError("Inconsistent TD3 resume readiness.")
        rollout_rng = state["rollout_rng"]
        if rollout_rng is not None and not (
            isinstance(rollout_rng, torch.Tensor)
            and rollout_rng.dtype == torch.uint8
            and rollout_rng.ndim == 1
            and rollout_rng.numel() > 0
        ):
            raise ValueError("TD3 rollout_rng must contain a generator state or None.")
        super()._load_checkpoint_state_dict(state)
        self.target_actor.load_state_dict(state["target_actor"], strict=True)
        self.noise_generator.set_state(state["noise_generator"].cpu())
        for name in (
            "update_step",
            "actor_revision",
            "ready_revision",
            "pretrained_ready",
            "rollout_rng",
        ):
            setattr(self, name, copy.deepcopy(state[name]))
