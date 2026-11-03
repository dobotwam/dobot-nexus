"""Deterministic residual TD3 and its offline training phases."""
from __future__ import annotations

import copy
from collections import deque

import torch

from .auxiliary import direction_loss, terminal_rank_loss
from .actor_critic import build_actor, build_critic
from .configuration import TD3Config, validate_recipe
from .identity import content_digest
from .learner import Learner
from .losses import frozen_parameters, td_target
from .transitions import residual_batch
from .routing import routed_actions


class TD3Learner(Learner):
    phases = {"actor_bc", "base_critic", "residual_critic", "offline_joint", "online"}
    resume_format = "rltoken_td3_resume_v2"

    def __init__(self, spec, heads, config, replay, *, device="cpu", policy=None):
        validate_recipe("rlt_td3_dual_residual_v1", heads, config)
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
        self.stage_steps = 0
        self.stage_history = []
        self.pretrained_ready = False
        self.ready_revision = 0
        self.initialized_components = []
        self.rollout_rng = None
        self.failed = False
        self.prefetched = deque()
        self.fingerprint = content_digest({"spec": spec, "heads": heads, "learning": config,
                                           "recipe": "rlt_td3_dual_residual_v1"})
        self.set_phase("online")

    @property
    def gamma_step(self):
        return self.config.gamma ** (1 / self.heads.chunk_size) if self.settings.discount_unit == "chunk" else self.config.gamma

    def set_phase(self, phase, *, reset_targets=True):
        if phase not in self.phases:
            raise ValueError(f"Unknown TD3 phase: {phase}")
        self.barrier()
        self.phase, self.stage_steps = phase, 0
        self.critic.requires_grad_(False)
        if phase == "base_critic":
            self.critic.base.requires_grad_(True)
        elif phase != "actor_bc":
            self.critic.offset.requires_grad_(True)
            self.critic.correction.requires_grad_(True)
        self.base_frozen = phase != "base_critic"
        self.critic.freeze_base(self.base_frozen)
        if reset_targets:
            self.target.load_state_dict(self.critic.state_dict())
            self.target_actor.load_state_dict(self.actor.state_dict())
        self.target.requires_grad_(False).eval()
        self.target_actor.requires_grad_(False).eval()
        # Separate parameter sets make a frozen base ineligible for weight decay/momentum.
        parameters = list(self.critic.base.parameters()) if phase == "base_critic" else [
            *self.critic.offset.parameters(), *self.critic.correction.parameters()]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(parameters, lr=self.config.critic_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lambda _: 1.)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lambda _: 1.)

    def _optimize(self, loss, optimizer, scheduler):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite TD3 loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, self.settings.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for group in optimizer.param_groups for p in group["params"]):
            raise FloatingPointError("Nonfinite TD3 parameters.")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _polyak(self, *, actor=False):
        for name, parameter in self.target.named_parameters():
            if self.base_frozen and name.startswith("base."):
                continue
            parameter.lerp_(dict(self.critic.named_parameters())[name], self.config.tau)
        if actor:
            for target, current in zip(self.target_actor.parameters(), self.actor.parameters(), strict=True):
                target.lerp_(current, self.config.tau)

    @torch.no_grad()
    def target_values(self, batch, *, generator=None):
        values = torch.empty(0, device=self.device)
        if batch["bootstrap"].any():
            ref = batch["next_reference"][:, :self.heads.chunk_size]
            enabled = batch["next_actor_enabled"]
            actions = ref if self.phase == "base_critic" else routed_actions(self.target_actor,
                batch["next_latent"], batch["next_state"], ref, enabled,
                self.heads.chunk_size, deterministic=True)
            noise = torch.randn(actions.shape, device=self.device,
                                generator=self.noise_generator if generator is None else generator)
            noise = (noise * self.settings.target_noise).clamp(-self.settings.target_noise_clip,
                                                                            self.settings.target_noise_clip)
            actions = self.actor.project(actions + noise, ref)
            # Outside the Actor's execution phase, bootstrap the exact frozen VLA action.
            actions = torch.where(enabled[:, None, None], actions, ref)
            values = (self.target.base_values(batch["next_latent"], batch["next_state"], actions)
                      if self.phase == "base_critic" else
                      self.target(batch["next_latent"], batch["next_state"], actions, ref)).amin(-1)
        target = td_target(batch["rewards"], batch["bootstrap"], values, self.gamma_step)
        if self.settings.target_bounds is not None:
            target = target.clamp(*self.settings.target_bounds)
        return target

    def critic_objective(self, batch, target, *, generator=None):
        ref, actions = batch["reference_condition"], batch["critic_actions"]
        projected = self.actor.project(actions, ref)
        metrics = {"td_projection_fraction": float((projected - actions).abs().gt(1e-7).any(-1).float().mean())}
        if self.settings.td_action_view == "project":
            actions = projected
        if self.phase == "base_critic":
            # Base Q is the value of the VLA macro-action under its own bootstrap policy.
            q = self.critic.base_values(batch["latent"], batch["state"], actions)
            loss = (q - target[:, None]).square().mean(0).sum()
            mask = batch["mc_valid"]
            if mask.any():
                mc = (q[mask] - batch["mc_return"][mask, None]).square().mean()
                loss = loss + self.settings.base_mc_weight * mc
                metrics["base_mc_loss"] = float(mc.detach())
        else:
            base, residual = self.critic.components(batch["latent"], batch["state"], actions, ref)
            q = base + residual
            loss = (q - target[:, None]).square().mean(0).sum()
            loss = loss + self.settings.residual_l2 * residual.square().mean()
            if self.settings.direction_weight:
                direction = direction_loss(self.critic, self.actor, batch)
                loss = loss + self.settings.direction_weight * direction
                metrics["direction_loss"] = float(direction.detach())
        if self.settings.rank_weight:
            rank = terminal_rank_loss(self.critic, self.actor, batch, self.settings,
                self.generator if generator is None else generator, base_only=self.phase == "base_critic")
            loss = loss + self.settings.rank_weight * rank
            metrics["rank_loss"] = float(rank.detach())
        metrics["critic_q"] = float(q.detach().mean())
        return loss, metrics

    def actor_objective(self, batch):
        ref = batch["reference_condition"]
        actions = self.actor(batch["latent"], batch["state"], ref, deterministic=True)
        if self.phase == "actor_bc":
            mask = batch["supervision_mask"]
            if not mask.any():
                raise ValueError("Actor BC batch has no valid expert supervision.")
            target = self.actor.project(batch["supervision"], ref)
            loss = (actions[mask] - target[mask]).square().mean()
            return loss, {"actor_bc": float(loss.detach())}
        with frozen_parameters(self.critic):
            q = self.critic(batch["latent"], batch["state"], actions, ref).amin(-1)
            alpha = torch.ones((), device=self.device)
            if self.settings.q_normalization == "gradient":
                gradient = torch.autograd.grad(q.sum(), actions, retain_graph=True)[0]
                alpha = (gradient.flatten(1).norm(dim=-1).mean() + 1e-6).reciprocal().detach()
            regularizer = (actions[batch["valid"]] - ref[batch["valid"]]).square().mean()
            loss = -self.config.q_weight * alpha * q.mean() + self.config.bc_weight * regularizer
        return loss, {"actor_q": float(q.detach().mean()), "actor_bc": float(regularizer.detach()),
                      "q_alpha": float(alpha)}

    def train_rows(self, rows):
        self._healthy()
        batch = residual_batch(rows, self.heads.chunk_size, self.device)
        metrics = {}
        try:
            if self.phase != "actor_bc":
                loss, metrics = self.critic_objective(batch, self.target_values(batch))
                self._optimize(loss, self.critic_optimizer, self.critic_scheduler)
                self.critic_updates += 1
                metrics["critic_loss"] = float(loss.detach())
            train_actor = self.phase == "actor_bc" or (self.phase in {"offline_joint", "online"}
                           and (self.stage_steps + 1) % self.config.critic_actor_ratio == 0)
            if train_actor:
                loss, values = self.actor_objective(batch)
                self._optimize(loss, self.actor_optimizer, self.actor_scheduler)
                self.actor_updates += 1
                self.actor_revision += 1
                metrics.update(values, actor_loss=float(loss.detach()))
            if self.phase in {"base_critic", "residual_critic"}:
                self._polyak()
            elif train_actor and self.phase != "actor_bc":
                self._polyak(actor=True)
            self.stage_steps += 1
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
            rows = self.prefetched.popleft()[1] if self.prefetched else self.replay.sample(self.config.batch_size)
            metrics = self.train_rows(rows)
            self.replay.budget -= 1
            completed += 1
        return dict(updates=completed, critic_updates=self.critic_updates, actor_updates=self.actor_updates,
                    actor_revision=self.actor_revision, ready_revision=self.ready_revision, pretrained_ready=self.pretrained_ready,
                    budget=self.replay.budget, admitted=self.replay.admitted, replay_size=len(self.replay.rows), acks=acks,
                    metrics={**metrics, "sample_demo_fraction": self.replay.last_sample_demo_fraction,
                             "online_admitted": self.replay.online_admitted,
                             "target_utd": self.replay.utd,
                             "actual_utd": self.critic_updates / max(1, self.replay.online_admitted)})

    def state_dict(self):
        result = super().state_dict()
        result.update(format=self.resume_format, target_actor=copy.deepcopy(self.target_actor.state_dict()),
                      noise_generator=self.noise_generator.get_state(), phase=self.phase,
                      stage_steps=self.stage_steps, stage_history=copy.deepcopy(self.stage_history),
                      actor_revision=self.actor_revision, ready_revision=self.ready_revision, pretrained_ready=self.pretrained_ready,
                      base_frozen=self.base_frozen,
                      initialized_components=list(self.initialized_components), rollout_rng=self.rollout_rng)
        return result

    def load_state_dict(self, state):
        if state.get("format") != self.resume_format or state["fingerprint"] != self.fingerprint:
            raise ValueError("TD3 resume recipe/feature identity mismatch.")
        self.set_phase(state["phase"], reset_targets=False)
        if state["base_frozen"] != self.base_frozen:
            raise ValueError("Checkpoint violates phase base freezing.")
        super().load_state_dict(state)
        self.target_actor.load_state_dict(state["target_actor"])
        self.noise_generator.set_state(state["noise_generator"].cpu())
        for name in ("stage_steps", "stage_history", "actor_revision", "pretrained_ready", "ready_revision", "initialized_components", "rollout_rng"):
            setattr(self, name, copy.deepcopy(state[name]))
