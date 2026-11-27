from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict

import torch

from .actor_critic import Actor, TwinQ
from .configuration import recipe_identity
from .identity import content_digest
from .losses import actor_loss, critic_loss, td_target
from .transitions import training_batch
from .routing import routed_actions


class Learner:
    """One ordered update stream; a failed optimizer cycle poisons this session."""

    def __init__(self, spec, heads, config, replay, *, device="cpu", policy=None):
        self.spec, self.heads, self.config, self.replay = spec, heads, config, replay
        if replay.capacity < config.warmup:
            raise ValueError("Replay capacity must be at least warmup.")
        if replay.utd != config.updates_per_new_transition:
            raise ValueError("Replay and learner UTD budgets disagree.")
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.actor = policy.actor if policy is not None else Actor(spec, heads)
            self.critic = policy.critic if policy is not None else TwinQ(spec, heads)
        self.actor = self.actor.to(self.device).float()
        self.critic = self.critic.to(self.device).float()
        self.target = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        # Constant schedule is explicit and has its own successful-step clock.
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optimizer, lambda _: 1.0
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lambda _: 1.0
        )
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed + 1)
        self.critic_updates = self.actor_updates = 0
        self.failed = False
        self.prefetched = deque()
        self.fingerprint = content_digest(
            {
                "spec": spec,
                "heads": heads,
                "learning": {k: v for k, v in asdict(config).items() if k != "td3" or v is not None},
                "recipe": recipe_identity(
                    "rlt_ac_terminal_v2" if replay.allow_terminal_chunks else "rlt_ac_fixed_v1"
                ),
            }
        )

    def _healthy(self):
        if self.failed:
            raise RuntimeError(
                "Learner session failed; restore a complete checkpoint before continuing."
            )

    def prefetch(self, count=1):
        self._healthy()
        if not self.replay.rows:
            return
        for _ in range(count):
            rng = self.replay.generator.get_state().clone()
            self.prefetched.append((rng, self.replay._sample_views(self.config.batch_size)))

    def barrier(self):
        """Discard speculative samples and rewind their RNG in the continuing run too."""
        self._healthy()
        if self.prefetched:
            self.replay.generator.set_state(self.prefetched[0][0])
            self.prefetched.clear()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _step(loss, module, optimizer, scheduler):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite RL loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()
        ):
            raise FloatingPointError("Nonfinite RL gradient.")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in module.parameters()):
            raise FloatingPointError("Nonfinite RL parameters after optimizer step.")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def update(self, max_updates=1, *, incoming=()):
        self._healthy()
        if type(max_updates) is not int or max_updates < 0:
            raise ValueError("max_updates must be a nonnegative integer.")
        # Admission and durable ACK must progress even when budget or warmup is zero.
        acks = [self.replay.commit(row) for row in incoming]
        metrics = {}
        completed = 0
        for _ in range(max_updates):
            if self.replay.budget <= 0 or len(self.replay.rows) < self.config.warmup:
                break
            rows = (
                self.prefetched.popleft()[1]
                if self.prefetched
                else self.replay._sample_views(self.config.batch_size)
            )
            batch = training_batch(rows, self.heads.chunk_size, self.device)
            try:
                with torch.no_grad():
                    next_values = torch.empty(0, device=self.device)
                    if batch["bootstrap"].any():
                        next_actions = routed_actions(self.actor,
                            batch["next_latent"],
                            batch["next_state"],
                            batch["next_reference"],
                            batch["next_actor_enabled"], self.heads.chunk_size,
                            generator=self.generator,
                        )
                        next_values = self.target(
                            batch["next_latent"], batch["next_state"], next_actions
                        ).amin(-1)
                    target = td_target(
                        batch["rewards"], batch["bootstrap"], next_values, self.config.gamma
                    )
                loss = critic_loss(
                    self.critic(batch["latent"], batch["state"], batch["critic_actions"]), target
                )
                self._step(loss, self.critic, self.critic_optimizer, self.critic_scheduler)
                next_clock = self.critic_updates + 1
                metrics["critic_loss"] = float(loss.detach())
                if next_clock % self.config.critic_actor_ratio == 0:
                    loss, actor_metrics = actor_loss(
                        self.actor, self.critic, batch, self.config, generator=self.generator
                    )
                    self._step(loss, self.actor, self.actor_optimizer, self.actor_scheduler)
                    self.actor_updates += 1
                    metrics.update({key: float(value) for key, value in actor_metrics.items()})
                    metrics["actor_loss"] = float(loss.detach())
                if next_clock % self.config.target_update_frequency == 0:
                    with torch.no_grad():
                        for target_param, current in zip(
                            self.target.parameters(), self.critic.parameters(), strict=True
                        ):
                            target_param.lerp_(current.float(), self.config.tau)
                self.critic_updates = next_clock
                self.replay.budget -= 1
                completed += 1
            except BaseException:
                self.failed = True
                raise
        return {
            "updates": completed,
            "critic_updates": self.critic_updates,
            "actor_updates": self.actor_updates,
            "budget": self.replay.budget,
            "admitted": self.replay.admitted,
            "metrics": metrics,
            "acks": acks,
        }

    def state_dict(self):
        result = self._checkpoint_state_dict()
        result["replay"] = copy.deepcopy(result["replay"])
        return result

    def _checkpoint_state_dict(self):
        self.barrier()
        result = {
            "fingerprint": self.fingerprint,
            "critic_updates": self.critic_updates,
            "actor_updates": self.actor_updates,
            "generator": self.generator.get_state(),
            "replay": self.replay._checkpoint_state_dict(),
        }
        for name in (
            "actor",
            "critic",
            "target",
            "actor_optimizer",
            "critic_optimizer",
            "actor_scheduler",
            "critic_scheduler",
        ):
            result[name] = copy.deepcopy(getattr(self, name).state_dict())
        return result

    def load_state_dict(self, state):
        self._load_checkpoint_state_dict(copy.deepcopy(state))

    def _load_checkpoint_state_dict(self, state):
        self._healthy()
        if self.fingerprint != state["fingerprint"]:
            raise ValueError("RLToken learning recipe or feature fingerprint mismatch.")
        self.barrier()
        for name in (
            "actor",
            "critic",
            "target",
            "actor_optimizer",
            "critic_optimizer",
            "actor_scheduler",
            "critic_scheduler",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.critic_updates, self.actor_updates = state["critic_updates"], state["actor_updates"]
        self.generator.set_state(state["generator"].cpu())
        self.replay._load_checkpoint_state_dict(state["replay"])


def create_learner(spec, heads, config, replay, **kwargs):
    if config.td3 is not None:
        from .td3 import TD3Learner
        return TD3Learner(spec, heads, config, replay, **kwargs)
    if heads.residual is not None:
        raise ValueError("Residual heads require TD3 learning configuration.")
    return Learner(spec, heads, config, replay, **kwargs)
