from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def recipe_identity(name):
    recipes = {
        "rlt_ac_fixed_v1": "rlt_ac_fixed_replace_for_training_v1",
    }
    if name not in recipes:
        raise ValueError(f"Unknown RLToken recipe {name!r}; expected one of {tuple(recipes)}.")
    return recipes[name]


@dataclass(frozen=True)
class HeadConfig:
    chunk_size: int
    hidden_dims: tuple[int, ...] = (256, 256)
    fixed_std: float = 0.1

    def __post_init__(self):
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not self.hidden_dims or any(type(d) is not int or d <= 0 for d in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if not math.isfinite(self.fixed_std) or self.fixed_std <= 0:
            raise ValueError("fixed_std must be positive and finite.")


@dataclass(frozen=True)
class LearningConfig:
    gamma: float = 0.99
    tau: float = 0.005
    q_weight: float = 1.0
    bc_weight: float = 1.0
    reference_dropout: float = 0.0
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003
    critic_actor_ratio: int = 1
    target_update_frequency: int = 1
    batch_size: int = 256
    warmup: int = 256
    updates_per_new_transition: int = 1
    seed: int = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma must be in [0,1]; tau in (0,1].")
        if not 0 <= self.reference_dropout <= 1:
            raise ValueError("reference_dropout must be in [0,1].")
        if min(self.q_weight, self.bc_weight) < 0 or self.q_weight + self.bc_weight <= 0:
            raise ValueError("Loss weights must be nonnegative and not both zero.")
        if min(self.actor_lr, self.critic_lr) <= 0:
            raise ValueError("Learning rates must be positive.")
        for name in ("critic_actor_ratio", "target_update_frequency", "batch_size", "warmup"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.updates_per_new_transition) is not int or self.updates_per_new_transition < 0:
            raise ValueError("updates_per_new_transition must be a nonnegative integer.")
