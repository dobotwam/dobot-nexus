from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


def recipe_identity(name):
    recipes = {
        "rlt_td3_dual_residual_v1": "dobot_exp18_explicit_frozen_base_v1",
        "rlt_ac_fixed_v1": "rlt_ac_fixed_replace_for_training_v1",
        "rlt_ac_terminal_v2": "rlt_ac_terminal_candidate_completion_v2",
    }
    if name not in recipes:
        raise ValueError(f"Unknown RLToken recipe {name!r}; expected one of {tuple(recipes)}.")
    return recipes[name]


@dataclass(frozen=True)
class HeadConfig:
    chunk_size: int
    hidden_dims: tuple[int, ...] = (256, 256)
    fixed_std: float = 0.1
    residual: dict | None = field(default=None, metadata={"digest_omit_none": True})

    def __post_init__(self):
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not self.hidden_dims or any(type(d) is not int or d <= 0 for d in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if self.residual is not None:
            ResidualConfig(**self.residual)
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
    td3: dict | None = field(default=None, metadata={"digest_omit_none": True})

    def __post_init__(self):
        if self.td3 is not None:
            TD3Config(**self.td3)
            if self.reference_dropout or self.target_update_frequency != 1:
                raise ValueError("TD3 requires full reference and actor-clock target updates.")
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


@dataclass(frozen=True)
class ResidualConfig:
    # Scales are in the provider codec, not physical robot units.
    rho: tuple[float, ...]
    actor_width: int = 128
    critic_width: int = 256
    residual_limit: float = 0.2
    exploration_std: float = 0.0
    exploration_mode: str = "step"

    def __post_init__(self):
        if not self.rho or any(not math.isfinite(x) or x <= 0 for x in self.rho):
            raise ValueError("rho requires positive finite per-axis codec scales.")
        if any(type(value) is not int or value <= 0 for value in (self.actor_width, self.critic_width)):
            raise ValueError("Residual network widths must be positive.")
        if not math.isfinite(self.residual_limit) or self.residual_limit <= 0:
            raise ValueError("residual_limit must be positive.")
        if not math.isfinite(self.exploration_std) or self.exploration_std < 0:
            raise ValueError("exploration_std must be nonnegative.")
        if self.exploration_mode not in {"step", "chunk"}:
            raise ValueError("exploration_mode must be step or chunk.")


@dataclass(frozen=True)
class TD3Config:
    discount_unit: str = "chunk"
    target_noise: float = 0.02
    target_noise_clip: float = 0.05
    target_bounds: tuple[float, float] | None = (0.0, 1.0)
    td_action_view: str = "project"
    q_normalization: str = "gradient"
    grad_clip: float = 1.0
    residual_l2: float = 0.0
    direction_weight: float = 0.0
    rank_weight: float = 0.0
    rank_margin: float = 0.01
    rank_noise_std: float = 0.02
    rank_failure_weight: float = 0.2
    rank_temperature: float = 0.1
    rank_q_bound_weight: float = 0.0
    rank_loss_type: str = "softplus"
    base_mc_weight: float = 1.0

    def __post_init__(self):
        if self.discount_unit not in {"step", "chunk"}:
            raise ValueError("discount_unit must be step or chunk.")
        if self.td_action_view not in {"project", "executed"}:
            raise ValueError("td_action_view must be project or executed.")
        if self.q_normalization not in {"gradient", "none"}:
            raise ValueError("q_normalization must be gradient or none.")
        for name in ("target_noise", "target_noise_clip", "grad_clip", "residual_l2",
                     "direction_weight", "rank_weight", "rank_margin",
                     "base_mc_weight", "rank_noise_std", "rank_failure_weight",
                     "rank_temperature", "rank_q_bound_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.rank_loss_type not in {"softplus", "hinge"} or self.rank_temperature <= 0:
            raise ValueError("RankQ requires softplus/hinge and a positive temperature.")
        if self.grad_clip == 0:
            raise ValueError("grad_clip must be positive.")
        if self.target_bounds is not None and (
            len(self.target_bounds) != 2 or not all(map(math.isfinite, self.target_bounds))
            or self.target_bounds[0] >= self.target_bounds[1]
        ):
            raise ValueError("target_bounds must be an increasing finite pair.")


def validate_recipe(name, heads, learning):
    recipe_identity(name)
    dual = name == "rlt_td3_dual_residual_v1"
    if dual != (heads.residual is not None) or dual != (learning.td3 is not None):
        raise ValueError("Recipe, residual heads and TD3 learning configuration must agree.")
    if dual and learning.critic_actor_ratio != 2:
        raise ValueError("The Dobot recipe requires critic_actor_ratio=2.")
