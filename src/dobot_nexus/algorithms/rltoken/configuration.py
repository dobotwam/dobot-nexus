from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


def recipe_identity(name: str) -> str:
    recipes = {
        "rlt_td3_residual_v1": "rlt_in_practice_preclip_residual_td3_v1",
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
            if self.reference_dropout != 0.0:
                raise ValueError(
                    "Residual TD3 requires complete reference conditioning; "
                    "reference_dropout must be 0."
                )
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


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}, got {value!r}.")
    return float(value)


@dataclass(frozen=True)
class ResidualConfig:
    scale: float | tuple[float, ...] = 1.0
    hidden_dims: tuple[int, ...] = (256, 256)
    exploration_std: float = 0.1
    latent_dim: int | None = None
    eval_latent_mode: str = "sample"
    critic_layer_norm: bool = False
    critic_output: str = "linear"

    def __post_init__(self):
        if isinstance(self.scale, (list, tuple)):
            if not self.scale:
                raise ValueError("scale must contain at least one value.")
            scale = tuple(_finite("scale", x, positive=True) for x in self.scale)
        else:
            scale = _finite("scale", self.scale, positive=True)
        object.__setattr__(self, "scale", scale)
        widths = tuple(self.hidden_dims)
        if not widths or any(type(x) is not int or x <= 0 for x in widths):
            raise ValueError(f"hidden_dims must contain positive integers, got {widths!r}.")
        object.__setattr__(self, "hidden_dims", widths)
        object.__setattr__(
            self, "exploration_std", _finite("exploration_std", self.exploration_std)
        )
        if self.latent_dim is not None and (
            type(self.latent_dim) is not int or self.latent_dim < 1
        ):
            raise ValueError("latent_dim must be a positive integer.")
        if self.eval_latent_mode not in {"sample", "zero"}:
            raise ValueError("eval_latent_mode must be sample or zero.")
        if type(self.critic_layer_norm) is not bool:
            raise ValueError("critic_layer_norm must be boolean.")
        if self.critic_output not in {"linear", "sigmoid"}:
            raise ValueError("critic_output must be linear or sigmoid.")

    def validate_scale(self, chunk_size: int, action_dim: int) -> None:
        if isinstance(self.scale, tuple) and len(self.scale) not in {
            action_dim,
            chunk_size * action_dim,
        }:
            raise ValueError(
                f"scale requires a scalar or {action_dim} or "
                f"{chunk_size * action_dim} values, got {len(self.scale)}."
            )


@dataclass(frozen=True)
class WeightScheduleConfig:
    enable: bool = True
    warmup_updates: int = 20000
    ramp_updates: int = 50000
    warmup_q_weight: float = 0.05
    warmup_bc_weight: float = 7.0
    online_q_weight: float = 0.5
    online_bc_weight: float = 2.8

    def __post_init__(self):
        if type(self.enable) is not bool:
            raise ValueError("weight_schedule.enable must be boolean.")
        for name in ("warmup_updates", "ramp_updates"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
        for phase in ("warmup", "online"):
            q = _finite(f"{phase}_q_weight", getattr(self, f"{phase}_q_weight"))
            bc = _finite(f"{phase}_bc_weight", getattr(self, f"{phase}_bc_weight"))
            if self.enable and q + bc == 0:
                raise ValueError(f"{phase} Q and BC weights cannot both be zero.")

    def resolve(
        self, update_step: int, default_q: float, default_bc: float
    ) -> tuple[float, float, dict[str, float]]:
        warmup = self.enable and update_step < self.warmup_updates
        progress = 1.0
        bc, q = default_bc, default_q
        if warmup:
            progress = 0.0
            bc, q = self.warmup_bc_weight, self.warmup_q_weight
        elif self.enable:
            if self.ramp_updates:
                progress = min(1.0, (update_step - self.warmup_updates + 1) / self.ramp_updates)
            bc = (1 - progress) * self.warmup_bc_weight + progress * self.online_bc_weight
            q = (1 - progress) * self.warmup_q_weight + progress * self.online_q_weight
        return (
            bc,
            q,
            dict(
                bc_weight=bc,
                q_weight=q,
                actor_weight_schedule_enabled=float(self.enable),
                actor_weight_in_warmup=float(warmup),
                actor_weight_ramp_progress=progress,
            ),
        )


@dataclass(frozen=True)
class TD3Config:
    bc_action_space: str = "preclip"
    success_anchor: str = "known_success_behavior"
    actor_update_action_noise: bool = True
    critic_objective: str = "mse"
    target_q_clip: tuple[float, float] | None = None
    grad_clip: float = 10.0
    weight_schedule: dict = field(default_factory=lambda: {"enable": True})
    rank_weight: float = 0.0
    rank_noise_std: float = 0.025
    rank_failure_weight: float = 0.1
    rank_temperature: float = 1.0

    def __post_init__(self):
        if self.bc_action_space not in {"preclip", "postclip"}:
            raise ValueError("bc_action_space must be preclip or postclip.")
        if self.success_anchor not in {
            "known_success_behavior",
            "q_gated_success",
            "reference_only",
        }:
            raise ValueError(f"Unknown success_anchor {self.success_anchor!r}.")
        if type(self.actor_update_action_noise) is not bool:
            raise ValueError("actor_update_action_noise must be boolean.")
        if self.critic_objective not in {"mse", "bce"}:
            raise ValueError("critic_objective must be mse or bce.")
        if self.target_q_clip is not None:
            clip = self.target_q_clip
            if (
                not isinstance(clip, (tuple, list))
                or len(clip) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in clip)
                or clip[0] >= clip[1]
            ):
                raise ValueError(f"target_q_clip must be an increasing finite pair, got {clip!r}.")
            object.__setattr__(self, "target_q_clip", tuple(map(float, clip)))
        for name in ("rank_weight", "rank_failure_weight"):
            _finite(name, getattr(self, name))
        for name in ("grad_clip", "rank_temperature", "rank_noise_std"):
            _finite(name, getattr(self, name), positive=True)
        if not isinstance(self.weight_schedule, dict):
            raise ValueError("weight_schedule must be a mapping of schedule settings.")
        WeightScheduleConfig(**dict(self.weight_schedule))


def validate_recipe(name: str, heads: HeadConfig, learning: LearningConfig) -> None:
    recipe_identity(name)
    is_td3 = name == "rlt_td3_residual_v1"
    if is_td3 != (heads.residual is not None) or is_td3 != (learning.td3 is not None):
        raise ValueError("Recipe, residual heads and TD3 learning configuration must agree.")
    if is_td3:
        res = ResidualConfig(**heads.residual)
        td3 = TD3Config(**learning.td3)
        if res.latent_dim is not None and (res.exploration_std or td3.actor_update_action_noise):
            raise ValueError(
                "Latent residual requires exploration_std=0 and actor_update_action_noise=false."
            )
        if (res.critic_output == "sigmoid") != (td3.critic_objective == "bce"):
            raise ValueError(
                "Sigmoid critic_output and BCE critic_objective must be used together."
            )
        if td3.critic_objective == "bce" and td3.target_q_clip is not None:
            raise ValueError("BCE requires target_q_clip=None.")
