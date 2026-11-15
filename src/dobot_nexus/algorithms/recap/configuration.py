"""RECAP recipe and offline job configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dobot_nexus.core.config import (
    Config,
    apply_overrides,
    resolve_environment,
)
from dobot_nexus.modules.value.fusion.configuration import ValueModelConfig

from ._validation import check_keys, integer, mapping, nonempty, number, path_value

RECIPE = "dobot_acp_v1"
SCHEMA_VERSION = 2
CAMERAS = ("base", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class ReCapPolicyConfig:
    schema_version: int = SCHEMA_VERSION
    recipe: str = RECIPE
    unconditional_prob: float = 0.3


@dataclass(frozen=True)
class ReCapInferenceConfig:
    condition: str = "base"


@dataclass(frozen=True)
class OutcomeSource:
    kind: str
    field: str | None = None
    path: Path | None = None


@dataclass(frozen=True)
class ReturnConfig:
    c_fail_coef: float = 1.0


@dataclass(frozen=True)
class AdvantageConfig:
    n_step: int = 50
    positive_ratio: float = 0.3
    force_intervention_positive: bool = True


@dataclass(frozen=True)
class ReCapDatasetSpec:
    path: Path
    dataset_id: str
    format_version: str
    image_fields: Mapping[str, str]
    state_key: str
    action_key: str
    language_key: str | None
    fps: int | None
    tolerance_s: float = 1e-4

    def __post_init__(self):
        nonempty(self.dataset_id, "dataset_id")
        if self.format_version != "v2.1":
            raise ValueError("RECAP requires one local LeRobot v2.1 dataset.")
        if set(self.image_fields) != set(CAMERAS):
            raise ValueError(f"RECAP camera mapping must have {CAMERAS}.")
        for value in self.image_fields.values():
            nonempty(value, "image field")
        if len(set(self.image_fields.values())) != len(CAMERAS):
            raise ValueError("RECAP camera fields must be distinct.")
        nonempty(self.state_key, "state field")
        nonempty(self.action_key, "action field")
        if self.language_key is not None:
            nonempty(self.language_key, "language field")
        if self.fps is not None:
            integer(self.fps, "data.fps")
        if number(self.tolerance_s, "data.tolerance_s") <= 0:
            raise ValueError("data.tolerance_s must be positive.")


@dataclass(frozen=True)
class ValueTrainConfig:
    output_dir: Path
    device: str = "cuda:0"
    seed: int = 0
    batch_size: int = 56
    grad_accum_steps: int = 1
    max_steps: int = 20000
    stop_steps: int | None = None
    save_every: int = 2000
    log_every: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    stable_steps: int | None = None
    decay_steps: int = 19000
    decay_lr: float = 2.5e-6
    val_ratio: float = 0.05
    val_episodes: int | None = None
    resume_from: Path | None = None
    gradient_checkpointing: bool = False
    num_workers: int = 4
    cache_files: int = 2


@dataclass(frozen=True)
class LabelingConfig:
    device: str = "cuda:0"
    batch_size: int = 16
    num_workers: int = 0


@dataclass(frozen=True)
class ReCapOfflineConfig:
    schema_version: int
    recipe: str
    dataset: ReCapDatasetSpec
    dataset_type: str
    outcome: OutcomeSource | None
    intervention_field: str | None
    returns: ReturnConfig
    advantage: AdvantageConfig
    annotation_dir: Path
    value: ValueModelConfig
    train: ValueTrainConfig
    labeling: LabelingConfig
    tracking: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for output in (self.annotation_dir, self.train.output_dir):
            if output.resolve().is_relative_to(self.dataset.path.resolve()):
                raise ValueError("RECAP outputs must be outside the immutable source dataset.")


def _schema(version, recipe) -> None:
    if integer(version, "recap schema_version") != SCHEMA_VERSION or recipe != RECIPE:
        raise ValueError(f"Only RECAP schema {SCHEMA_VERSION}, recipe {RECIPE} is supported.")


def dataset_spec(data: Mapping, dataset_id: str) -> ReCapDatasetSpec:
    fields = mapping(data.get("field_mapping", {}), "data.field_mapping")
    check_keys(fields, {"images", "state", "action", "language"}, "data.field_mapping")
    return ReCapDatasetSpec(
        path=path_value(data.get("path"), "data.path"),
        dataset_id=dataset_id,
        format_version=data.get("format_version", "v2.1"),
        image_fields=mapping(fields.get("images", {}), "data.field_mapping.images"),
        state_key=fields.get("state", "observation.state"),
        action_key=fields.get("action", "action"),
        language_key=fields.get("language", "task"),
        fps=data.get("fps"),
        tolerance_s=data.get("tolerance_s", 1e-4),
    )


def parse_offline_config(payload: Mapping) -> ReCapOfflineConfig:
    raw = mapping(payload, "offline configuration")
    check_keys(
        raw,
        {
            "recap_schema_version",
            "recipe",
            "data",
            "annotations",
            "returns",
            "value",
            "train",
            "labeling",
            "advantage",
            "tracking",
        },
        "offline configuration",
    )
    version, recipe = raw.get("recap_schema_version", SCHEMA_VERSION), raw.get("recipe", RECIPE)
    _schema(version, recipe)
    data = mapping(raw.get("data"), "data")
    check_keys(
        data,
        {
            "path",
            "dataset_id",
            "format_version",
            "dataset_type",
            "outcome",
            "intervention_field",
            "field_mapping",
            "fps",
            "tolerance_s",
        },
        "data",
    )
    dataset = dataset_spec(data, nonempty(data.get("dataset_id"), "data.dataset_id"))
    kind = data.get("dataset_type")
    if kind not in {"rollout", "sft"}:
        raise ValueError("data.dataset_type must explicitly be rollout or sft.")
    outcome = None
    if data.get("outcome") is not None:
        src = mapping(data["outcome"], "data.outcome")
        check_keys(src, {"kind", "field", "path"}, "data.outcome")
        if src.get("kind") == "frame_column":
            if src.get("path") is not None:
                raise ValueError("frame_column outcome must not specify path.")
            outcome = OutcomeSource(
                "frame_column", field=nonempty(src.get("field"), "outcome.field")
            )
        elif src.get("kind") == "episode_jsonl":
            if src.get("field") is not None:
                raise ValueError("episode_jsonl outcome must not specify field.")
            outcome = OutcomeSource(
                "episode_jsonl", path=path_value(src.get("path"), "outcome.path")
            )
        else:
            raise ValueError("outcome.kind must be frame_column or episode_jsonl.")
    if kind == "rollout" and outcome is None:
        raise ValueError("Rollout RECAP requires explicit final episode outcomes.")
    intervention = data.get("intervention_field")
    if intervention is not None:
        nonempty(intervention, "data.intervention_field")
    ret = mapping(raw.get("returns", {}), "returns")
    check_keys(ret, {"c_fail_coef"}, "returns")
    coefficient = number(ret.get("c_fail_coef", 1.0), "returns.c_fail_coef")
    if coefficient < 0:
        raise ValueError("returns.c_fail_coef must be non-negative.")
    adv = mapping(raw.get("advantage", {}), "advantage")
    check_keys(adv, AdvantageConfig.__dataclass_fields__, "advantage")
    n = integer(adv.get("n_step", 50), "advantage.n_step")
    ratio = number(adv.get("positive_ratio", 0.3), "advantage.positive_ratio")
    force = adv.get("force_intervention_positive", True)
    if not 0 < ratio <= 1 or not isinstance(force, bool):
        raise ValueError("positive_ratio must be in (0,1] and force_intervention_positive boolean.")
    annotation = mapping(raw.get("annotations"), "annotations")
    check_keys(annotation, {"output_dir"}, "annotations")
    value = mapping(raw.get("value"), "value")
    check_keys(value, ValueModelConfig.__dataclass_fields__, "value")
    value = dict(value)
    for key in ("siglip_path", "gemma3_path", "tokenizer_path"):
        value[key] = path_value(value.get(key), f"value.{key}")
    value_config = ValueModelConfig(**value)
    train = mapping(raw.get("train"), "train")
    check_keys(train, ValueTrainConfig.__dataclass_fields__, "train")
    train["output_dir"] = path_value(train.get("output_dir"), "train.output_dir")
    if train.get("resume_from") is not None:
        train["resume_from"] = path_value(train["resume_from"], "train.resume_from")
    train_config = ValueTrainConfig(**train)
    for key in ("batch_size", "grad_accum_steps", "max_steps", "save_every", "log_every"):
        integer(getattr(train_config, key), f"train.{key}")
    integer(train_config.cache_files, "train.cache_files")
    if train_config.stop_steps is not None:
        integer(train_config.stop_steps, "train.stop_steps")
        if train_config.stop_steps > train_config.max_steps:
            raise ValueError("train.stop_steps cannot exceed train.max_steps.")
    for key in ("seed", "num_workers"):
        integer(getattr(train_config, key), f"train.{key}", minimum=0)
    if number(train_config.learning_rate, "train.learning_rate") <= 0:
        raise ValueError("train.learning_rate must be positive.")
    if not isinstance(train_config.gradient_checkpointing, bool):
        raise ValueError("train.gradient_checkpointing must be boolean.")
    for key in ("warmup_steps", "decay_steps"):
        integer(getattr(train_config, key), f"train.{key}", minimum=0)
    for key in ("stable_steps", "val_episodes"):
        if getattr(train_config, key) is not None:
            integer(getattr(train_config, key), f"train.{key}", minimum=0)
    if not 0 <= number(train_config.val_ratio, "train.val_ratio") < 1:
        raise ValueError("train.val_ratio must be in [0,1).")
    if number(train_config.weight_decay, "train.weight_decay") < 0:
        raise ValueError("train.weight_decay must be non-negative.")
    if not 0 <= number(train_config.decay_lr, "train.decay_lr") <= train_config.learning_rate:
        raise ValueError("train.decay_lr must be between zero and the peak learning rate.")
    label = mapping(raw.get("labeling", {}), "labeling")
    check_keys(label, LabelingConfig.__dataclass_fields__, "labeling")
    labeling = LabelingConfig(**label)
    integer(labeling.batch_size, "labeling.batch_size")
    integer(labeling.num_workers, "labeling.num_workers", minimum=0)
    for device in (train_config.device, labeling.device):
        nonempty(device, "device")
        if (
            device != "cpu"
            and device != "cuda"
            and not (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise ValueError("Value jobs support one cpu or cuda device, not distributed launch.")
    tracking = mapping(raw.get("tracking", {}), "tracking")
    check_keys(tracking, {"tensorboard", "wandb"}, "tracking")
    for backend, settings in tracking.items():
        mapping(settings, f"tracking.{backend}")
    return ReCapOfflineConfig(
        version,
        recipe,
        dataset,
        kind,
        outcome,
        intervention,
        ReturnConfig(coefficient),
        AdvantageConfig(n, ratio, force),
        path_value(annotation.get("output_dir"), "annotations.output_dir"),
        value_config,
        train_config,
        labeling,
        tracking,
    )


def load_offline_config(path: Path, overrides: Sequence[str] = ()) -> ReCapOfflineConfig:
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        raw = mapping(yaml.safe_load(handle), "offline configuration")
    return parse_offline_config(apply_overrides(resolve_environment(raw), list(overrides)))


def validate_policy_config(config: Config) -> ReCapPolicyConfig:
    raw = mapping(config.get("model", {}).get("recap", {}), "model.recap")
    check_keys(raw, ReCapPolicyConfig.__dataclass_fields__, "model.recap")
    result = ReCapPolicyConfig(**raw)
    _schema(result.schema_version, result.recipe)
    if not 0 <= number(result.unconditional_prob, "unconditional_prob") <= 1:
        raise ValueError("model.recap.unconditional_prob must be in [0,1].")
    return result


def resolve_inference_config(config: Config) -> ReCapInferenceConfig:
    for section in ("model", "inference"):
        unsupported = {"guidance_scale", "cfg"} & set(config.get(section, {}))
        if unsupported:
            raise ValueError(f"RECAP does not support {section}.{sorted(unsupported)}.")
    raw = mapping(config.get("inference", {}).get("recap", {}), "inference.recap")
    check_keys(raw, {"condition"}, "inference.recap")
    result = ReCapInferenceConfig(**raw)
    if result.condition not in {"positive", "negative", "base"}:
        raise ValueError("inference.recap.condition must be positive, negative or base.")
    return result
