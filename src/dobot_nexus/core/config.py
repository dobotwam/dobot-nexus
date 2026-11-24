from __future__ import annotations

import ast
import copy
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


class Config(dict):
    """A recursively wrapped mapping with attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, path: str, default: Any = None) -> Any:
        value: Any = self
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config({str(key): _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    return value


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        resolved = os.getenv(name, default)
        if resolved is None:
            raise ValueError(f"Environment variable {name!r} is required.")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def resolve_environment(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_environment(item) for item in value]
    if isinstance(value, str):
        return _resolve_env_string(value)
    return value


def parse_override(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}.")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"Override has an empty key: {item!r}.")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            child = target.get(part)
            if child is None:
                child = {}
                target[part] = child
            if not isinstance(child, dict):
                raise ValueError(f"Override path {key!r} crosses non-mapping field {part!r}.")
            target = child
        target[parts[-1]] = parse_override(raw_value)
    return config


def resolve_autocast(config: Mapping[str, Any]) -> str:
    """Resolve the outer autocast policy without importing a model or Torch."""
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be a mapping.")
    value = runtime.get("autocast", "auto")
    # PyYAML reads an unquoted `off` as False; CLI overrides retain the string.
    if value is False:
        return "off"
    if isinstance(value, str) and value.lower() in {"auto", "off"}:
        return value.lower()
    raise ValueError("runtime.autocast must be 'auto' or 'off'.")


def validate_config(config: Mapping[str, Any], expected_task: str | None = None) -> None:
    resolve_autocast(config)
    schema_version = config.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"schema_version must be 1, got {schema_version!r}.")
    task = config.get("task")
    if task not in {"train", "infer"}:
        raise ValueError(f"task must be 'train' or 'infer', got {task!r}.")
    if expected_task is not None and task != expected_task:
        raise ValueError(f"Expected task={expected_task!r}, got {task!r}.")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping.")
    target = model.get("target")
    if not isinstance(target, str) or ":" not in target:
        raise ValueError("model.target must use the 'module:object' form.")
    for key in ("state_dim", "action_dim", "action_horizon"):
        value = model.get(key)
        if value is None and task == "infer" and model.get("artifact"):
            continue  # Artifact loading resolves public dimensions before model construction.
        if type(value) is not int or value <= 0:
            raise ValueError(f"model.{key} must be a positive integer, got {value!r}.")
    if task == "train":
        train = config.get("train")
        if not isinstance(train, Mapping):
            raise ValueError("train configuration is required for task='train'.")
        if not isinstance(config.get("data"), Mapping):
            raise ValueError("data configuration is required for task='train'.")
        resume_limit = train.get("resume_limit", 2)
        if resume_limit is not None and (
            not isinstance(resume_limit, int)
            or isinstance(resume_limit, bool)
            or resume_limit < 0
        ):
            raise ValueError("train.resume_limit must be null or a non-negative integer.")
        resume_from = train.get("resume_from")
        if resume_from is not None and not isinstance(resume_from, (str, Path)):
            raise ValueError("train.resume_from must be null, 'auto', or a checkpoint path.")
    if task == "infer" and not model.get("artifact") and not model.get("checkpoint"):
        raise ValueError("Inference requires model.artifact or model.checkpoint.")


def _merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; scalars, lists and null replace the base value."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def _read_config(path: Path, parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in parents:
        chain = " -> ".join(str(item) for item in (*parents, path))
        raise ValueError(f"Circular config extends: {chain}")
    if len(parents) >= 32:
        raise ValueError(f"Config extends exceeds 32 levels: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    if "extends" not in payload:
        return payload
    base = payload.pop("extends")
    if not isinstance(base, str) or not base.strip():
        raise ValueError(f"Config extends must name one YAML file: {path}")
    base_path = (path.parent / Path(base).expanduser()).resolve()
    return _merge_config(_read_config(base_path, (*parents, path)), payload)


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    expected_task: str | None = None,
) -> Config:
    """Load a YAML template or recipe, then apply environment and CLI overrides.

    ``extends`` names one base file relative to the declaring YAML. It is removed
    during loading, so saved resolved configs do not depend on template files.
    """
    config_path = Path(path).expanduser().resolve()
    payload = _read_config(config_path)
    payload = resolve_environment(payload)
    payload = apply_overrides(payload, overrides)
    if "extends" in payload:
        raise ValueError("Declare extends in the YAML file, not as a command-line override.")
    validate_config(payload, expected_task=expected_task)
    config = _wrap(payload)
    config.setdefault("exp", Config())
    config.exp.setdefault("config_path", str(config_path))
    return config


def clone_config(config: Config) -> Config:
    return _wrap(copy.deepcopy(to_plain_dict(config)))


def resolve_training_runtime(config: Mapping[str, Any], *, world_size: int | None = None) -> Config:
    """Resolve auto launch settings without importing Torch or mutating the input.

    The shell launcher supplies its planned process count; training workers use
    WORLD_SIZE from torchrun. Persist the returned settings before saving a run
    so checkpoint compatibility checks use the actual execution strategy.
    """
    resolve_autocast(config)
    if world_size is None:
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as exc:
            raise ValueError("WORLD_SIZE must be a positive integer.") from exc
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer.")
    runtime = _wrap(copy.deepcopy(to_plain_dict(config.get("runtime", {}))))
    launcher = str(runtime.get("launcher", "local")).lower()
    if launcher not in {"auto", "local", "accelerate"}:
        raise ValueError(f"Unsupported runtime.launcher={launcher!r}.")
    fsdp = runtime.get("fsdp", {})
    enabled = fsdp.get("enabled", False)
    if type(enabled) is not bool and enabled != "auto":
        raise ValueError("runtime.fsdp.enabled must be true, false, or 'auto'.")
    if launcher == "auto":
        launcher = "local" if world_size == 1 else "accelerate"
        runtime["launcher"] = launcher
    if enabled == "auto":
        enabled = launcher == "accelerate" and world_size > 1
        fsdp["enabled"] = enabled
    if launcher == "local" and world_size != 1:
        raise ValueError("runtime.launcher=local requires exactly one process.")
    if enabled and world_size < 2:
        raise ValueError("Config enables FSDP and requires at least two processes.")
    return runtime


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_dict(config), handle, sort_keys=False, allow_unicode=True)
