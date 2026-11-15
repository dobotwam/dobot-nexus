from __future__ import annotations

import importlib
from typing import Any


def import_from_string(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"Import path must use module:object, got {path!r}.")
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise AttributeError(f"{object_name!r} was not found in {module_name!r}.") from exc


def build_model(config, *, mode: str):
    target = import_from_string(str(config.model.target))
    factory = getattr(target, "from_config", None)
    if factory is None:
        raise TypeError(
            f"{config.model.target} must define classmethod from_config(config, mode=...)."
        )
    model = factory(config, mode=mode)

    from dobot_nexus.models.base import BaseVLAModule

    if not isinstance(model, BaseVLAModule):
        raise TypeError(
            f"{config.model.target}.from_config() must return a BaseVLAModule, "
            f"got {type(model)!r}."
        )
    return model
