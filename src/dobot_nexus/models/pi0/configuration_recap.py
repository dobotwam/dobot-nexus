"""Bind the RECAP recipe and saved policy assets to Pi0."""

from __future__ import annotations

from dataclasses import asdict

from dobot_nexus.algorithms.recap._validation import check_keys, mapping, path_value
from dobot_nexus.algorithms.recap.conditioning import NEGATIVE_TAG, POSITIVE_TAG
from dobot_nexus.algorithms.recap.configuration import (
    ReCapPolicyConfig,
    resolve_inference_config,
    validate_policy_config,
)
from dobot_nexus.core.config import Config, clone_config

from .configuration_pi0 import restore_token_limit

MODEL_TARGET = "dobot_nexus.models.pi0.modeling_recap:Pi0ReCapModel"


def resolve_policy_config(config: Config, *, mode: str) -> Config:
    resolved = clone_config(config)
    if mode not in {"train", "infer"}:
        raise ValueError("RECAP mode must be train or infer.")
    resolved["task"] = mode
    resolve_inference_config(resolved)
    artifact = resolved.get("model", {}).get("artifact")
    if artifact:
        from dobot_nexus.core.artifacts import load_manifest, resolve_artifact
        from dobot_nexus.core.config import load_config

        directory = resolve_artifact(str(artifact))
        manifest = load_manifest(directory)
        trained = load_config(directory / "config_resolved.yaml", expected_task="train")
        meta_model = manifest.get("model", {})
        if meta_model.get("target") != MODEL_TARGET or trained.model.target != MODEL_TARGET:
            raise ValueError("The Artifact is not a RECAP policy Artifact.")
        meta = mapping(meta_model.get("recap"), "Artifact model.recap")
        restore_token_limit(resolved.model, trained.model, meta_model)
        saved = validate_policy_config(trained)
        expected = {
            **asdict(saved),
            "positive_tag": POSITIVE_TAG,
            "negative_tag": NEGATIVE_TAG,
            "default_condition": "base",
        }
        if meta != expected:
            raise ValueError("Artifact RECAP metadata disagrees with its resolved configuration.")
        explicit = mapping(resolved.model.get("recap", {}), "model.recap")
        check_keys(explicit, ReCapPolicyConfig.__dataclass_fields__, "model.recap")
        if any(value != asdict(saved)[key] for key, value in explicit.items()):
            raise ValueError("Cannot override the trained RECAP recipe or condition dropout.")
        resolved.model.recap = Config(asdict(saved))
    else:
        resolved.setdefault("model", Config())["recap"] = Config(
            asdict(validate_policy_config(resolved))
        )
    data = resolved.setdefault("data", Config())
    if data.setdefault("normalization", "zscore") != "zscore":
        raise ValueError("Pi0 RECAP requires mean/std (zscore) normalization.")
    if mode == "train":
        recap = mapping(data.get("recap", {}), "data.recap")
        check_keys(recap, {"annotations"}, "data.recap")
        path_value(recap.get("annotations"), "data.recap.annotations")
    return resolved
