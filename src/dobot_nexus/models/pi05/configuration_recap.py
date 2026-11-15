"""Bind the RECAP recipe and saved policy assets to Pi0.5."""

from __future__ import annotations

from dataclasses import asdict

from dobot_nexus.algorithms.recap._validation import check_keys, mapping, path_value
from dobot_nexus.algorithms.recap.configuration import (
    ReCapPolicyConfig,
    resolve_inference_config,
    validate_policy_config,
)
from dobot_nexus.core.config import Config, clone_config

MODEL_TARGET = "dobot_nexus.models.pi05.modeling_recap:Pi05ReCapModel"
LEGACY_MODEL_TARGET = "dobot_nexus.models.pi05.recap.modeling_recap:Pi05ReCapModel"
POLICY_TARGETS = (MODEL_TARGET, LEGACY_MODEL_TARGET)


def resolve_policy_config(config: Config, *, mode: str) -> Config:
    resolved = clone_config(config)
    if mode not in {"train", "infer"}:
        raise ValueError("RECAP mode must be train or infer.")
    resolve_inference_config(resolved)
    artifact = resolved.get("model", {}).get("artifact")
    if artifact:
        from dobot_nexus.core.artifacts import load_manifest, resolve_artifact
        from dobot_nexus.core.config import load_config

        directory = resolve_artifact(str(artifact))
        manifest = load_manifest(directory)
        trained = load_config(directory / "config_resolved.yaml", expected_task="train")
        meta_model = manifest.get("model", {})
        meta = mapping(meta_model.get("recap"), "Artifact model.recap")
        if (
            meta_model.get("target") not in POLICY_TARGETS
            or trained.model.target != meta_model.get("target")
        ):
            raise ValueError("The Artifact is not a RECAP policy Artifact.")
        saved = validate_policy_config(trained)
        expected = {
            **asdict(saved),
            "positive_tag": "Advantage: positive",
            "negative_tag": "Advantage: negative",
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
    if data.setdefault("normalization", "quantile") != "quantile":
        raise ValueError("Dobot ACP requires data.normalization=quantile.")
    if mode == "train":
        recap = mapping(data.get("recap", {}), "data.recap")
        check_keys(recap, {"annotations"}, "data.recap")
        path_value(recap.get("annotations"), "data.recap.annotations")
    return resolved
