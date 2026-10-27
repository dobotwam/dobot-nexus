"""Self-contained v2 deployment composition through existing model export hooks."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

import torch

from dobot_nexus.core.artifacts import (
    _deployment_files,
    artifact_weight_path,
    load_manifest,
    resolve_artifact,
)
from dobot_nexus.core.config import clone_config, load_config, save_config
from dobot_nexus.core.contracts import RLFeatureSpec


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_metadata(path):
    root = resolve_artifact(path)
    manifest = load_manifest(root)
    if "feature_manifest" in manifest.get("files", {}):
        raise ValueError("A composed RLToken policy cannot itself be a feature provider.")
    metadata = manifest["model"]
    if "rl_features" not in metadata:
        raise ValueError("Feature Artifact does not declare the optional RL feature capability.")
    return root, RLFeatureSpec.from_dict(metadata["rl_features"]), metadata


def load_provider(path, *, device="cpu"):
    from dobot_nexus.inference.policy import InferencePolicy

    root, spec, _ = feature_metadata(path)
    config = load_config(root / "config_resolved.yaml", expected_task="train")
    config.task = "infer"
    config.model.artifact = str(root)
    config.setdefault("runtime", {})["device"] = str(device)
    policy = InferencePolicy(config)
    actual = RLFeatureSpec.from_dict(policy.model.artifact_metadata()["rl_features"])
    if actual != spec:
        raise ValueError("Loaded provider feature identity/spec differs from its Artifact.")
    return policy.model, policy.config, spec


def write_model_bundle(root, model, config):
    """Write a fresh directory from live parameters; caller owns atomic publication."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    config = clone_config(config)
    config.task = "train"  # Artifact contract records a train-shaped resolved config.
    config.setdefault("train", {})
    config.setdefault("data", {})
    config.model.pop("artifact", None)
    config.model.pop("checkpoint", None)
    weights = root / "deployment/pytorch/full_weights.pt"
    weights.parent.mkdir(parents=True)
    state = {
        key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()
    }
    torch.save(model.export_backend_state_dict(state), weights)
    files = {"weights": str(weights.relative_to(root)), "resolved_config": "config_resolved.yaml"}
    if model.save_nexus_weights():
        nexus = root / "deployment/nexus/full_weights.pt"
        nexus.parent.mkdir(parents=True)
        torch.save(state, nexus)
        files["nexus_weights"] = str(nexus.relative_to(root))
    assets = dict(model.artifact_assets())
    if config.data.get("norm_stats_path"):
        assets["data.norm_stats_path"] = config.data.norm_stats_path
    for key, source in assets.items():
        source = Path(source)
        if source.is_dir() and key == "data.norm_stats_path":
            source /= "norm_stats.json"
        if not source.is_file():
            raise FileNotFoundError(f"Required model asset {key}: {source}")
        destination = root / "deployment/assets" / (key.replace(".", "_") + source.suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.setdefault("model_assets", {})[key] = str(destination.relative_to(root))
    save_config(config, root / "config_resolved.yaml")
    files.update(
        _deployment_files(
            root, model.export_deployment_artifacts(source_weights=weights, artifact_dir=root)
        )
    )
    manifest = {
        "schema_version": 2,
        "model": {**model.artifact_metadata(), "target": config.model.target},
        "training": {},
        "components": {"deployment": {"status": "ready", "path": "deployment"}},
        "files": files,
        "weights_sha256": file_digest(weights),
    }
    save_config(manifest, root / "manifest.yaml")
    return root
