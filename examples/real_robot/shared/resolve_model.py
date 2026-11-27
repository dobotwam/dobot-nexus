#!/usr/bin/env python3
"""Artifact preflight used by the policy launcher."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ResolvedModel:
    artifact_dir: Path
    torch_checkpoint: Path
    norm_stats: Path


def _latest_target(path: Path) -> Path:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Latest-checkpoint pointer is empty: {path}")
    target = Path(value).expanduser()
    return (target if target.is_absolute() else path.parent / target).resolve()


def _find_artifact(path: Path) -> Path:
    if path.is_file() and path.name in {"latest.txt", "latest_deployment.txt"}:
        path = _latest_target(path)
    elif path.is_dir():
        pointer = next(
            (
                candidate
                for candidate in (
                    path / "checkpoints" / "latest_deployment.txt",
                    path / "checkpoints" / "latest.txt",
                    path / "latest_deployment.txt",
                    path / "latest.txt",
                )
                if candidate.is_file()
            ),
            None,
        )
        if pointer is not None:
            path = _latest_target(pointer)

    search_root = path if path.is_dir() else path.parent
    for candidate in (search_root, *search_root.parents):
        if (candidate / "manifest.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find manifest.yaml above trained-model path: {path}. "
        "Pass a path inside a complete Nexus Artifact so normalization and action "
        "decoding can be restored safely."
    )


def _artifact_file(artifact: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Artifact {label} must be a relative path, got: {relative}")
    path = (artifact / relative).resolve()
    if not path.is_relative_to(artifact.resolve()):
        raise ValueError(f"Artifact {label} escapes its bundle: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Artifact {label} was not found: {path}")
    return path


def resolve_model_path(value: str | Path) -> ResolvedModel:
    supplied = Path(value).expanduser()
    if not supplied.exists():
        raise FileNotFoundError(f"Trained-model path was not found: {supplied}")
    supplied = supplied.resolve()
    artifact = _find_artifact(supplied)

    with (artifact / "manifest.yaml").open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"Unsupported Artifact schema: {artifact / 'manifest.yaml'}")
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"Artifact files must be a mapping: {artifact / 'manifest.yaml'}")

    _artifact_file(
        artifact,
        files.get("resolved_config", "config_resolved.yaml"),
        label="resolved config",
    )
    torch_checkpoint = _artifact_file(
        artifact,
        files.get(
            "weights",
            "deployment/pytorch/full_weights.pt"
            if schema_version == 2
            else "backend/full_weights.pt",
        ),
        label="Torch deployment weights",
    )
    stats_root, stats_path = artifact, files.get("norm_stats", "assets/norm_stats.json")
    if manifest.get("model", {}).get("target") == "dobot_nexus.algorithms.rltoken.policy:RLTokenPolicy":
        relative = files.get("feature_manifest")
        if not isinstance(relative, str) or Path(relative).name != "manifest.yaml":
            raise ValueError("Composed policy requires a declared feature_manifest.")
        child = _artifact_file(artifact, relative, label="feature manifest")
        stats_root = child.parent
        with child.open(encoding="utf-8") as handle:
            provider = yaml.safe_load(handle)
        if not isinstance(provider, dict) or not isinstance(provider.get("files"), dict):
            raise ValueError(f"Invalid feature manifest: {child}")
        provider_files = provider["files"]
        stats_path = provider_files.get("model_assets", {}).get(
            "data.norm_stats_path", provider_files.get("norm_stats", "assets/norm_stats.json")
        )
    norm_stats = _artifact_file(
        stats_root, stats_path,
        label="normalization statistics",
    )
    for key, value in files.get("model_assets", {}).items():
        _artifact_file(artifact, value, label=key)
    for key in ("tokenizer", "memory_vision"):
        if files.get(key):
            _artifact_file(artifact, files[key], label=key)

    return ResolvedModel(
        artifact_dir=artifact.resolve(),
        torch_checkpoint=torch_checkpoint.resolve(),
        norm_stats=norm_stats.resolve(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    args = parser.parse_args()
    resolved = resolve_model_path(args.model_path)
    # The shell launcher consumes exactly two newline-delimited paths. Linux model
    # paths containing newlines are deliberately unsupported.
    print(resolved.artifact_dir)
    print(resolved.torch_checkpoint)


if __name__ == "__main__":
    main()
