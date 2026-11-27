from __future__ import annotations

import copy
import hashlib
import json
import shutil
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import yaml

from dobot_nexus.core.config import resolve_autocast, save_config, to_plain_dict

_verified_snapshot = ContextVar("verified_artifact_snapshot", default=None)


@contextmanager
def verified_artifact_snapshot(artifact_dir, manifest, training_config):
    """Pin the artifact metadata used by constructors in this execution context."""
    root = Path(artifact_dir).expanduser().resolve()
    token = _verified_snapshot.set((root, copy.deepcopy((manifest, training_config))))
    try:
        yield
    finally:
        _verified_snapshot.reset(token)


def get_verified_artifact_snapshot(path, *, resolve_pointer=True):
    active = _verified_snapshot.get()
    if active is None:
        return None
    root, snapshot = active
    candidate = Path(path).expanduser().resolve()
    if candidate != root:
        if not resolve_pointer:
            return None
        try:
            if resolve_artifact(candidate) != root:
                return None
        except (FileNotFoundError, ValueError):
            return None
    return copy.deepcopy(snapshot)


def resume_fingerprint(config) -> str:
    """Hash fields that must stay compatible across a training restart."""
    model = dict(to_plain_dict(config.get("model", {})))
    # Ignore deployment settings, including obsolete fields in old resume configs.
    for key in ("artifact", "jax_checkpoint", "jax_export_base_checkpoint"):
        model.pop(key, None)
    data = dict(to_plain_dict(config.get("data", {})))
    # Worker/cache sizing and confirmation of legacy store columns do not change samples.
    for key in ("num_workers", "memory_legacy_columns", "memory_cache_bytes"):
        data.pop(key, None)
    runtime = dict(to_plain_dict(config.get("runtime", {})))
    autocast = resolve_autocast(config)
    if autocast == "auto":
        # The default preserves pre-option behavior and resume fingerprints.
        runtime.pop("autocast", None)
    else:
        runtime["autocast"] = autocast
    scheduler = dict(to_plain_dict(config.get("scheduler", {})))
    if str(scheduler.get("name", "warmup_cosine")).lower() == "warmup_cosine":
        scheduler.setdefault("total_steps", int(config.train.max_steps))
    payload = {
        "model": model,
        "data": data,
        "optim": to_plain_dict(config.get("optim", {})),
        "scheduler": scheduler,
        "seed": config.get("exp", {}).get("seed", 0),
        "runtime": runtime,
        "grad_accum_steps": int(config.get("train", {}).get("grad_accum_steps", 1)),
    }
    if config.get("train", {}).get("ema_decay") is not None:
        payload["ema_decay"] = config.train.ema_decay
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_state_dict(state_dict: Any, *, label: str) -> None:
    import torch

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"{label} must be a non-empty state dict.")
    non_tensors = [key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise RuntimeError(f"{label} contains non-tensor entries: {non_tensors[:8]}.")
    tensor_subclasses = [
        key for key, value in state_dict.items()
        if type(value) not in (torch.Tensor, torch.nn.Parameter)
    ]
    if tensor_subclasses:
        raise RuntimeError(
            f"{label} contains tensor subclasses or distributed shards: {tensor_subclasses[:8]}; "
            "gather a full state dict of ordinary tensors before deployment export."
        )
    non_finite = [key for key, value in state_dict.items()
                  if (value.is_floating_point() or value.is_complex())
                  and not torch.isfinite(value).all().item()]
    if non_finite:
        raise RuntimeError(f"{label} contains non-finite tensors: {non_finite[:8]}.")
    flat_keys = [key for key in state_dict if "flat_param" in str(key).lower()]
    if flat_keys:
        raise RuntimeError(
            f"{label} contains FSDP FlatParameter keys instead of a full state dict: "
            f"{flat_keys[:8]}."
        )


def _deployment_files(root: Path, produced: dict[str, str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for logical_name, relative_value in produced.items():
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"Deployment artifact {logical_name!r} must be relative to {root}: {relative}."
            )
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Deployment artifact {logical_name!r} was not created: {path}."
            )
        files[str(logical_name)] = relative.as_posix()
    return files


def _resolve_latest_file(path: Path) -> Path:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Latest-checkpoint pointer is empty: {path}")
    target = Path(value)
    return target.resolve() if target.is_absolute() else (path.parent / target).resolve()


def _write_pointer(path: Path, target_name: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{target_name}\n", encoding="utf-8")
    temporary.replace(path)


def _save_manifest(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    save_config(payload, temporary)
    temporary.replace(path)


def resolve_artifact(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    active = _verified_snapshot.get()
    if active is not None and candidate == active[0]:
        return candidate
    if candidate.is_file() and candidate.name in {
        "latest.txt",
        "latest_deployment.txt",
        "latest_resume.txt",
    }:
        candidate = _resolve_latest_file(candidate)
    if candidate.is_dir():
        pointer_candidates = (
            candidate / "checkpoints" / "latest_deployment.txt",
            candidate / "checkpoints" / "latest.txt",
            candidate / "latest_deployment.txt",
            candidate / "latest.txt",
        )
        pointer = next((item for item in pointer_candidates if item.is_file()), None)
        if pointer is not None:
            candidate = _resolve_latest_file(pointer)
    if not candidate.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {candidate}")
    if (active is None or candidate != active[0]) and not (candidate / "manifest.yaml").is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {candidate / 'manifest.yaml'}")
    return candidate


def load_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    root = resolve_artifact(artifact_dir)
    snapshot = get_verified_artifact_snapshot(root, resolve_pointer=False)
    if snapshot is not None:
        return snapshot[0]
    with (root / "manifest.yaml").open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError(f"Unsupported artifact schema in {root / 'manifest.yaml'}.")
    return payload


class ArtifactManager:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_root = self.run_dir / "checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model,
        runtime,
        optimizer,
        scheduler,
        config,
        step: int,
        metrics: dict[str, float],
        data_state: dict[str, Any] | None = None,
        final: bool = False,
        ema=None,
    ) -> Path:
        import torch

        del final  # Deployment artifacts are intentionally produced at every checkpoint.
        resume_limit = config.get("train", {}).get("resume_limit", 2)
        if resume_limit is not None:
            resume_limit = int(resume_limit)
            if resume_limit < 0:
                raise ValueError("train.resume_limit must be null or a non-negative integer.")
        checkpoint_name = f"step_{step:08d}"
        checkpoint_dir = self.checkpoint_root / checkpoint_name
        staging_dir = self.checkpoint_root / f".{checkpoint_name}.tmp"

        preflight_error = None
        if runtime.is_main_process:
            if checkpoint_dir.exists():
                preflight_error = f"Checkpoint already exists: {checkpoint_dir}"
            else:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
                staging_dir.mkdir(parents=True)
        preflight_error = runtime.broadcast_object(preflight_error)
        if preflight_error:
            raise FileExistsError(preflight_error)
        runtime.wait_for_everyone()

        resume_enabled = resume_limit != 0
        if resume_enabled:
            if ema is not None:
                ema.save(staging_dir / "resume", runtime.process_index)
            runtime.save_training_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                resume_dir=staging_dir / "resume",
                trainer_state={
                    "step": step,
                    "metrics": metrics,
                    "data_state": data_state or {},
                    "config_fingerprint": resume_fingerprint(config),
                },
            )
        runtime.wait_for_everyone()

        # Full state collection is deployment-only. DCP above already saved the
        # sharded model and optimizer without gathering optimizer state on rank 0.
        with ema.apply() if ema is not None else nullcontext():
            # Clone before restoring live parameters; LocalRuntime returns aliased tensors.
            collected = runtime.model_state_dict(model)
            state_dict = (
                {k: v.detach().clone() for k, v in collected.items()}
                if ema is not None else collected
            )
            del collected
        runtime.wait_for_everyone()
        save_error = None
        if runtime.is_main_process:
            try:
                unwrapped = runtime.unwrap_model(model)
                _validate_state_dict(state_dict, label="Nexus model state")
                backend_state = unwrapped.export_backend_state_dict(state_dict)
                _validate_state_dict(backend_state, label="Backend model state")

                pytorch_dir = staging_dir / "deployment" / "pytorch"
                pytorch_dir.mkdir(parents=True)
                backend_path = pytorch_dir / "full_weights.pt"
                torch.save(backend_state, backend_path)
                del backend_state
                files: dict[str, Any] = {
                    "weights": "deployment/pytorch/full_weights.pt",
                    "resolved_config": "config_resolved.yaml",
                }
                if resume_enabled:
                    files["trainer_state"] = "resume/trainer_state.pt"

                if unwrapped.save_nexus_weights():
                    nexus_dir = staging_dir / "deployment" / "nexus"
                    nexus_dir.mkdir(parents=True)
                    torch.save(state_dict, nexus_dir / "full_weights.pt")
                    files["nexus_weights"] = "deployment/nexus/full_weights.pt"

                assets_dir = staging_dir / "deployment" / "assets"
                norm_stats = config.get("data", {}).get("norm_stats_path")
                if norm_stats:
                    source = Path(str(norm_stats)).expanduser()
                    if source.is_dir():
                        source = source / "norm_stats.json"
                    if source.is_file():
                        assets_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, assets_dir / "norm_stats.json")
                        files["norm_stats"] = "deployment/assets/norm_stats.json"
                model_assets = {}
                for config_path, configured_path in unwrapped.artifact_assets().items():
                    source = Path(str(configured_path)).expanduser()
                    if not source.is_file():
                        continue
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = config_path.replace(".", "_") + source.suffix
                    destination = assets_dir / safe_name
                    shutil.copy2(source, destination)
                    model_assets[config_path] = str(destination.relative_to(staging_dir))
                if model_assets:
                    files["model_assets"] = model_assets
                save_config(config, staging_dir / "config_resolved.yaml")

                # Model-owned assets (e.g. Memory's frozen vision encoder) accompany
                # the Torch weights. The common save path performs no JAX conversion.
                produced = unwrapped.export_deployment_artifacts(
                    source_weights=backend_path,
                    artifact_dir=staging_dir,
                )
                files.update(_deployment_files(staging_dir, produced))
                resume_component = {
                    "status": "ready" if resume_enabled else "disabled",
                    "format": "torch_dcp" if resume_enabled else None,
                    "path": "resume/dcp" if resume_enabled else None,
                    "world_size": runtime.world_size,
                }
                manifest = {
                    "schema_version": 2,
                    # FSDP2 changes the Python class in place. Its generated
                    # class is not an importable deployment model target.
                    "model": {**unwrapped.artifact_metadata(), "target": config.model.target},
                    "training": {
                        "step": step,
                        "metrics": metrics,
                        **({"ema_decay": ema.decay, "deployment_weights": "ema"}
                           if ema is not None else {}),
                    },
                    "components": {
                        "resume": resume_component,
                        "deployment": {"status": "ready", "path": "deployment"},
                    },
                    # Keep the flat file index so existing inference code can read
                    # schema v1 and v2 without model-specific branching.
                    "files": files,
                }
                _save_manifest(manifest, staging_dir / "manifest.yaml")
                staging_dir.replace(checkpoint_dir)
                _write_pointer(self.checkpoint_root / "latest_deployment.txt", checkpoint_name)
                _write_pointer(self.checkpoint_root / "latest.txt", checkpoint_name)
                if resume_enabled:
                    _write_pointer(self.checkpoint_root / "latest_resume.txt", checkpoint_name)
                else:
                    (self.checkpoint_root / "latest_resume.txt").unlink(missing_ok=True)
            except Exception as exc:  # propagate a rank-0 export failure to every worker
                save_error = f"{type(exc).__name__}: {exc}"
        del state_dict
        save_error = runtime.broadcast_object(save_error)
        if save_error:
            raise RuntimeError(f"Checkpoint deployment export failed: {save_error}")
        runtime.wait_for_everyone()

        prune_error = None
        if runtime.is_main_process:
            try:
                self._prune_resume_checkpoints(resume_limit)
            except Exception as exc:
                prune_error = f"{type(exc).__name__}: {exc}"
        prune_error = runtime.broadcast_object(prune_error)
        if prune_error:
            raise RuntimeError(f"Checkpoint was saved, but resume pruning failed: {prune_error}")
        runtime.wait_for_everyone()
        return checkpoint_dir

    def _prune_resume_checkpoints(self, limit: int | None) -> None:
        if limit is None:
            return
        resumable: list[tuple[int, Path, dict[str, Any]]] = []
        for checkpoint_dir in self.checkpoint_root.glob("step_*"):
            manifest_path = checkpoint_dir / "manifest.yaml"
            if not manifest_path.is_file():
                continue
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = yaml.safe_load(handle) or {}
            if manifest.get("schema_version") != 2:
                continue
            resume = manifest.get("components", {}).get("resume", {})
            if resume.get("status") != "ready":
                continue
            if not (checkpoint_dir / str(resume.get("path", "resume/dcp"))).is_dir():
                continue
            step = int(manifest.get("training", {}).get("step", -1))
            resumable.append((step, checkpoint_dir, manifest))
        resumable.sort(key=lambda item: item[0], reverse=True)
        for _, checkpoint_dir, manifest in resumable[limit:]:
            resume_dir = checkpoint_dir / "resume"
            if resume_dir.is_dir():
                shutil.rmtree(resume_dir)
            resume = manifest["components"]["resume"]
            resume["status"] = "pruned"
            manifest.get("files", {}).pop("trainer_state", None)
            _save_manifest(manifest, checkpoint_dir / "manifest.yaml")


def artifact_weight_path(artifact_dir: str | Path) -> Path:
    root = resolve_artifact(artifact_dir)
    manifest = load_manifest(root)
    default = (
        "deployment/pytorch/full_weights.pt"
        if manifest.get("schema_version") == 2
        else "backend/full_weights.pt"
    )
    relative = manifest.get("files", {}).get("weights", default)
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact weights not found: {path}")
    return path
