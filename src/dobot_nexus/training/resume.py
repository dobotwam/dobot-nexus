from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dobot_nexus.training.seed import data_generators

TRAINING_STATE_VERSION = 3
SEED_POLICY = "shared_initialization_and_sampler_rank_training_v1"


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def preserve_rng_state():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def data_rng_state(data_loader) -> dict[str, torch.Tensor]:
    return {
        key: generator.get_state().clone()
        for key, generator in data_generators(data_loader).items()
    }


def restore_data_rng_state(data_loader, state: Mapping[str, torch.Tensor]) -> None:
    generators = data_generators(data_loader)
    if set(state) != set(generators):
        raise RuntimeError("Resume DataLoader generators do not match the checkpoint.")
    for key, value in state.items():
        generators[key].set_state(value)


def restore_worker_generator(data_loader, state: Mapping[str, torch.Tensor]) -> None:
    generators = data_generators(data_loader)
    worker = generators.get("loader")
    if worker is not None and all(
        worker is not value for key, value in generators.items() if key != "loader"
    ):
        worker.set_state(state["loader"])


def validate_training_state(payload, *, world_size: int, expected_fingerprint: str | None) -> None:
    if (
        payload.get("training_state_version") != TRAINING_STATE_VERSION
        or payload.get("scheduler_step_unit") != "optimizer_update"
        or payload.get("seed_policy") != SEED_POLICY
    ):
        raise ValueError(
            "Resume checkpoint predates the scheduler/seed training-state contract or has an "
            "unsupported version. Load deployment weights into a new run instead of exact resume."
        )
    saved_world_size = int(payload.get("world_size", 1))
    if saved_world_size != world_size:
        raise ValueError(
            "Exact training resume requires the same world size: "
            f"checkpoint={saved_world_size}, current={world_size}."
        )
    if (
        expected_fingerprint is not None
        and payload.get("config_fingerprint") != expected_fingerprint
    ):
        raise ValueError(
            "Resume checkpoint is incompatible with the current seed, model, data, optimizer, "
            "scheduler, precision, or FSDP configuration."
        )
    scheduler_state = payload.get("scheduler")
    if scheduler_state is not None and scheduler_state.get("last_epoch") != payload.get("step"):
        raise ValueError(
            "Resume scheduler clock does not match the successful optimizer update count."
        )


def resolve_resume_checkpoint(run_dir: str | Path, value: str | Path) -> Path:
    run_root = Path(run_dir).expanduser().resolve()
    if str(value).lower() == "auto":
        candidate = run_root / "checkpoints" / "latest_resume.txt"
    else:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_dir() and (candidate / "checkpoints" / "latest_resume.txt").is_file():
            candidate = candidate / "checkpoints" / "latest_resume.txt"
    if candidate.is_file() and candidate.name in {"latest_resume.txt", "latest.txt"}:
        target = Path(candidate.read_text(encoding="utf-8").strip())
        candidate = target if target.is_absolute() else candidate.parent / target
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {candidate}")
    manifest_path = candidate / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Resume manifest not found: {manifest_path}")
    import yaml

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Training resume requires Artifact schema v2: {manifest_path}")
    resume = manifest.get("components", {}).get("resume", {})
    if resume.get("status") != "ready":
        raise ValueError(
            f"Checkpoint cannot be resumed because resume.status={resume.get('status')!r}: "
            f"{candidate}"
        )
    dcp_dir = candidate / str(resume.get("path", "resume/dcp"))
    if not dcp_dir.is_dir():
        raise FileNotFoundError(f"DCP resume state not found: {dcp_dir}")
    return candidate
