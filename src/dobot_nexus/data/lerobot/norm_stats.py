from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dobot_nexus.data.cache import file_lock
from dobot_nexus.data.lerobot.metadata import find_parquet_files, load_info
from dobot_nexus.data.statistics import NormStats, RunningStats, load_norm_stats, save_norm_stats
from dobot_nexus.data.transforms import ActionTransform, apply_action_transform

LOGGER = logging.getLogger(__name__)
_ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class LeRobotNormStatsSpec:
    """Dataset fields and preprocessing semantics that affect normalization stats."""

    state_key: str
    action_key: str
    state_dim: int
    action_dim: int
    action_horizon: int
    action_transform: ActionTransform

    def validate(self) -> None:
        if not self.state_key or not self.action_key:
            raise ValueError("LeRobot state_key and action_key must be non-empty.")
        if self.state_dim <= 0 or self.action_dim <= 0 or self.action_horizon <= 0:
            raise ValueError("LeRobot state/action dimensions and action_horizon must be positive.")
        if self.action_transform.delta_mask.shape != (self.action_dim,):
            raise ValueError(
                f"delta_mask must contain action_dim={self.action_dim} boolean values."
            )
        if self.action_transform.kind == "delta" and self.state_dim != self.action_dim:
            raise ValueError(
                "Delta action statistics require state_dim == action_dim, got "
                f"{self.state_dim} != {self.action_dim}."
            )

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "algorithm_version": _ALGORITHM_VERSION,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "action_transform": self.action_transform.kind,
            "delta_mask": self.action_transform.delta_mask.tolist(),
        }


def _vector_array(name: str, values: Any, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"{name} must have shape [frames,{dimension}], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_episode(
    episode_index: int,
    states: Any,
    actions: Any,
    frame_indices: Any,
    spec: LeRobotNormStatsSpec,
) -> tuple[np.ndarray, np.ndarray]:
    state_array = _vector_array("observation.state", states, spec.state_dim)
    action_array = _vector_array("action", actions, spec.action_dim)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if state_array.shape[0] == 0:
        raise ValueError(f"LeRobot episode {episode_index} is empty.")
    if action_array.shape[0] != state_array.shape[0] or frames.shape != (state_array.shape[0],):
        raise ValueError(f"LeRobot episode {episode_index} columns have different frame counts.")
    expected = np.arange(state_array.shape[0], dtype=np.int64)
    if not np.array_equal(frames, expected):
        raise ValueError(
            f"LeRobot episode {episode_index} frame_index must be contiguous and start at zero; "
            f"got {frames.tolist()}."
        )
    return state_array, action_array


def _iter_lerobot_episodes(
    dataset_root: Path,
    spec: LeRobotNormStatsSpec,
    *,
    parquet_batch_size: int = 65536,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Automatic LeRobot normalization requires pyarrow; "
            "install the Pi0.5 extras with `pip install -e '.[pi05]'`."
        ) from exc

    columns = list(dict.fromkeys([spec.state_key, spec.action_key, "episode_index", "frame_index"]))
    current_episode: int | None = None
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []

    def finish() -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        assert current_episode is not None
        return (
            current_episode,
            np.concatenate(state_parts, axis=0),
            np.concatenate(action_parts, axis=0),
            np.concatenate(frame_parts, axis=0),
        )

    for parquet_path in find_parquet_files(dataset_root):
        parquet_file = parquet.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        missing = [name for name in columns if name not in available]
        if missing:
            raise KeyError(f"LeRobot Parquet {parquet_path} is missing columns: {missing}.")
        for batch in parquet_file.iter_batches(batch_size=parquet_batch_size, columns=columns):
            values = {
                name: np.asarray(batch.column(name).to_pylist())
                for name in columns
            }
            episode_indices = np.asarray(values["episode_index"], dtype=np.int64)
            frame_indices = np.asarray(values["frame_index"], dtype=np.int64)
            states = np.asarray(values[spec.state_key], dtype=np.float64)
            actions = np.asarray(values[spec.action_key], dtype=np.float64)
            if episode_indices.ndim != 1 or frame_indices.ndim != 1:
                raise ValueError("LeRobot episode_index and frame_index must be scalar columns.")
            boundaries = np.flatnonzero(episode_indices[1:] != episode_indices[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [episode_indices.shape[0]]))
            for start, end in zip(starts, ends, strict=True):
                if start == end:
                    continue
                episode_index = int(episode_indices[start])
                if current_episode is not None and episode_index < current_episode:
                    raise ValueError(
                        "LeRobot Parquet rows must be ordered by episode_index; "
                        f"encountered {episode_index} after {current_episode}."
                    )
                if current_episode is not None and episode_index != current_episode:
                    yield finish()
                    state_parts = []
                    action_parts = []
                    frame_parts = []
                current_episode = episode_index
                state_parts.append(states[start:end])
                action_parts.append(actions[start:end])
                frame_parts.append(frame_indices[start:end])
    if current_episode is not None:
        yield finish()


def calculate_norm_stats_from_episodes(
    episodes: Iterator[tuple[int, Any, Any, Any]],
    spec: LeRobotNormStatsSpec,
    *,
    action_block_size: int = 4096,
) -> dict[str, NormStats]:
    """Compute stats over the exact state/action values seen before normalization."""
    spec.validate()
    if action_block_size <= 0:
        raise ValueError("action_block_size must be positive.")
    state_stats = RunningStats()
    action_stats = RunningStats()
    episode_count = 0
    frame_count = 0
    offsets = np.arange(spec.action_horizon, dtype=np.int64)
    for episode_index, states, actions, frame_indices in episodes:
        state_array, action_array = _validate_episode(
            int(episode_index), states, actions, frame_indices, spec
        )
        episode_count += 1
        frame_count += state_array.shape[0]
        state_stats.update(state_array)
        for start in range(0, state_array.shape[0], action_block_size):
            end = min(start + action_block_size, state_array.shape[0])
            query = np.minimum(
                np.arange(start, end, dtype=np.int64)[:, None] + offsets,
                state_array.shape[0] - 1,
            )
            chunks = action_array[query]
            chunks = apply_action_transform(
                chunks,
                state_array[start:end, None, :],
                spec.action_transform,
            )
            action_stats.update(chunks)
    if episode_count == 0:
        raise ValueError("Cannot calculate normalization statistics for an empty dataset.")
    LOGGER.info(
        "Calculated LeRobot normalization statistics from %d episodes and %d frames.",
        episode_count,
        frame_count,
    )
    return {"state": state_stats.result(), "actions": action_stats.result()}


def calculate_lerobot_norm_stats(
    dataset_root: str | Path,
    spec: LeRobotNormStatsSpec,
) -> dict[str, NormStats]:
    root = Path(dataset_root).expanduser().resolve()
    _ = load_info(root)
    return calculate_norm_stats_from_episodes(_iter_lerobot_episodes(root, spec), spec)


def lerobot_norm_stats_fingerprint(
    dataset_root: str | Path,
    spec: LeRobotNormStatsSpec,
) -> str:
    spec.validate()
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    _ = load_info(root)
    files = [info_path, *find_parquet_files(root)]
    payload = spec.fingerprint_payload()
    payload["dataset_root"] = str(root)
    payload["files"] = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    digest.update(info_path.read_bytes())
    return digest.hexdigest()


def _valid_generated_stats(path: Path, spec: LeRobotNormStatsSpec) -> bool:
    if not path.is_file():
        return False
    try:
        stats = load_norm_stats(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {"state": spec.state_dim, "actions": spec.action_dim}
    return all(
        name in stats
        and stats[name].mean.shape == (dimension,)
        and stats[name].std.shape == (dimension,)
        for name, dimension in expected.items()
    )


def resolve_or_compute_lerobot_norm_stats(
    dataset_root: str | Path,
    spec: LeRobotNormStatsSpec,
    *,
    cache_root: str | Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Return a generated stats path; one process computes each fingerprint."""
    active_logger = logger or LOGGER
    root = Path(dataset_root).expanduser().resolve()
    fingerprint = lerobot_norm_stats_fingerprint(root, spec)
    cache = Path(cache_root).expanduser().resolve()
    destination = cache / fingerprint / "norm_stats.json"
    if _valid_generated_stats(destination, spec):
        active_logger.info("Using cached LeRobot normalization statistics: %s", destination)
        return destination

    lock_path = cache / f"{fingerprint}.lock"
    with file_lock(lock_path):
        if _valid_generated_stats(destination, spec):
            active_logger.info("Using cached LeRobot normalization statistics: %s", destination)
            return destination
        active_logger.info(
            "Computing LeRobot normalization statistics for %s; this runs once per dataset/config.",
            root,
        )
        stats = calculate_lerobot_norm_stats(root, spec)
        save_norm_stats(destination, stats, atomic=True)
        active_logger.info("Saved LeRobot normalization statistics: %s", destination)
    return destination
