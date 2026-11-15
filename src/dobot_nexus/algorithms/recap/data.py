"""RECAP frame indexing, outcome provenance and policy-label dataset checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from dobot_nexus.data.lerobot.metadata import find_parquet_files, load_info, load_jsonl

from ._validation import integer, nonempty
from .configuration import ReCapDatasetSpec, dataset_spec
from .targets import intervention_vector
from .types import EpisodeRecord, FrameIndex, FrameKey


def _columns(dataset: ReCapDatasetSpec, columns: list[str]) -> dict[str, list]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "RECAP metadata/annotations require pyarrow; install the pi05 extra."
        ) from exc
    result = {name: [] for name in columns}
    for path in find_parquet_files(dataset.path):
        schema = pq.read_schema(path)
        if len(set(schema.names)) != len(schema.names):
            raise ValueError(f"Duplicate Parquet columns: {path}.")
        missing = set(columns) - set(schema.names)
        if missing:
            raise ValueError(f"{path} is missing RECAP columns {sorted(missing)}.")
        table = pq.read_table(path, columns=columns)
        for key in columns:
            result[key].extend(table[key].to_pylist())
    return result


def _index_integer(value, name: str) -> int:
    a = np.asarray(value)
    if a.size != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a scalar integer.")
    result = int(a.reshape(-1)[0])
    return integer(result, name, minimum=0)


def _bool(value, name: str) -> bool:
    a = np.asarray(value)
    if a.size != 1 or a.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean scalar or single-element boolean array.")
    return bool(a.reshape(-1)[0])


def build_frame_index(dataset: ReCapDatasetSpec) -> FrameIndex:
    info = load_info(dataset.path)
    if info.get("codebase_version") != "v2.1":
        raise ValueError("RECAP requires LeRobot v2.1 metadata.")
    integer(info.get("fps"), "meta.info.fps")
    if dataset.fps is not None and dataset.fps != info["fps"]:
        raise ValueError("RECAP data.fps must match meta.info.fps; resample before returns/labels.")
    features = info.get("features", {})
    for field in dataset.image_fields.values():
        kind = features.get(field, {}).get("dtype")
        if kind not in {"video", "image"}:
            raise ValueError(f"RECAP image field {field} must be an image or video feature.")
        if kind == "video" and not info.get("video_path"):
            raise ValueError("Video features require a video_path template.")
    for field in (dataset.state_key, dataset.action_key):
        if field not in features:
            raise ValueError(f"Missing dataset feature {field}.")
    metadata = load_jsonl(dataset.path / "meta/episodes.jsonl")
    tasks = load_jsonl(dataset.path / "meta/tasks.jsonl")
    task_ids = set()
    for task in tasks:
        task_id = _index_integer(task.get("task_index"), "task_index")
        if task_id in task_ids:
            raise ValueError("Duplicate task_index in metadata.")
        task_ids.add(task_id)
        nonempty(task.get("task"), "task text")
    raw = _columns(dataset, ["episode_index", "frame_index", "task_index"])
    n = len(raw["episode_index"])
    if not n or not metadata:
        raise ValueError("RECAP dataset must contain non-empty episodes.")
    triples = [tuple(_index_integer(raw[k][i], k) for k in raw) for i in range(n)]
    episodes = []
    cursor = 0
    seen = set()
    for entry in metadata:
        episode = _index_integer(entry.get("episode_index"), "episode_index")
        length = integer(entry.get("length"), "episode length")
        if episode in seen:
            raise ValueError("Duplicate episode_index in metadata.")
        seen.add(episode)
        for offset, row in enumerate(triples[cursor : cursor + length]):
            if row[:2] != (episode, offset):
                raise ValueError(
                    "Dataset rows do not match the local LeRobot reader's episode layout: "
                    f"expected ({episode},{offset}), found {row[:2]}. "
                    "Repair frame ordering/duplicates/gaps during dataset conversion."
                )
            if row[2] not in task_ids:
                raise ValueError(f"Unknown task_index {row[2]}.")
        cursor += length
        episodes.append(EpisodeRecord(episode, length))
    if cursor != n:
        raise ValueError(f"Episode metadata length {cursor} does not match Parquet rows {n}.")
    for key, expected in (("total_frames", n), ("total_episodes", len(episodes))):
        if key in info and info[key] != expected:
            raise ValueError(f"meta.info.{key} disagrees with dataset rows.")
    order = np.asarray(sorted(range(n), key=lambda i: triples[i][:2]), dtype=np.int64)
    keys = tuple(FrameKey(dataset.dataset_id, *triples[i][:2]) for i in order)
    if len(set(keys)) != n:
        raise ValueError("Duplicate RECAP frame keys.")
    episodes.sort(key=lambda e: e.episode_index)
    task_indices = np.asarray([triples[i][2] for i in order], dtype=np.int64)
    signature = hashlib.sha256()
    # Content based and relocatable; deliberately excludes outcomes, sidecars and video pixels.
    signature.update(
        json.dumps(
            {
                "dataset_id": dataset.dataset_id,
                "info": info,
                "episodes": metadata,
                "tasks": tasks,
                "frames": [triples[i] for i in order],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return FrameIndex(
        dataset.dataset_id,
        keys,
        order,
        np.concatenate(([0], np.cumsum([e.length for e in episodes]))).astype(np.int64),
        task_indices,
        tuple(episodes),
        signature.hexdigest(),
    )


def read_episode_outcomes(dataset, index, *, dataset_type: str, source) -> Mapping[int, bool]:
    expected = {e.episode_index for e in index.episodes}
    if dataset_type not in {"rollout", "sft"}:
        raise ValueError("dataset_type must be rollout or sft.")
    if source is None:
        if dataset_type != "sft":
            raise ValueError("Rollout episodes require explicit success/failure outcomes.")
        return dict.fromkeys(expected, True)
    outcomes = {}
    if source.kind == "episode_jsonl":
        for row in load_jsonl(source.path):
            if set(row) != {"episode_index", "success"}:
                raise ValueError("Outcome JSONL requires exactly episode_index and success.")
            ep = _index_integer(row["episode_index"], "outcome episode_index")
            if ep in outcomes:
                raise ValueError(f"Duplicate outcome for episode {ep}.")
            # JSONL is scalar bool, whereas a frame column may be bool[1].
            if not isinstance(row["success"], bool):
                raise ValueError("Outcome JSONL success must be boolean.")
            outcomes[ep] = row["success"]
    elif source.kind == "frame_column":
        values = _columns(dataset, [source.field])[source.field]
        if len(values) != len(index.keys):
            raise ValueError("Outcome column length changed after indexing.")
        for episode, end in zip(index.episodes, index.episode_offsets[1:], strict=True):
            outcomes[episode.episode_index] = _bool(
                values[index.source_indices[end - 1]], f"episode {episode.episode_index} outcome"
            )
    else:
        raise ValueError("Unknown outcome kind.")
    if set(outcomes) != expected:
        raise ValueError("Outcome episode IDs must exactly cover the dataset.")
    if dataset_type == "sft" and not all(outcomes.values()):
        raise ValueError("SFT metadata contains a failed episode; cannot assume all success.")
    return outcomes


def read_interventions(dataset, index, *, field: str | None):
    if field is None:
        return None
    values = _columns(dataset, [field])[field]
    if len(values) != len(index.keys):
        raise ValueError("Intervention column length changed after indexing.")
    return intervention_vector(
        [
            None if values[i] is None else _bool(values[i], f"{field} row {i}")
            for i in index.source_indices
        ],
        length=len(index.keys),
    )


def resolve_policy_dataset_spec(config, manifest) -> ReCapDatasetSpec:
    spec = dataset_spec(config.data, manifest["dataset_id"])
    recorded = manifest["dataset_config"]
    for field in ("image_fields", "state_key", "action_key", "language_key", "tolerance_s"):
        if getattr(spec, field) != recorded[field]:
            raise ValueError(f"RECAP policy {field} must match the annotation dataset_config.")
    return spec
