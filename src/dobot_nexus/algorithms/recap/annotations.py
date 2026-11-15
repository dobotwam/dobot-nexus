"""Complete, keyed, content-checked sidecars; never mutate the source dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dobot_nexus.core.config import to_plain_dict

from .configuration import RECIPE, SCHEMA_VERSION
from .targets import finite_vector, intervention_vector, label_advantages, resolve_return_scale
from .types import FrameKey, PolicyLabelTable, ReturnScale, ReturnTable

KEY_COLUMNS = ("dataset_id", "episode_index", "frame_index")


def check_return_job(returned, config) -> None:
    """Keep value training and labeling bound to the same return recipe."""
    if returned.manifest["dataset_type"] != config.dataset_type or returned.manifest[
        "returns_config"
    ] != asdict(config.returns):
        raise ValueError(
            "Value/label job must use the return job's dataset type and return config."
        )
    expected, current = returned.manifest["dataset_config"], asdict(config.dataset)
    for key in ("image_fields", "state_key", "action_key", "language_key", "fps", "tolerance_s"):
        if expected[key] != current[key]:
            raise ValueError(f"Value/label dataset setting {key} differs from the return job.")


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("RECAP sidecars require pyarrow; install the pi05 extra.") from exc
    return pa, pq


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            to_plain_dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def load_annotation_manifest(annotation_dir, *, stage: str) -> dict:
    directory = Path(annotation_dir)
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("recipe") != RECIPE
        or manifest.get("dataset_type") not in {"rollout", "sft"}
        or manifest.get("stages", {}).get(stage) is not True
        or manifest.get("stages", {}).get("returns") is not True
    ):
        raise ValueError(f"Incomplete or unsupported RECAP {stage} manifest: {directory}.")
    ReturnScale(**manifest["return_scale"])
    return manifest


def _check_index(manifest, index) -> None:
    if (
        manifest["dataset_signature"] != index.dataset_signature
        or manifest["dataset_id"] != index.dataset_id
        or manifest["num_frames"] != len(index.keys)
        or manifest["num_episodes"] != len(index.episodes)
    ):
        raise ValueError("RECAP annotations do not match this dataset's frame index/signature.")


def _key_columns(index):
    pa, _ = _arrow()
    return {
        "dataset_id": pa.array([key.dataset_id for key in index.keys], type=pa.string()),
        "episode_index": pa.array([key.episode_index for key in index.keys], type=pa.int64()),
        "frame_index": pa.array([key.frame_index for key in index.keys], type=pa.int64()),
    }


def _read_table(directory, name, manifest, index, fields):
    pa, pq = _arrow()
    path = Path(directory) / name
    if manifest.get("files", {}).get(name) != file_digest(path):
        raise ValueError(f"RECAP sidecar checksum mismatch: {name}.")
    table = pq.read_table(path)
    expected = {
        "dataset_id": pa.string(),
        "episode_index": pa.int64(),
        "frame_index": pa.int64(),
        **fields,
    }
    if len(set(table.column_names)) != len(table.column_names) or set(table.column_names) != set(
        expected
    ):
        raise ValueError(f"Invalid RECAP columns in {name}.")
    for key, dtype in expected.items():
        if table.schema.field(key).type != dtype:
            raise ValueError(f"RECAP {name}.{key} must have dtype {dtype}.")
        if key != "human" and table[key].null_count:
            raise ValueError(f"RECAP {name}.{key} contains nulls.")
    keys = [FrameKey(*v) for v in zip(*(table[k].to_pylist() for k in KEY_COLUMNS), strict=True)]
    if len(set(keys)) != len(keys) or set(keys) != set(index.keys):
        raise ValueError(f"Duplicate/missing/unexpected frame keys in {name}.")
    lookup = {key: row for row, key in enumerate(keys)}
    return table.take(pa.array([lookup[key] for key in index.keys]))


def _vector(value, index, name):
    v = finite_vector(value, name).astype(np.float32)
    if len(v) != len(index.keys) or not np.isfinite(v).all():
        raise ValueError(f"{name} length/range does not match the FrameIndex.")
    return v


@contextmanager
def _stage(directory: Path, filenames: tuple[str, ...]):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".write.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"RECAP output has an active or interrupted writer: {lock}.") from exc
    os.close(descriptor)
    try:
        if any((directory / name).exists() for name in filenames):
            raise FileExistsError(
                "RECAP stage output already exists; use a new annotation directory."
            )
        with tempfile.TemporaryDirectory(prefix=".recap-", dir=directory) as temporary:
            yield Path(temporary)
    finally:
        lock.unlink()


def _commit(directory, temporary, manifest, filenames):
    for name in filenames:
        manifest.setdefault("files", {})[name] = file_digest(temporary / name)
    (temporary / "manifest.json").write_text(
        json.dumps(to_plain_dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for name in (*filenames, "manifest.json"):
        os.replace(temporary / name, directory / name)


def write_returns(
    output_dir,
    index,
    rewards,
    returns,
    scale,
    *,
    outcome_signature: str,
    config,
    interventions=None,
) -> Path:
    pa, pq = _arrow()
    output_dir = Path(output_dir)
    rewards = _vector(rewards, index, "rewards")
    returns = _vector(returns, index, "returns")
    if (
        scale != resolve_return_scale(index, config.returns.c_fail_coef)
        or returns.min() < -1
        or returns.max() > 0
    ):
        raise ValueError("Return targets disagree with ReturnScale.")
    if interventions is not None:
        interventions = intervention_vector(interventions, length=len(returns))
    if bool(config.intervention_field) != (interventions is not None):
        raise ValueError("Intervention annotations disagree with the configured source field.")
    human = interventions.tolist() if interventions is not None else [None] * len(index.keys)
    table = pa.table(
        {
            **_key_columns(index),
            "task_index": pa.array(index.task_indices, type=pa.int64()),
            "reward": pa.array(rewards, type=pa.float32()),
            "return": pa.array(returns, type=pa.float32()),
            "human": pa.array(human, type=pa.bool_()),
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "recipe": RECIPE,
        "dataset_type": config.dataset_type,
        "dataset_id": index.dataset_id,
        "dataset_signature": index.dataset_signature,
        "outcome_signature": outcome_signature,
        "return_scale": asdict(scale),
        "num_frames": len(index.keys),
        "num_episodes": len(index.episodes),
        "returns_config": asdict(config.returns),
        "dataset_config": asdict(config.dataset),
        "outcome_source": (asdict(config.outcome) if config.outcome is not None else None),
        "intervention_field": config.intervention_field,
        "intervention_signature": object_digest(human),
        "source_counts": {
            "human": human.count(True),
            "model": human.count(False),
            "unknown": human.count(None),
        },
        "stages": {"returns": True, "labels": False},
        "files": {},
    }
    with _stage(output_dir, ("returns.parquet", "manifest.json")) as temporary:
        pq.write_table(table, temporary / "returns.parquet")
        _commit(output_dir, temporary, manifest, ("returns.parquet",))
    return output_dir / "manifest.json"


def read_returns(annotation_dir, *, expected_index) -> ReturnTable:
    pa, _ = _arrow()
    manifest = load_annotation_manifest(annotation_dir, stage="returns")
    _check_index(manifest, expected_index)
    table = _read_table(
        annotation_dir,
        "returns.parquet",
        manifest,
        expected_index,
        {
            "task_index": pa.int64(),
            "reward": pa.float32(),
            "return": pa.float32(),
            "human": pa.bool_(),
        },
    )
    if table["task_index"].to_pylist() != expected_index.task_indices.tolist():
        raise ValueError("Return sidecar task indices disagree with the dataset.")
    r, g = [_vector(table[k].to_numpy(), expected_index, k) for k in ("reward", "return")]
    scale = ReturnScale(**manifest["return_scale"])
    if scale != resolve_return_scale(expected_index, manifest["returns_config"]["c_fail_coef"]) or (
        g.min() < -1 or g.max() > 0
    ):
        raise ValueError("Return sidecar normalization is inconsistent.")
    for begin, end in zip(
        expected_index.episode_offsets[:-1], expected_index.episode_offsets[1:], strict=True
    ):
        if not np.allclose(
            g[begin:end], r[begin:end].astype(np.float64)[::-1].cumsum()[::-1], rtol=1e-6, atol=1e-5
        ):
            raise ValueError("Return sidecar is not an episode-local reward cumulative sum.")
    human = table["human"].to_pylist()
    if object_digest(human) != manifest["intervention_signature"]:
        raise ValueError("Intervention signature mismatch.")
    if manifest["intervention_field"] is None:
        if any(h is not None for h in human):
            raise ValueError("Unknown action provenance must remain null.")
        interventions = None
    else:
        interventions = intervention_vector(human, length=len(expected_index.keys))
    return ReturnTable(expected_index, r, g, interventions, scale, manifest)


def write_labels(annotation_dir, *, index, predicted_values, labels, value_checkpoint_id, config):
    pa, pq = _arrow()
    directory = Path(annotation_dir)
    with _stage(directory, ("values.parquet", "advantages.parquet")) as temporary:
        returned = read_returns(directory, expected_index=index)
        manifest = dict(returned.manifest)
        if manifest["stages"].get("labels"):
            raise FileExistsError("RECAP labels are already complete.")
        if manifest["dataset_type"] != config.dataset_type or manifest["returns_config"] != asdict(
            config.returns
        ):
            raise ValueError("Label job must use the same dataset type and return configuration.")
        values = _vector(predicted_values, index, "predicted_values")
        if values.min() < -1.000001 or values.max() > 0.000001:
            raise ValueError("Predicted values must be normalized to [-1,0].")
        advantages = _vector(labels.advantages, index, "advantages")
        expected = label_advantages(
            advantages,
            index.task_indices,
            positive_ratio=config.advantage.positive_ratio,
            interventions=returned.interventions,
            force_intervention_positive=config.advantage.force_intervention_positive,
        )
        if (
            np.asarray(labels.positive).dtype != np.bool_
            or not np.array_equal(labels.positive, expected.positive)
            or labels.thresholds != expected.thresholds
        ):
            raise ValueError(
                "Policy labels must use the configured task thresholds and intervention rule."
            )
        if not isinstance(value_checkpoint_id, str) or not value_checkpoint_id:
            raise ValueError("A value checkpoint identifier is required.")
        pq.write_table(
            pa.table(
                {**_key_columns(index), "predicted_value": pa.array(values, type=pa.float32())}
            ),
            temporary / "values.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    **_key_columns(index),
                    "advantage": pa.array(advantages, type=pa.float32()),
                    "positive": pa.array(labels.positive, type=pa.bool_()),
                }
            ),
            temporary / "advantages.parquet",
        )
        manifest.update(
            advantage_config=asdict(config.advantage),
            thresholds=labels.thresholds,
            positive_fraction=float(labels.positive.mean()),
            value_checkpoint_id=value_checkpoint_id,
            value_config=asdict(config.value),
        )
        manifest["stages"]["labels"] = True
        _commit(directory, temporary, manifest, ("values.parquet", "advantages.parquet"))
    return directory / "manifest.json"


def read_policy_labels(annotation_dir, *, expected_index) -> PolicyLabelTable:
    pa, _ = _arrow()
    manifest = load_annotation_manifest(annotation_dir, stage="labels")
    _check_index(manifest, expected_index)
    table = _read_table(
        annotation_dir,
        "advantages.parquet",
        manifest,
        expected_index,
        {"advantage": pa.float32(), "positive": pa.bool_()},
    )
    advantages = _vector(table["advantage"].to_numpy(), expected_index, "advantages")
    positive = table["positive"].to_numpy().astype(np.bool_)
    returned = read_returns(annotation_dir, expected_index=expected_index)
    expected = label_advantages(
        advantages,
        expected_index.task_indices,
        positive_ratio=manifest["advantage_config"]["positive_ratio"],
        interventions=returned.interventions,
        force_intervention_positive=manifest["advantage_config"]["force_intervention_positive"],
    )
    if {
        int(k): v for k, v in manifest["thresholds"].items()
    } != expected.thresholds or not np.array_equal(positive, expected.positive):
        raise ValueError("Policy labels disagree with the saved task thresholds/interventions.")
    return PolicyLabelTable(
        expected_index, positive, advantages, ReturnScale(**manifest["return_scale"]), manifest
    )
