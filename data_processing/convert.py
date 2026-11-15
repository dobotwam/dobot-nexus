"""Convert trusted Dobot PKL episodes to LeRobot v2.1 for downstream training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from .fields import extra_schema, extra_values

CAMERAS = ("top_camera", "left_camera", "right_camera")


def prepare_image(jpeg, *, roi=None, size=224):
    import cv2

    raw = np.frombuffer(jpeg, dtype=np.uint8) if isinstance(jpeg, bytes) else np.asarray(jpeg)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Camera frame is not a valid encoded image.")
    if roi is not None:
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            raise ValueError(f"ROI {roi} exceeds image shape {image.shape}.")
        image = image[y : y + h, x : x + w]
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    matrix = np.array(
        [[scale, 0, (size - int(w * scale)) / 2], [0, scale, (size - int(h * scale)) / 2]]
    )
    return np.ascontiguousarray(cv2.warpAffine(image, matrix, (size, size))[..., ::-1])


def _boolean(value, name, *, unknown=False):
    if unknown and value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if name == "episode_success" and isinstance(value, str) and value in {"success", "failure"}:
        return value == "success"
    raise ValueError(f"{name} must be an explicit boolean; got {value!r}.")


def _vector(frame, prefix):
    values = [
        np.asarray(frame[f"{prefix}{side}_dobot_arm_joints"], dtype=np.float32)
        for side in ("left", "right")
    ]
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in values):
        raise ValueError(f"{prefix or 'state'} requires two finite 7-D arm vectors.")
    return np.concatenate(values)


def _episode_outcome(frames, source, *, required, path):
    labels = [frame.get("episode_success") for frame in frames]
    labels.append(source.get("success"))
    known = [_boolean(value, "episode_success") for value in labels if value is not None]
    if len(set(known)) > 1:
        raise ValueError(f"{path}: conflicting episode success labels.")
    if not known and required:
        raise ValueError(
            f"{path}: episode_success is required; record an outcome or set source.success."
        )
    return known[0] if known else None


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def convert(config, output: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    if set(config) - {
        "sources", "fps", "top_roi_xywh", "image_size", "require_success", "extra_fields"
    }:
        raise ValueError("Unknown conversion config keys.")
    require_success = config.get("require_success", False)
    if type(require_success) is not bool:
        raise ValueError("require_success must be a boolean.")
    fps, size = config.get("fps", 20), config.get("image_size", 224)
    if type(fps) is not int or fps <= 0 or type(size) is not int or size <= 0:
        raise ValueError("fps and image_size must be positive integers.")
    roi = config.get("top_roi_xywh", [145, 30, 400, 400])
    if roi is not None and (len(roi) != 4 or any(type(x) is not int for x in roi)):
        raise ValueError("top_roi_xywh must contain four integers or be null.")
    sources = config.get("sources", [])
    jobs = []
    seen = set()
    for source in sources:
        if set(source) - {"path", "task", "success", "is_human"}:
            raise ValueError("Unknown source config keys.")
        path = Path(source["path"]).expanduser().resolve()
        files = sorted(path.glob("*.pkl")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"No PKL files in {path}.")
        for file in files:
            if file in seen:
                raise ValueError(f"Duplicate source: {file}")
            seen.add(file)
            jobs.append((file, source))
    if not jobs:
        raise ValueError("sources must select at least one episode.")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dobot-convert-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        (root / "meta").mkdir(parents=True)
        tasks, episodes, provenance, all_states, all_actions = [], [], [], [], []
        cursor = 0
        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        schema = pa.schema(
            [
                *[
                    (k, pa.int64())
                    for k in (
                        "index",
                        "episode_index",
                        "frame_index",
                        "task_index",
                        "source_frame_index",
                    )
                ],
                ("timestamp", pa.float32()),
                ("source_timestamp", pa.float64()),
                ("observation.state", pa.list_(pa.float32(), 14)),
                ("action", pa.list_(pa.float32(), 14)),
                ("is_success", pa.bool_()),
                ("is_human", pa.bool_()),
                *[(f"observation.images.{c}", image_type) for c in CAMERAS],
            ]
        )
        extras = config.get("extra_fields", {})
        for field in extra_schema(extras, reserved=schema.names):
            schema = schema.append(field)
        for ep, (path, source) in enumerate(jobs):
            blob = path.read_bytes()
            frames = pickle.loads(blob)
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"{path} must contain a non-empty frame list.")
            times = np.asarray([f["timestamp"] for f in frames], dtype=np.float64)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"{path}: timestamps must be finite and strictly increasing.")
            prompts = [source.get("task", f.get("task")) for f in frames]
            if not isinstance(prompts[0], str) or not prompts[0].strip() or len(set(prompts)) != 1:
                raise ValueError(f"{path}: requires one non-empty task per episode.")
            task = prompts[0]
            if task not in tasks:
                tasks.append(task)
            outcome = _episode_outcome(
                frames, source, required=require_success, path=path
            )
            rows, states, actions = [], [], []
            for i, frame in enumerate(frames):
                human = _boolean(
                    source.get("is_human", frame.get("is_human")), "is_human", unknown=True
                )
                if frame.get("is_human") is not None and human != _boolean(
                    frame["is_human"], "is_human"
                ):
                    raise ValueError(f"{path}: conflicting is_human labels.")
                state, action = _vector(frame, ""), _vector(frame, "action_")
                states.append(state)
                actions.append(action)
                row = dict(
                    index=cursor + i,
                    episode_index=ep,
                    frame_index=i,
                    task_index=tasks.index(task),
                    source_frame_index=i,
                    timestamp=i / fps,
                    source_timestamp=float(times[i]),
                    is_success=outcome,
                    is_human=human,
                )
                row.update({"observation.state": state.tolist(), "action": action.tolist()})
                row.update(extra_values(frame, extras))
                for camera in CAMERAS:
                    image = prepare_image(
                        frame[f"{camera}_frame"],
                        roi=roi if camera == "top_camera" else None,
                        size=size,
                    )
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="PNG")
                    row[f"observation.images.{camera}"] = {"bytes": buffer.getvalue(), "path": None}
                rows.append(row)
            destination = root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, destination)
            if not pq.read_table(destination).equals(table):
                raise RuntimeError(f"Parquet round-trip failed: {destination}")
            n = len(frames)
            episodes.append(dict(episode_index=ep, tasks=[task], length=n))
            gaps = np.diff(times)
            provenance.append(
                dict(
                    episode_index=ep,
                    source=str(path),
                    source_sha256=hashlib.sha256(blob).hexdigest(),
                    length=n,
                    observed_median_dt=float(np.median(gaps)) if n > 1 else None,
                    observed_max_dt=float(gaps.max()) if n > 1 else None,
                    episode_success=outcome,
                    unknown_human_frames=sum(r["is_human"] is None for r in rows),
                    unknown_extra_frames={
                        name: sum(row[name] is None for row in rows) for name in extras
                    },
                )
            )
            all_states.extend(states)
            all_actions.extend(actions)
            cursor += n
        features = {}
        for field in schema:
            if field.name.startswith("observation.images."):
                features[field.name] = {
                    "dtype": "image",
                    "shape": [size, size, 3],
                    "names": ["height", "width", "channel"],
                }
            else:
                vector = field.name in {"observation.state", "action"}
                dtype = "float32" if vector or field.name == "timestamp" else str(field.type)
                if field.name == "source_timestamp":
                    dtype = "float64"
                if field.name in extras:
                    dtype = extras[field.name]["dtype"]
                features[field.name] = {
                    "dtype": dtype,
                    "shape": [14 if vector else 1],
                    "names": None,
                }
        _json(
            root / "meta/info.json",
            dict(
                codebase_version="v2.1",
                robot_type="dobot_dual_arm",
                fps=fps,
                total_episodes=len(episodes),
                total_frames=cursor,
                total_tasks=len(tasks),
                total_videos=0,
                total_chunks=(len(episodes) + 999) // 1000,
                chunks_size=1000,
                splits={"train": f"0:{len(episodes)}"},
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                features=features,
            ),
        )
        stats = {}
        for name, values in (("observation.state", all_states), ("action", all_actions)):
            a = np.asarray(values, dtype=np.float64)
            stats[name] = {
                k: v.tolist()
                for k, v in {
                    "min": a.min(0),
                    "max": a.max(0),
                    "mean": a.mean(0),
                    "std": a.std(0),
                    "count": np.array([len(a)]),
                }.items()
            }
        _json(root / "meta/stats.json", stats)
        _jsonl(root / "meta/episodes.jsonl", episodes)
        _jsonl(root / "meta/tasks.jsonl", [dict(task_index=i, task=t) for i, t in enumerate(tasks)])
        _jsonl(root / "meta/sources.jsonl", provenance)
        _json(
            root / "meta/conversion.json",
            dict(
                schema_version=2,
                config=config,
                require_success=require_success,
                extra_fields=extras,
                time_mode="recorded_steps",
                resampled=False,
                resize="opencv_warp_affine",
                image_encoding="PNG",
                top_roi_xywh=roi,
            ),
        )
        if output.exists():
            raise FileExistsError(output)
        root.rename(output)
    return output


def main(*, converter=convert):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = converter(yaml.safe_load(args.config.read_text()), args.output)
    print(f"Wrote {result}; recorded steps preserved. FPS is an index clock, not resampling.")


if __name__ == "__main__":
    main()
