"""Content-addressed frozen features, with atomic per-episode publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .sampling import assemble_memory, select_frames


def content_sha256(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Memory asset not found: {path}")
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not paths:
        raise ValueError(f"Memory asset is empty: {path}")
    for item in paths:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def feature_spec(config):
    return {
        "schema_version": 1,
        "vision_sha256": config.vision_sha256,
        "preprocessing": config.preprocessing,
        "camera": config.camera,
        "tokens_per_image": 16,
        "image_dim": 2048,
        "storage_dtype": "float32",
        "vision_parameter_dtype": "float32",
        "vision_compute_dtype": "bfloat16",
        "pool": "spatial_mean_16x16_to_4x4",
    }


def write_episode(root, episode, features, task_ids, *, config, dataset_id):
    """A new dataset/encoder identity requires a different store directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, np.float32)
    tasks = np.asarray(task_ids, np.int64)
    if (
        features.shape != (len(tasks), 16, 2048)
        or not len(tasks)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Episode features/tasks must be finite aligned nonempty arrays.")
    if not dataset_id or not config.vision_sha256:
        raise ValueError("Feature stores require dataset and vision content identities.")
    from dobot_nexus.data.cache import file_lock

    with file_lock(root / ".write.lock"):
        manifest_path = root / "manifest.json"
        identity = {"dataset_id": dataset_id, "feature_spec": feature_spec(config)}
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {**identity, "episodes": {}}
        )
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError(
                "Stale feature store identity; build a new store "
                "instead of relabeling old features."
            )
        with tempfile.NamedTemporaryFile(dir=root, suffix=".npy", delete=False) as tmp:
            np.save(tmp, features, allow_pickle=False)
            temporary = Path(tmp.name)
        digest = content_sha256(temporary)
        filename = f"episode_{int(episode):06d}_{digest}.npy"
        os.replace(temporary, root / filename)
        manifest["episodes"][str(int(episode))] = {
            "file": filename,
            "frames": len(tasks),
            "sha256": digest,
            "segment_starts": [0] + (np.flatnonzero(tasks[1:] != tasks[:-1]) + 1).tolist(),
        }
        temporary = root / ".manifest.tmp"
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, manifest_path)


class FrameFeatureStore:
    def __init__(self, root, config, *, dataset_id=None, max_cached_episodes=8):
        if type(max_cached_episodes) is not int or max_cached_episodes <= 0:
            raise ValueError("max_cached_episodes must be a positive integer.")
        self.max_cached_episodes = max_cached_episodes
        self.root, self.config = Path(root), config
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("feature_spec") != feature_spec(config):
            raise ValueError(
                "Feature store does not match the memory encoder/preprocessing identity."
            )
        if dataset_id is not None and self.manifest.get("dataset_id") != dataset_id:
            raise ValueError("Feature store belongs to a different dataset.")
        self._arrays = OrderedDict()

    def __getstate__(self):
        # Spawned DataLoader workers reopen their own maps instead of serializing
        # already mapped feature contents into each child process.
        return {**self.__dict__, "_arrays": OrderedDict()}

    def lookup(self, episode, frame):
        key = str(int(episode))
        entry = self.manifest["episodes"][key]
        frame = int(frame)
        if not 0 <= frame < entry["frames"]:
            raise IndexError("History frame is outside the episode.")
        if key not in self._arrays:
            path = self.root / entry["file"]
            if content_sha256(path) != entry["sha256"]:
                raise ValueError(f"Corrupt memory features: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (entry["frames"], 16, 2048) or array.dtype != np.float32:
                raise ValueError("Feature array layout disagrees with its manifest.")
            self._arrays[key] = array
        self._arrays.move_to_end(key)
        while len(self._arrays) > self.max_cached_episodes:
            self._arrays.popitem(last=False)
        starts = entry["segment_starts"]
        start = starts[np.searchsorted(starts, frame, side="right") - 1]
        if self.config.window_frames is None and frame - start >= self.config.max_frames:
            raise ValueError(
                "Global history capacity exceeded; training and runtime must share the same limit."
            )
        indices, positions = select_frames(
            frame,
            self.config.sampled_frames,
            segment_start=start,
            window_frames=self.config.window_frames,
        )
        return assemble_memory(self._arrays[key][indices], positions, self.config)
