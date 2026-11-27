"""Shared image/video dispatch for local LeRobot datasets."""

from __future__ import annotations

import bisect
import io
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.data.images import read_contained_bytes
from dobot_nexus.data.lerobot import LeRobotSource
from dobot_nexus.data.lerobot.metadata import find_parquet_files, load_info, load_jsonl


def build_source(**kwargs):
    info = load_info(Path(kwargs["data_path"]))
    kinds = {f.get("dtype") for f in info.get("features", {}).values()}
    if "image" in kinds:
        if "video" in kinds:
            raise ValueError("LeRobotSource does not mix inline-image and video cameras in one dataset.")
        return ImageLeRobotSource(**kwargs)
    return LeRobotSource(**kwargs)


class ImageLeRobotSource(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data_path,
        action_key,
        language_key,
        action_horizon,
        format_version_name="v2.1",
        fps=None,
        tolerance_s=1e-4,
    ):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "LeRobot inline images require pyarrow; install dobot-nexus[pi05]."
            ) from exc

        self.dataset_root = Path(data_path).resolve()
        info = load_info(self.dataset_root)
        if info.get("codebase_version") != "v2.1" or format_version_name not in {"auto", "v2.1"}:
            raise ValueError("Inline images require LeRobot v2.1.")
        if fps is not None and fps != info["fps"]:
            raise ValueError("data.fps must match meta.info.fps; resample before training.")
        self.detected_format_version = "v2.1"
        self.resolved_fps = int(fps or info["fps"])
        self.action_key, self.language_key = action_key, language_key
        if type(action_horizon) is not int or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer.")
        self.action_horizon = action_horizon
        self.images = [k for k, f in info["features"].items() if f.get("dtype") == "image"]
        self.paths = find_parquet_files(self.dataset_root)
        self.starts = np.concatenate(
            ([0], np.cumsum([pq.read_metadata(p).num_rows for p in self.paths]))
        )
        episodes = load_jsonl(self.dataset_root / "meta/episodes.jsonl")
        lengths = {int(ep["episode_index"]): int(ep["length"]) for ep in episodes}
        if len(lengths) != len(episodes) or any(n < 1 for n in lengths.values()):
            raise ValueError("Episode metadata requires unique indices and positive lengths.")
        tasks = load_jsonl(self.dataset_root / "meta/tasks.jsonl")
        self.tasks = {int(t["task_index"]): t["task"] for t in tasks}
        if len(self.tasks) != len(tasks):
            raise ValueError("Task metadata requires unique indices.")
        self.ends = {}
        current, count, cursor = None, 0, 0
        for path in self.paths:
            # Validate identities once, without touching image bytes or action windows.
            table = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
            columns = [table[k].to_numpy() for k in ("episode_index", "frame_index", "task_index")]
            if any(not np.issubdtype(c.dtype, np.integer) for c in columns):
                raise ValueError("Episode, frame and task identities must be integers without nulls.")
            for ep, frame, task in zip(*columns, strict=True):
                ep, frame, task = int(ep), int(frame), int(task)
                if ep != current:
                    if current is not None:
                        if count != lengths.get(current):
                            raise ValueError("Episode metadata length does not match Parquet rows.")
                        self.ends[current] = cursor
                    if ep in self.ends or ep not in lengths:
                        raise ValueError("Parquet episodes must be contiguous and declared in metadata.")
                    current, count = ep, 0
                if frame != count or task not in self.tasks:
                    raise ValueError("Parquet frame order or task identity is invalid.")
                count += 1
                cursor += 1
        if current is not None:
            if count != lengths.get(current):
                raise ValueError("Episode metadata length does not match Parquet rows.")
            self.ends[current] = cursor
        if set(self.ends) != set(lengths) or cursor != len(self):
            raise ValueError("Episode metadata does not match inline-image rows.")
        # Whole-episode subsets use the verified Parquet row order, including across shards.
        self.episode_bounds = {ep: (end - lengths[ep], end) for ep, end in self.ends.items()}
        self.cache, self.image_cache = OrderedDict(), OrderedDict()
        self.cache_capacity = 2
        self.group_starts = []
        for path in self.paths:
            metadata = pq.read_metadata(path)
            self.group_starts.append(np.concatenate(([0], np.cumsum([
                metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]))))

    def __len__(self):
        return int(self.starts[-1])

    def _file(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return bisect.bisect_right(self.starts, index) - 1

    def _numeric(self, file):
        import pyarrow.parquet as pq
        if file not in self.cache:
            names = [name for name in pq.read_schema(self.paths[file]).names if name not in self.images]
            table = pq.read_table(self.paths[file], columns=names)
            actions = np.asarray(table[self.action_key].to_pylist(), dtype=np.float32)
            if actions.ndim != 2 or not np.isfinite(actions).all():
                raise ValueError("Actions must be finite vectors.")
            self.cache[file] = (table, actions)
        self.cache.move_to_end(file)
        while len(self.cache) > self.cache_capacity:
            self.cache.popitem(last=False)
        return self.cache[file]

    def _row(self, index):
        import pyarrow.parquet as pq
        file = self._file(index)
        offset = index - int(self.starts[file])
        table, _ = self._numeric(file)
        item = table.slice(offset, 1).to_pylist()[0]
        group = bisect.bisect_right(self.group_starts[file], offset) - 1
        key = (file, group)
        if key not in self.image_cache:
            self.image_cache[key] = pq.ParquetFile(self.paths[file]).read_row_group(group, columns=self.images)
        self.image_cache.move_to_end(key)
        while len(self.image_cache) > self.cache_capacity:
            self.image_cache.popitem(last=False)
        image_row = self.image_cache[key].slice(offset-int(self.group_starts[file][group]), 1)
        item.update(image_row.to_pylist()[0])
        return item

    def _actions(self, index, end):
        # Slice only numeric columns, including episodes split across Parquet files.
        stop, chunks = min(index + self.action_horizon, end), []
        cursor = index
        while cursor < stop:
            file = self._file(cursor)
            _, actions = self._numeric(file)
            count = min(stop, int(self.starts[file+1])) - cursor
            offset = cursor-int(self.starts[file])
            chunks.append(actions[offset:offset+count])
            cursor += count
        result = np.concatenate(chunks)
        if len(result) < self.action_horizon:
            result = np.concatenate((result, np.repeat(result[-1:], self.action_horizon-len(result), axis=0)))
        return result.copy()

    def __getitem__(self, index):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "LeRobot inline images require Pillow; install dobot-nexus[pi05]."
            ) from exc

        index = int(index)
        item = self._row(index)
        end = self.ends[int(item["episode_index"])]
        item[self.action_key] = self._actions(index, end)
        for key in self.images:
            cell = item[key]
            if isinstance(cell, dict):
                data = cell.get("bytes")
                if data is None and cell.get("path"):
                    data = read_contained_bytes(self.dataset_root, cell["path"])
            else:
                data = cell
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise ValueError(f"Inline image {key} requires encoded bytes or a file path.")
            with Image.open(io.BytesIO(data)) as image:
                item[key] = np.array(image.convert("RGB"))
        item["prompt"] = (item.get(self.language_key) if self.language_key else None) or self.tasks[
            int(item["task_index"])
        ]
        return item
