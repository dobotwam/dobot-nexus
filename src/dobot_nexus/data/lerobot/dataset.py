from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dobot_nexus.data.lerobot.metadata import format_version, load_info, load_jsonl


class LeRobotSource(torch.utils.data.Dataset):
    """Version-aware LeRobot source that returns raw frames and action chunks."""

    def __init__(
        self,
        *,
        data_path: str,
        action_key: str,
        language_key: str | None,
        action_horizon: int,
        format_version_name: str,
        fps: int | None,
        tolerance_s: float,
    ) -> None:
        path = Path(data_path).expanduser()
        kwargs: dict[str, Any] = {}
        if path.exists():
            root = path.resolve()
            info = load_info(root)
            kwargs.update(repo_id=root.name, root=str(root))
        else:
            info = {}
            kwargs.update(repo_id=data_path)
        resolved_fps = int(fps) if fps is not None else int(info.get("fps", 10))
        if resolved_fps <= 0:
            raise ValueError("data.fps must be positive.")
        detected = format_version(info.get("codebase_version", info.get("version")))
        requested = format_version(format_version_name)
        if str(format_version_name).strip().lower() == "auto":
            requested = "auto"
        if requested not in {"auto", "v2.1", "v3.0"}:
            raise ValueError("data.format_version must be auto, v2.1, or v3.0.")
        if requested != "auto" and detected in {"v2.1", "v3.0"} and requested != detected:
            raise ValueError(
                f"LeRobot format mismatch: configured={requested}, detected={detected}."
            )
        if detected == "v2.1" and path.exists():
            self.dataset = LocalLeRobotV21Dataset(
                root=root,
                info=info,
                action_key=action_key,
                action_horizon=action_horizon,
                tolerance_s=tolerance_s,
            )
        else:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ModuleNotFoundError:
                try:
                    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
                except ModuleNotFoundError as exc:
                    raise ImportError(
                        "LeRobot training requires `pip install -e '.[pi05]'`."
                    ) from exc
            kwargs.update(
                delta_timestamps={
                    action_key: [index / float(resolved_fps) for index in range(action_horizon)]
                },
                tolerance_s=tolerance_s,
            )
            self.dataset = LeRobotDataset(**kwargs)
        self.dataset_root = Path(getattr(self.dataset, "root", path)).resolve()
        self.resolved_fps = resolved_fps
        self.detected_format_version = detected
        self.language_key = language_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        if "prompt" in item:
            return item
        if self.language_key is not None and item.get(self.language_key) is not None:
            item["prompt"] = item[self.language_key]
            return item
        if item.get("task") is not None:
            item["prompt"] = item["task"]
            return item
        task_index = item.get("task_index")
        tasks = getattr(getattr(self.dataset, "meta", None), "tasks", None)
        if task_index is not None and tasks is not None:
            task_index = int(task_index.item() if hasattr(task_index, "item") else task_index)
            item["prompt"] = (
                tasks.get(task_index, tasks.get(str(task_index)))
                if isinstance(tasks, dict)
                else tasks[task_index]
            )
        return item


@dataclass(frozen=True)
class _LocalV21Metadata:
    tasks: dict[int, str]


class LocalLeRobotV21Dataset(torch.utils.data.Dataset):
    """Read immutable local LeRobot v2.1 data with current dependencies."""

    def __init__(
        self,
        *,
        root: Path,
        info: Mapping[str, Any],
        action_key: str,
        action_horizon: int,
        tolerance_s: float,
    ) -> None:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Local LeRobot v2.1 training requires `datasets`."
            ) from exc

        self.root = root
        self.info = dict(info)
        self.action_key = action_key
        self.action_horizon = action_horizon
        self.tolerance_s = tolerance_s
        self.video_keys = [
            str(key)
            for key, feature in self.info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]
        self.image_keys = [
            str(key)
            for key, feature in self.info.get("features", {}).items()
            if feature.get("dtype") == "image"
        ]
        if not self.video_keys and not self.image_keys:
            raise ValueError("The local LeRobot v2.1 reader requires image or video camera features.")
        if self.video_keys:
            try:
                from lerobot.datasets.video_utils import get_safe_default_codec
            except ModuleNotFoundError as exc:
                raise ImportError("LeRobot v2.1 video cameras require `lerobot`.") from exc
            self.video_backend = get_safe_default_codec()

        data_root = self.root / "data"
        if not data_root.is_dir():
            raise FileNotFoundError(f"LeRobot v2.1 data directory not found: {data_root}.")
        self.hf_dataset = load_dataset("parquet", data_dir=str(data_root), split="train")
        # Selecting columns before indexing avoids decoding future camera frames
        # when assembling an action chunk from image-backed datasets.
        self.action_dataset = self.hf_dataset.select_columns([self.action_key])

        episodes = load_jsonl(self.root / "meta" / "episodes.jsonl")
        self.episode_bounds: dict[int, tuple[int, int]] = {}
        cursor = 0
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            self.episode_bounds[episode_index] = (cursor, cursor + length)
            cursor += length
        if cursor != len(self.hf_dataset):
            raise ValueError(
                "LeRobot v2.1 episode lengths do not match parquet rows: "
                f"metadata={cursor}, parquet={len(self.hf_dataset)}."
            )

        tasks = {
            int(item["task_index"]): str(item["task"])
            for item in load_jsonl(self.root / "meta" / "tasks.jsonl")
        }
        self.meta = _LocalV21Metadata(tasks=tasks)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.hf_dataset[int(index)])
        episode_index = int(item["episode_index"])
        try:
            episode_start, episode_end = self.episode_bounds[episode_index]
        except KeyError as exc:
            raise KeyError(f"Unknown LeRobot episode_index={episode_index}.") from exc

        action_indices = [
            max(episode_start, min(episode_end - 1, int(index) + offset))
            for offset in range(self.action_horizon)
        ]
        action_rows = self.action_dataset[action_indices][self.action_key]
        item[self.action_key] = np.asarray(action_rows, dtype=np.float32)

        if not self.video_keys:
            return item
        timestamp = float(item["timestamp"])
        episode_chunk = episode_index // int(self.info.get("chunks_size", 1000))
        video_template = str(self.info["video_path"])
        from lerobot.datasets.video_utils import decode_video_frames

        for video_key in self.video_keys:
            relative = video_template.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            video_path = self.root / relative
            if not video_path.is_file():
                raise FileNotFoundError(f"LeRobot video not found: {video_path}.")
            frames = decode_video_frames(
                video_path,
                [timestamp],
                self.tolerance_s,
                self.video_backend,
            )
            item[video_key] = frames.squeeze(0)
        return item


def episode_indices(source, episode_ids):
    """Resolve a whole-episode subset without decoding images or action windows."""
    wanted = {int(value) for value in episode_ids}
    if not wanted:
        raise ValueError("episode_ids must be nonempty.")
    dataset = getattr(source, "dataset", source)
    bounds = getattr(dataset, "episode_bounds", None)
    metadata = Path(getattr(source, "dataset_root", ".")) / "meta/episodes.jsonl"
    if bounds is None and metadata.is_file():
        bounds, cursor = {}, 0
        for episode in load_jsonl(metadata):
            length = int(episode["length"])
            bounds[int(episode["episode_index"])] = (cursor, cursor + length)
            cursor += length
        if cursor != len(source):
            raise ValueError("Episode metadata does not describe the full source.")
    if bounds is not None:
        missing = wanted - set(bounds)
        indices = [i for episode in sorted(wanted & set(bounds)) for i in range(*bounds[episode])]
    else:
        table = getattr(dataset, "hf_dataset", None)
        if table is None:
            raise ValueError("Dataset must expose episode bounds or an episode_index column.")
        column = [int(value) for value in table["episode_index"]]
        missing = wanted - set(column)
        indices = [i for i, episode in enumerate(column) if episode in wanted]
    if missing:
        raise ValueError(f"Unknown episode IDs: {sorted(missing)}")
    return indices
