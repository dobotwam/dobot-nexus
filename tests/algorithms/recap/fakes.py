from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.algorithms.recap.configuration import parse_offline_config


def write_dataset(root: Path, *, order=(3, 8), split_first=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    lengths = {3: 5, 8: 3}
    cameras = {key: f"observation.images.{key}" for key in ("base", "left_wrist", "right_wrist")}
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "chunks_size": 1000,
        "total_frames": 8,
        "total_episodes": 2,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            **{k: {"dtype": "video", "shape": [16, 16, 3]} for k in cameras.values()},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(
        json.dumps({"observation.state": {"min": [0, 0], "max": [4, 1]}})
    )
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "insert plug"}) + "\n"
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": ep, "length": lengths[ep]}) + "\n" for ep in order)
    )
    for position, ep in enumerate(order):
        rows = [
            {
                "episode_index": ep,
                "frame_index": f,
                "task_index": 0,
                "timestamp": f / 10,
                "observation.state": [float(f), 0.0],
                "action": [float(f + 1), float(ep)],
                "is_success": ep == 3,
                "is_human": 1 <= f <= 2,
            }
            for f in range(lengths[ep])
        ]
        parts = [rows[:2], rows[2:]] if split_first and position == 0 else [rows]
        for part, subset in enumerate(parts):
            pq.write_table(
                pa.Table.from_pylist(subset), root / f"data/chunk-000/{position:03d}_{part}.parquet"
            )
    return cameras

def offline_config(root: Path, *, order=(3, 8), split_first=False):
    dataset = root / "dataset"
    cameras = write_dataset(dataset, order=order, split_first=split_first)
    return parse_offline_config(
        {
            "data": {
                "path": str(dataset),
                "dataset_id": "fixture",
                "dataset_type": "rollout",
                "format_version": "v2.1",
                "outcome": {"kind": "frame_column", "field": "is_success"},
                "intervention_field": "is_human",
                "field_mapping": {"images": cameras},
            },
            "annotations": {"output_dir": str(root / "annotations")},
            "value": {k: str(root / k) for k in ("siglip_path", "gemma3_path", "tokenizer_path")},
            "train": {
                "output_dir": str(root / "value_run"),
                "device": "cpu",
                "num_workers": 0,
                "max_steps": 2,
                "batch_size": 2,
                "save_every": 2,
                "grad_accum_steps": 2,
                "val_ratio": 0.0,
            },
            "labeling": {"device": "cpu", "batch_size": 3},
            "advantage": {"n_step": 2},
        }
    )

class FakeSource(torch.utils.data.Dataset):
    def __init__(self, data_path, action_horizon, **kwargs):
        import pyarrow.parquet as pq

        self.dataset_root = Path(data_path)
        self.detected_format_version = "v2.1"
        self.resolved_fps = 10
        self.rows = [
            r
            for p in sorted((self.dataset_root / "data").rglob("*.parquet"))
            for r in pq.read_table(p).to_pylist()
        ]
        self.horizon = action_horizon

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        indices = [i for i, r in enumerate(self.rows) if r["episode_index"] == row["episode_index"]]
        row["action"] = np.asarray(
            [self.rows[min(index + k, indices[-1])]["action"] for k in range(self.horizon)],
            dtype=np.float32,
        )
        row["observation.state"] = np.asarray(row["observation.state"], dtype=np.float32)
        for camera in ("base", "left_wrist", "right_wrist"):
            row[f"observation.images.{camera}"] = np.full((16, 16, 3), index * 10, dtype=np.uint8)
        row["prompt"] = "insert plug"
        return row

class FakeTokenizer:
    def encode(self, text, **kwargs):
        return [
            1,
            5 if "Advantage: positive" in text else 6 if "Advantage: negative" in text else 2,
            3,
        ]

    def __call__(self, prompts, **kwargs):
        encoded = [self.encode(p) for p in prompts]
        limit = kwargs.get("max_length", 200)
        size = min(max(map(len, encoded)), limit)
        tokens = torch.zeros(len(prompts), size, dtype=torch.long)
        mask = torch.zeros_like(tokens)
        for i, ids in enumerate(encoded):
            ids = ids[:size]
            tokens[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        return {"input_ids": tokens, "attention_mask": mask}

class FakeValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 201)

    def gradient_checkpointing_enable(self):
        pass

    def forward(self, observation):
        pixels = observation["image_tensor"].mean((1, 2, 3, 4))[:, None]
        return self.linear(pixels).float()
