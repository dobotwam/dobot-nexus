"""Map immutable LeRobot frames and RECAP returns to Value samples."""

from __future__ import annotations

import numpy as np
import torch

from dobot_nexus.data.images import convert_image
from dobot_nexus.data.lerobot.source import build_source

from .._validation import nonempty
from ..configuration import CAMERAS
from ..data import _index_integer


class ReCapValueDataset(torch.utils.data.Dataset):
    def __init__(self, config, index, returns, *, cache_files=2):
        if returns.index.dataset_signature != index.dataset_signature:
            raise ValueError("Value targets and dataset signatures disagree.")
        self.config, self.index, self.returns = config, index, returns
        spec = config.dataset
        self.source = build_source(
            data_path=str(spec.path),
            action_key=spec.action_key,
            language_key=spec.language_key,
            action_horizon=1,
            format_version_name=spec.format_version,
            fps=spec.fps,
            tolerance_s=spec.tolerance_s,
        )
        if hasattr(self.source, "cache_capacity"):
            self.source.cache_capacity = cache_files
            # Read immutable Arrow tables before workers fork, sharing their buffers.
            if cache_files >= len(self.source.paths):
                for start in self.source.starts[:-1]:
                    self.source._row(int(start))
        if len(self.source) != len(index.keys):
            raise ValueError("Value dataset length changed after indexing.")

    def __len__(self):
        return len(self.index.keys)

    def __getitem__(self, row):
        from .processing import ValueSample

        frame = self.source[int(self.index.source_indices[row])]
        key = self.index.keys[row]
        if (
            _index_integer(frame["episode_index"], "episode_index"),
            _index_integer(frame["frame_index"], "frame_index"),
        ) != (key.episode_index, key.frame_index):
            raise ValueError("Value source row does not match the indexed FrameKey.")
        return ValueSample(
            key=key,
            images={
                camera: convert_image(frame[self.config.dataset.image_fields[camera]])
                for camera in CAMERAS
            },
            prompt=nonempty(frame.get("prompt"), "value task"),
            target_value=float(self.returns.returns[row]),
            state=np.asarray(frame[self.config.dataset.state_key], dtype=np.float32),
        )
