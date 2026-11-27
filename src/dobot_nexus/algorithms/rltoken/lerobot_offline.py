"""Read finalized trajectories through Nexus's shared LeRobot reader."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.core.contracts import ObservationBatch
from dobot_nexus.data.images import image_uint8
from dobot_nexus.data.lerobot.dataset import LeRobotSource, episode_indices
from .offline_data import Episode


def _unique_fields(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate key in episode labels.")
    return fields


def _episode_labels(path):
    raw = json.loads(Path(path).read_text(), object_pairs_hook=_unique_fields)
    if not isinstance(raw, dict):
        raise ValueError("Episode labels must be an object keyed by episode ID.")
    labels = {}
    for key, label in raw.items():
        if not re.fullmatch(r"0|[1-9][0-9]*", key):
            raise ValueError(f"Invalid episode ID {key!r}; expected a canonical nonnegative integer.")
        labels[int(key)] = label
    return labels


def read_episodes(config):
    if config.get("last_frame_is_final_observation") is not True:
        raise ValueError("Declare last_frame_is_final_observation=true only for datasets that store it; "
                         "otherwise supply a reader with actual post-action observations.")
    labels = _episode_labels(config.episode_labels)
    source = LeRobotSource(data_path=config.path, action_key=config.action_key,
        language_key=config.get("language_key"), action_horizon=1,
        format_version_name=config.get("format_version", "auto"), fps=config.get("fps"),
        tolerance_s=config.get("tolerance_s", 1e-4))
    for identity, label in labels.items():
        if not label.get("accepted", True):
            continue
        indices = episode_indices(source, [identity])
        frames = [source[i] for i in indices]
        if any("episode_index" in frame and int(frame["episode_index"]) != identity
               for frame in frames):
            raise ValueError(f"Offline frames disagree with episode label {identity}.")
        if len(frames) < 2:
            raise ValueError("Offline episodes need execution frames and a final observation.")
        observations = [ObservationBatch(
            {name: image_uint8(frame[key], root=source.dataset_root)[None] for name, key in config.images.items()},
            np.asarray(frame[config.state_key], dtype=np.float32).reshape(1, -1),
            [str(frame.get("prompt", config.get("prompt", "")))]) for frame in frames]
        switch_key = config.get("policy_switch_key")
        if switch_key:
            for frame, observation in zip(frames, observations, strict=True):
                # Validate booleans instead of treating arbitrary numeric/string labels as true.
                observation.control["rlt_switch_flags"] = np.asarray(frame[switch_key]).reshape(1)
                observation.validate()
        commands = torch.stack([torch.as_tensor(frame[config.action_key]).float().reshape(-1)
                                for frame in frames[:-1]])
        if config.get("reward_key"):
            rewards = torch.tensor([float(frame[config.reward_key]) for frame in frames[:-1]])
        else:
            if not label["terminated"]:
                raise ValueError("Truncated episodes require explicit per-step rewards.")
            rewards = torch.zeros(len(commands))
            rewards[-1] = float(label["success"])
        takeover = torch.tensor([bool(frame[config.takeover_key]) if config.get("takeover_key")
                                 else False for frame in frames[:-1]], dtype=torch.bool)
        yield Episode(str(identity), observations, commands, rewards, takeover,
                      label["origin"], bool(label["terminated"]), success=label.get("success"),
                      supervision_commands=torch.stack([torch.as_tensor(frame[config.supervision_action_key]).float().reshape(-1)
                          for frame in frames[:-1]]) if config.get("supervision_action_key") else None,
                      supervision_mask=torch.tensor([bool(frame[config.supervision_mask_key]) for frame in frames[:-1]],
                          dtype=torch.bool) if config.get("supervision_mask_key") else None,
                      timestamps=torch.tensor([float(frame["timestamp"]) for frame in frames], dtype=torch.float64))
