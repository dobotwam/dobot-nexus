"""Build dense history features with the same frozen Torch encoder as serving."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def prepare_features(config, *, device="cuda:0", batch_size=16):
    from dobot_nexus.data.lerobot.source import build_source
    from dobot_nexus.models.pi0.configuration_pi0 import Pi0Config
    from dobot_nexus.models.pi0.data.pipeline import (
        Pi0Pipeline,
        _convert_image,
    )

    from .configuration import MODEL_TARGET, resolve_config
    from .feature_store import write_episode
    from .torch_vision import TorchFrameTokenEncoder

    if config.model.target != MODEL_TARGET:
        raise ValueError("History feature preparation requires a Pi0MemoryModel config.")
    config, memory = resolve_config(config, mode="train")
    data = config.data
    if not data.get("memory_features") or not data.get("memory_dataset_id"):
        raise ValueError("Set data.memory_features and data.memory_dataset_id.")
    output = Path(data.memory_features).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            f"Use an empty data.memory_features directory; existing caches are immutable: {output}"
        )
    # Asset preparation is one process, independent of the training world size.
    # It reads only images/prompts, before action normalization statistics are needed.
    data.pop("global_batch_size", None)
    data.pop("norm_stats_path", None)
    pipeline = Pi0Pipeline(config, Pi0Config.from_config(config, mode="train"))
    cameras = pipeline.model_config.camera_mapping
    if memory.camera not in cameras:
        raise ValueError(f"Unknown memory camera: {memory.camera}")
    image_key = pipeline.image_fields[cameras[memory.camera]]
    source = build_source(
        data_path=data.path,
        action_key=pipeline.action_key,
        language_key=pipeline.language_key,
        action_horizon=1,
        format_version_name=data.get("format_version", "auto"),
        fps=data.get("fps"),
        tolerance_s=float(data.get("tolerance_s", 1e-4)),
    )
    if not len(source):
        raise ValueError("Cannot prepare history features from an empty dataset.")
    encoder = TorchFrameTokenEncoder(
        memory.vision_checkpoint,
        device=device,
        expected_sha256=memory.vision_sha256,
        batch_size=batch_size,
    )
    episode, features, tasks, pending = None, [], [], []
    completed, prompt_ids = set(), {}
    total_frames = 0

    def flush():
        if pending:
            features.append(encoder(pending))
            pending.clear()

    def publish():
        flush()
        write_episode(
            output,
            episode,
            np.concatenate(features),
            tasks,
            config=memory,
            dataset_id=data.memory_dataset_id,
            image_field=image_key,
            language_field=pipeline.language_key,
        )
        completed.add(episode)

    for index in range(len(source)):
        frame = source[index]
        current = int(frame["episode_index"])
        if episode is not None and current != episode:
            publish()
            features, tasks, pending = [], [], []
        if current in completed:
            raise ValueError("History source episodes must be contiguous and must not repeat.")
        episode = current
        if int(frame["frame_index"]) != len(tasks):
            raise ValueError("History features require dense zero-based frame indices per episode.")
        prompt = pipeline._prompt(frame)
        tasks.append(prompt_ids.setdefault(prompt, len(prompt_ids)))
        pending.append(_convert_image(frame[image_key]))
        total_frames += 1
        if len(pending) == batch_size:
            flush()
    publish()
    return {"directory": str(output), "episodes": len(completed), "frames": total_frames}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Memory training template or recipe.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16, help="Frozen vision encoding batch.")
    parser.add_argument(
        "--override", action="append", default=[], help="Training config key=value."
    )
    args = parser.parse_args(argv)
    from dobot_nexus.core.config import load_config

    config = load_config(args.config, args.override, expected_task="train")
    print(prepare_features(config, device=args.device, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
