"""Small Pi0 backbones for CPU training and Artifact integration tests."""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch

from dobot_nexus.core.config import Config, load_config
from dobot_nexus.models.pi0.training.source import load_core_source

ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def small_backbone(expert_width):
    source = load_core_source()
    module = importlib.import_module(source.package_name + ".pi0")
    gemma = module.gemma

    def expert(variant):
        width = {"gemma_2b": 8, "gemma_300m": expert_width}[variant]
        return gemma.Config(
            width=width, depth=1, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4
        )

    vision = dict(width=8, depth=1, mlp_dim=12, num_heads=2, patch_size=(112, 112))
    with (
        mock.patch.object(gemma, "get_config", side_effect=expert),
        mock.patch.object(gemma, "PALIGEMMA_VOCAB_SIZE", 32),
        mock.patch.object(module.siglip, "_decode_variant", return_value=vision),
    ):
        yield source


def train_config(root, variant, source):
    config = load_config(
        ROOT / f"examples/configs/train/pi0_{variant}.yaml",
        [
            f"exp.output_dir={root / 'runs'}",
            f"model.checkpoint={root / 'base.pt'}",
            "model.state_dim=2",
            "model.action_dim=2",
            "model.action_horizon=2",
            "model.max_token_len=12",
            "model.precision=fp32",
            "runtime.precision=fp32",
            "runtime.launcher=local",
            "runtime.fsdp.enabled=false",
            "runtime.device=cpu",
            "data.batch_size=2",
            "train.max_steps=1",
            "train.save_every=1",
            "train.log_every=1",
            "train.resume_limit=0",
            "tracking.tensorboard.enabled=false",
        ],
    )
    config.scheduler = Config(name="constant")
    config.data.norm_stats_path = str(root / "stats.json")
    Path(config.data.norm_stats_path).write_text(
        json.dumps(
            {
                "state": {"mean": [1.0, 2.0], "std": [2.0, 3.0]},
                "actions": {"mean": [3.0, 4.0], "std": [4.0, 5.0]},
            }
        )
    )
    config.data.tokenizer_path = str(root / "tokenizer.model")
    Path(config.data.tokenizer_path).write_bytes(b"test tokenizer")
    base = source.Pi0Config(pi05=False, dtype="float32", action_dim=32, action_horizon=2).create()
    torch.save(base.state_dict(), config.model.checkpoint)
    return config


def infer_config(artifact, variant):
    return load_config(
        ROOT / f"examples/configs/infer/pi0_{variant}.yaml",
        [
            f"model.artifact={artifact}",
            "runtime.device=cpu",
            "model.precision=fp32",
            "inference.num_steps=2",
            "inference.execute_steps=2",
        ],
    )
