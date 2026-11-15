from __future__ import annotations

import copy
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from dobot_nexus.algorithms.recap.configuration import (
    load_offline_config,
    parse_offline_config,
)
from dobot_nexus.algorithms.recap.value.configuration import job_device

from .fakes import offline_config


class ConfigTest(unittest.TestCase):
    def test_value_jobs_require_one_supported_device(self):
        with mock.patch.dict(os.environ, WORLD_SIZE="1"):
            self.assertEqual(job_device("cpu"), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "CPU or one CUDA"):
                job_device("meta")
            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                    job_device("cuda:0")
        with mock.patch.dict(os.environ, WORLD_SIZE="2"):
            with self.assertRaisesRegex(ValueError, "single-device"):
                job_device("cpu")

    def test_output_cannot_modify_source_and_wrong_value_resolution_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            with self.assertRaisesRegex(ValueError, "outside"):
                dataclasses.replace(config, annotation_dir=config.dataset.path / "annotations")
            for path in (
                config.value.siglip_path,
                config.value.gemma3_path,
                config.value.tokenizer_path,
            ):
                path.mkdir(exist_ok=True)
            architecture = config.value.architecture()
            vision = dict(architecture["vision"])
            vision["image_size"] = 224
            (config.value.siglip_path / "config.json").write_text(
                json.dumps({"model_type": "siglip", "vision_config": vision})
            )
            (config.value.gemma3_path / "config.json").write_text(
                json.dumps({"model_type": "gemma3_text", **architecture["language"]})
            )
            with self.assertRaisesRegex(ValueError, "image_size=224"):
                config.value.validate_assets()
            vision["image_size"] = 384
            (config.value.siglip_path / "config.json").write_text(
                json.dumps({"model_type": "siglip", "vision_config": vision})
            )
            config.value.validate_assets()


    def test_strict_offline_config_does_not_require_policy_or_backbone_load(self):
        raw = {
            "data": {
                "path": "/absent/data",
                "dataset_id": "usb",
                "dataset_type": "sft",
                "field_mapping": {"images": {k: k for k in ("base", "left_wrist", "right_wrist")}},
            },
            "annotations": {"output_dir": "/absent/annotations"},
            "value": {k: "/absent/asset" for k in ("siglip_path", "gemma3_path", "tokenizer_path")},
            "train": {"output_dir": "/absent/run"},
        }
        self.assertEqual(parse_offline_config(raw).dataset_type, "sft")
        for section, field, value in (
            ("returns", "gamma", 0.99),
            ("advantage", "n_step", True),
            ("train", "resume", True),
            ("train", "log_every", 0),
            ("train", "log_every", True),
            ("train", "stop_steps", 0),
            ("train", "stop_steps", 20001),
            ("train", "cache_files", 0),
            ("value", "num_bins", 5),
            ("returns", "scale", 0),
            ("data", "dataset_type", "rollout"),
        ):
            mutated = copy.deepcopy(raw)
            mutated.setdefault(section, {})[field] = value
            with self.assertRaises(ValueError):
                parse_offline_config(mutated)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offline.yaml"
            path.write_text(yaml.safe_dump(raw))
            config = load_offline_config(path, ["advantage.n_step=3"])
            self.assertEqual(config.advantage.n_step, 3)
