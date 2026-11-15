from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.config import Config, save_config
from dobot_nexus.core.contracts import ObservationBatch, PolicySpec
from dobot_nexus.inference.policy import InferencePolicy
from dobot_nexus.models.pi05.configuration_pi05 import Pi05Config
from dobot_nexus.models.pi05.data import Pi05Pipeline
from dobot_nexus.models.pi05.data.pipeline import _as_numpy


def _wrap(value):
    if isinstance(value, dict):
        return Config({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _pi_config(root: Path, *, backend: str = "torch") -> Config:
    stats = root / "norm_stats.json"
    stats.write_text(
        json.dumps(
            {
                "state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                "actions": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            }
        ),
        encoding="utf-8",
    )
    tokenizer = root / "tokenizer.model"
    tokenizer.write_bytes(b"unit-test")
    return _wrap(
        {
            "model": {
                "checkpoint": str(root / "unused.pt"),
                "precision": "bf16",
                "state_dim": 2,
                "action_dim": 2,
                "action_horizon": 2,
                "image_size": [224, 224],
                "inference_backend": backend,
            },
            "data": {
                "norm_stats_path": str(stats),
                "tokenizer_path": str(tokenizer),
                "field_mapping": {
                    "images": {
                        "base": "base_image",
                        "left_wrist": "left_image",
                        "right_wrist": "right_image",
                    },
                    "state": "state",
                    "action": "action",
                    "language": "task",
                },
                "transforms": {"action": {"type": "absolute"}},
            },
            "runtime": {"device": "cpu"},
            "inference": {"num_steps": 2, "seed": 7},
        }
    )


class _ArtifactModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_spec = PolicySpec(
            model_name="fake",
            implementation="artifact",
            camera_names=("base",),
            state_dim=2,
            action_dim=2,
            action_horizon=2,
            image_size=(8, 8),
        )


class Pi05RuntimeEncodingTest(unittest.TestCase):
    def test_bfloat16_action_can_cross_numpy_boundary(self) -> None:
        value = torch.ones((1, 2, 2), dtype=torch.bfloat16)
        converted = _as_numpy(value)
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_array_equal(converted, np.ones((1, 2, 2), dtype=np.float32))

    def test_numpy_runtime_encoding_matches_torch_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _pi_config(Path(directory))
            pipeline = Pi05Pipeline(config, Pi05Config.from_config(config, mode="infer"))
            pipeline.tokenizer._processor = SimpleNamespace(
                encode=lambda _text, add_bos: [1, 2]
            )
            image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            observation = ObservationBatch(
                images={"base": image, "left_wrist": image, "right_wrist": image},
                state=np.asarray([[0.25, -0.5]], dtype=np.float32),
                prompt=["pick"],
            )
            encoded, _ = pipeline.transform_runtime_numpy(observation)
            torch_encoded, _ = pipeline.transform_runtime(observation)
            for name in encoded["image"]:
                np.testing.assert_array_equal(
                    encoded["image"][name], torch_encoded["image"][name].numpy()
                )
            np.testing.assert_array_equal(encoded["state"], torch_encoded["state"].numpy())
            np.testing.assert_array_equal(
                encoded["tokenized_prompt"], torch_encoded["tokenized_prompt"].numpy()
            )

    def test_policy_loads_artifact_wrapper_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            (artifact / "nexus").mkdir()
            (artifact / "backend").mkdir()
            (artifact / "nexus" / "full_weights.pt").write_bytes(b"must-not-load")
            (artifact / "backend" / "full_weights.pt").write_bytes(b"constructor-owned")
            save_config(
                {
                    "schema_version": 1,
                    "model": {
                        "target": "tests.fakes:DummyVLAModule",
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 2,
                    },
                    "files": {
                        "weights": "backend/full_weights.pt",
                        "nexus_weights": "nexus/full_weights.pt",
                    },
                },
                artifact / "manifest.yaml",
            )
            save_config(
                {
                    "schema_version": 1,
                    "task": "train",
                    "model": {
                        "target": "tests.fakes:DummyVLAModule",
                        "checkpoint": "unused",
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 2,
                    },
                    "data": {},
                    "train": {},
                },
                artifact / "config_resolved.yaml",
            )
            infer_config = _wrap(
                {
                    "schema_version": 1,
                    "task": "infer",
                    "model": {
                        "target": "tests.fakes:DummyVLAModule",
                        "artifact": str(artifact),
                        "checkpoint": None,
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 2,
                    },
                    "runtime": {"device": "cpu"},
                }
            )
            with (
                mock.patch(
                    "dobot_nexus.inference.policy.build_model",
                    return_value=_ArtifactModel(),
                ),
                mock.patch("torch.load", return_value={}) as torch_load,
            ):
                policy = InferencePolicy(infer_config)
            self.assertTrue(policy.loaded)
            torch_load.assert_called_once_with(
                artifact / "nexus/full_weights.pt", map_location="cpu", weights_only=True
            )
