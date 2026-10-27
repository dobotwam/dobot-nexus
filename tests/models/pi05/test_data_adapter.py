"""CPU contract tests for the Pi0.5 data adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.config import clone_config
from dobot_nexus.data import NormStats, save_norm_stats
from dobot_nexus.models.pi05.data.pipeline import EncodedObservation, Pi05Pipeline
from dobot_nexus.models.pi05.data_adapter_pi05 import Pi05DataAdapter

from .fakes import FakeObservation

ROOT = Path(__file__).resolve().parents[3]


def _tensor_mock():
    value = mock.Mock(spec=torch.Tensor)
    value.to.return_value = value
    return value


class Pi05DataAdapterTest(unittest.TestCase):
    def test_move_encoded_preserves_extension_kwargs_and_nested_device_copies(self):
        adapter = Pi05DataAdapter.__new__(Pi05DataAdapter)
        observation = FakeObservation({}, {}, torch.zeros(1, 2), None, None)
        features, positions, mask, extra = [_tensor_mock() for _ in range(4)]
        kwargs = {"memory_inputs": (features, positions, mask), "extra": [extra, {"steps": 3}]}
        encoded = EncodedObservation(observation, np.zeros((1, 2)), kwargs)
        moved_obs = object()
        device = torch.device("cuda", 0)
        with mock.patch(
            "dobot_nexus.models.pi05.data_adapter_pi05._move_observation",
            return_value=moved_obs,
        ):
            moved = adapter.move_encoded(encoded, device)
        self.assertIs(moved.observation, moved_obs)
        self.assertIs(moved.raw_state, encoded.raw_state)
        self.assertIsNot(moved.model_kwargs, kwargs)
        self.assertIsInstance(moved.model_kwargs["memory_inputs"], tuple)
        self.assertEqual(moved.model_kwargs["extra"][1], {"steps": 3})
        for tensor in (features, positions, mask, extra):
            tensor.to.assert_called_once_with(device, non_blocking=True)

    def test_train_batch_uses_non_blocking_device_copies(self):
        tensors = {
            "image": {"base": _tensor_mock()},
            "image_mask": {"base": _tensor_mock()},
            "state": _tensor_mock(),
            "tokenized_prompt": _tensor_mock(),
            "tokenized_prompt_mask": _tensor_mock(),
            "actions": _tensor_mock(),
        }
        adapter = Pi05DataAdapter.__new__(Pi05DataAdapter)
        adapter.observation_type = FakeObservation
        device = torch.device("cuda", 0)

        batch = adapter.encode_train_batch(tensors, device)

        for tensor in (
            tensors["image"]["base"],
            tensors["image_mask"]["base"],
            tensors["state"],
            tensors["tokenized_prompt"],
            tensors["tokenized_prompt_mask"],
            tensors["actions"],
        ):
            tensor.to.assert_called_once_with(device, non_blocking=True)
        self.assertIs(batch["actions"], tensors["actions"])


class NormalizationTest(unittest.TestCase):
    def test_versioned_artifact_normalization_preserves_codec_and_saved_config(self):
        with tempfile.TemporaryDirectory() as root:
            for kind in ("quantile", "zscore"):
                with self.subTest(kind=kind):
                    saved = {"type": kind, "version": 1}
                    legacy = self.pipeline(root, saved)
                    current = self.pipeline(root, kind)
                    values = np.array([[-4.0, 3.0], [10.0, 3.0]], dtype=np.float32)
                    np.testing.assert_array_equal(
                        legacy.normalize(values, "state"), current.normalize(values, "state")
                    )
                    np.testing.assert_array_equal(
                        legacy.unnormalize(values, "actions"), current.unnormalize(values, "actions")
                    )
                    self.assertEqual(legacy.root_config["data"]["normalization"], saved)
            for invalid in (
                {"type": "quantile"}, {"type": "quantile", "version": 2},
                {"type": "quantile", "version": True},
                {"type": "quantile", "version": 1, "clip": True},
                {"type": [], "version": 1}, [], None,
            ):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "normalization"):
                    self.pipeline(root, invalid)

    def test_two_camera_training_and_runtime_use_the_same_missing_view_mask(self):
        from dobot_nexus.core.contracts import ObservationBatch

        with tempfile.TemporaryDirectory() as root:
            original = self.pipeline(root)
            config = clone_config(original.root_config)
            config.data.field_mapping.images = {"base": "base", "left_wrist": "hand"}
            model = SimpleNamespace(**vars(original.model_config))
            model.camera_mapping = {"base": "base_0_rgb", "hand": "left_wrist_0_rgb"}
            pipeline = Pi05Pipeline(config, model)
            pipeline.tokenize_prompt = lambda *args: (np.array([1]), np.array([True]))
            base, hand = np.full((224, 224, 3), 42, np.uint8), np.full((224, 224, 3), 84, np.uint8)
            state = np.array([0, 1], np.float32)
            training = pipeline.transform_frame({
                "base": base, "hand": hand, "observation.state": state,
                "action": np.zeros((3, 2), np.float32), "task": "insert",
            })
            runtime, _ = pipeline.transform_runtime_numpy(ObservationBatch(
                {"base": base[None], "hand": hand[None]}, state[None], ["insert"]
            ))
            for key in training["image"]:
                np.testing.assert_array_equal(training["image"][key], runtime["image"][key][0])
                self.assertEqual(training["image_mask"][key], runtime["image_mask"][key][0])
            self.assertFalse(training["image_mask"]["right_wrist_0_rgb"])
            self.assertFalse(training["image"]["right_wrist_0_rgb"].any())
            model.camera_mapping["extra"] = "right_wrist_0_rgb"
            with self.assertRaisesRegex(ValueError, "same slots"):
                Pi05Pipeline(config, model)

    def test_prefetch_configuration_respects_worker_mode(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = self.pipeline(root)
            pipeline.root_config = clone_config(pipeline.root_config)
            source = torch.utils.data.TensorDataset(torch.arange(8))
            source.dataset_root = root
            source.detected_format_version = "v2.1"
            source.resolved_fps = 20
            data = pipeline.root_config.data
            data.batch_size = 2
            data.prefetch_factor = 3
            with mock.patch.object(pipeline, "build_source", return_value=source):
                for workers in (2, 0):
                    data.num_workers = workers
                    loader, _ = pipeline.build_dataloader()
                    self.assertEqual(loader.num_workers, workers)
                    self.assertEqual(loader.prefetch_factor, 3 if workers else None)
                    self.assertEqual(loader.persistent_workers, bool(workers))
                for invalid in (0, -1, True, 1.5):
                    data.prefetch_factor = invalid
                    with self.assertRaisesRegex(ValueError, "data.prefetch_factor"):
                        pipeline.build_dataloader()

    def pipeline(self, root, mode="quantile", *, quantiles=True):
        stats = NormStats(
            mean=np.array([2.0, 4.0]),
            std=np.array([1.0, 2.0]),
            q01=np.array([-2.0, 3.0]) if quantiles else None,
            q99=np.array([6.0, 3.0]) if quantiles else None,
        )
        path = Path(root) / "norm_stats.json"
        save_norm_stats(path, {"state": stats, "actions": stats})
        return Pi05Pipeline(
            {
                "task": "infer",
                "data": {
                    "normalization": mode,
                    "norm_stats_path": str(path),
                    "field_mapping": {
                        "images": {k: k for k in ("base", "left_wrist", "right_wrist")}
                    },
                },
            },
            SimpleNamespace(
                state_dim=2, action_dim=2, model_action_dim=32, action_horizon=3, max_token_len=200
            ),
        )

    def test_quantiles_cover_outliers_constant_dimensions_and_action_inverse(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = self.pipeline(root)
            values = np.array([[-4.0, 3.0], [10.0, 3.0]], dtype=np.float32)
            expected = (values - [-2.0, 3.0]) / (np.array([8.0, 0.0]) + 1e-6) * 2 - 1
            np.testing.assert_array_equal(pipeline.normalize(values, "state"), expected)
            self.assertGreater(pipeline.normalize(values, "state")[1, 0], 1)
            normalized = pipeline.normalize(values, "actions")
            decoded = pipeline.decode_actions(normalized[None], np.zeros((1, 2)))[0]
            np.testing.assert_allclose(decoded, values, rtol=0, atol=1e-6)

    def test_zscore_and_invalid_quantile_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = self.pipeline(root, "zscore", quantiles=False)
            np.testing.assert_allclose(
                pipeline.normalize(np.array([4.0, 8.0]), "state"),
                np.array([2.0, 4.0]) / (np.array([1.0, 2.0]) + 1e-6),
            )
            with self.assertRaisesRegex(ValueError, "q01/q99"):
                self.pipeline(root, quantiles=False)
            with self.assertRaisesRegex(ValueError, "normalization"):
                self.pipeline(root, "invalid")

    def test_quantile_dimension_order_and_finiteness_validation(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = self.pipeline(root)
            for low, high in (
                ([0], [1]),
                ([2, 0], [1, 1]),
                ([0, 0], [1, np.inf]),
                (0.0, 1.0),
                ([0, 0], [1]),
            ):
                stats = NormStats(np.zeros(2), np.ones(2), np.asarray(low), np.asarray(high))
                path = Path(root) / "invalid_stats.json"
                save_norm_stats(path, {"state": stats, "actions": stats})
                with self.assertRaisesRegex(ValueError, "q01/q99|statistics"):
                    pipeline._set_stats(path)


class PipelineAssetsTest(unittest.TestCase):
    def test_deploy_tokenizer_overrides_artifact_only_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_tokenizer = root / "training.model"
            deploy_tokenizer = root / "deploy.model"
            training_tokenizer.write_bytes(b"training")
            deploy_tokenizer.write_bytes(b"deploy")
            stats = root / "norm_stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "state": {"mean": [0.0], "std": [1.0]},
                        "actions": {"mean": [0.0], "std": [1.0]},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "task": "infer",
                "data": {
                    "norm_stats_path": str(stats),
                    "tokenizer_path": str(training_tokenizer),
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
            }
            model_config = SimpleNamespace(max_token_len=200, action_dim=1, state_dim=1)
            with mock.patch.dict(
                "os.environ", {"OPENPI_TOKENIZER_PATH": str(deploy_tokenizer)}, clear=True
            ):
                inference = Pi05Pipeline(config, model_config)
                self.assertEqual(inference.tokenizer_path, deploy_tokenizer.resolve())
                config["task"] = "train"
                training = Pi05Pipeline(config, model_config)
                self.assertEqual(training.tokenizer_path, training_tokenizer.resolve())

    def test_pi05_training_does_not_treat_raw_lerobot_stats_as_final_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "lerobot_dataset"
            metadata = dataset / "meta"
            metadata.mkdir(parents=True)
            stats = metadata / "stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "observation.state": {
                            "mean": [1.0, 2.0],
                            "std": [3.0, 4.0],
                        },
                        "action": {
                            "mean": [5.0, 6.0],
                            "std": [7.0, 8.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "task": "train",
                "data": {
                    "path": str(dataset),
                    "field_mapping": {
                        "images": {
                            "base": "base_image",
                            "left_wrist": "left_image",
                            "right_wrist": "right_image",
                        },
                        "state": "observation.state",
                        "action": "action",
                        "language": "task",
                    },
                    "transforms": {"action": {"type": "absolute"}},
                },
            }
            model_config = SimpleNamespace(max_token_len=200, action_dim=2, state_dim=2)

            with mock.patch.dict("os.environ", {}, clear=True):
                pipeline = Pi05Pipeline(config, model_config)

            self.assertIsNone(pipeline.norm_stats_path)
            self.assertIsNone(pipeline.stats)
            self.assertNotIn("norm_stats_path", config["data"])
            self.assertEqual(
                config["data"]["tokenizer_path"],
                str(
                    (
                        ROOT
                        / "src/dobot_nexus/models/pi05/assets/pi05_tokenizer.model"
                    ).resolve()
                ),
            )

    def test_repository_contains_portable_pi05_tokenizer(self) -> None:
        tokenizer = ROOT / "src/dobot_nexus/models/pi05/assets/pi05_tokenizer.model"
        self.assertTrue(tokenizer.is_file())
        self.assertGreater(tokenizer.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
