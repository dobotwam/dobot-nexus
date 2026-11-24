"""CPU contracts for the public Torch history-feature preparation entry."""

from __future__ import annotations

import builtins
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.artifacts import ArtifactManager, resume_fingerprint
from dobot_nexus.core.config import load_config
from dobot_nexus.core.contracts import PolicyOutput, PolicySpec
from dobot_nexus.models.pi05.configuration_pi05 import Pi05Config
from dobot_nexus.models.pi05.data.pipeline import Pi05Pipeline
from dobot_nexus.models.pi05.deploy.memory_server import MemoryConnection
from dobot_nexus.models.pi05.memory.configuration import resolve_config
from dobot_nexus.models.pi05.memory.data import MemoryPipeline
from dobot_nexus.models.pi05.memory.feature_store import FrameFeatureStore, write_episode
from dobot_nexus.models.pi05.memory.prepare import prepare_features
from dobot_nexus.models.pi05.memory.torch_vision import TorchFrameTokenEncoder
from dobot_nexus.models.pi05.torch_core import siglip
from dobot_nexus.training.runtime import LocalRuntime
from tests.fakes import DummyVLAModule
from tests.models.pi05.test_memory_vision import VARIANT, vision_archive

ROOT = Path(__file__).resolve().parents[3]


def frame(episode, index, prompt, value):
    return {
        "episode_index": episode, "frame_index": index, "task": prompt,
        "observation.images.top_camera": np.full((224, 224, 3), value, np.uint8),
    }


class MemoryPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vision = self.root / "vision.npz"
        np.savez(self.vision, **vision_archive())
        patch = mock.patch.object(siglip, "_decode_variant", return_value=VARIANT)
        patch.start()
        self.addCleanup(patch.stop)
        self.config = load_config(
            ROOT / "examples/configs/train/pi05_memory.yaml",
            [f"model.memory.vision_checkpoint={self.vision}",
             f"data.memory_features={self.root / 'features'}",
             "data.memory_dataset_id=test-data-v1", "model.memory.budget=16",
             "data.global_batch_size=64", "data.norm_stats_path=/missing/not-needed-for-images"],
        )

    def test_torch_preparation_preserves_episode_and_prompt_boundaries(self):
        frames = [frame(0, 0, "first", 0), frame(0, 1, "second", 127),
                  frame(4, 0, "second", 255)]
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"jax", "jaxlib", "flax", "orbax", "openpi"}:
                raise AssertionError(f"Torch feature preparation imported {name}")
            return real_import(name, *args, **kwargs)

        with (mock.patch("builtins.__import__", side_effect=guarded),
              mock.patch("dobot_nexus.models.pi05.data.source.build_source",
                         return_value=frames) as source):
            result = prepare_features(self.config, device="cpu", batch_size=2)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(source.call_args.kwargs["action_horizon"], 1)
        # Single-process asset preparation must not mutate the training recipe.
        self.assertEqual(self.config.data.global_batch_size, 64)
        self.assertEqual(self.config.data.norm_stats_path, "/missing/not-needed-for-images")
        _, memory = resolve_config(self.config, mode="train")
        store = FrameFeatureStore(
            result["directory"], memory, dataset_id="test-data-v1",
            image_field="observation.images.top_camera", language_field="task",
        )
        self.addCleanup(store.close)
        manifest = json.loads((Path(result["directory"]) / "manifest.json").read_text())
        self.assertEqual(manifest["feature_spec"]["schema_version"], 2)
        self.assertEqual(manifest["feature_spec"]["image_field"], "observation.images.top_camera")
        self.assertEqual(manifest["feature_spec"]["language_field"], "task")
        self.assertEqual(set(manifest["episodes"]), {"0", "4"})
        self.assertEqual(manifest["episodes"]["0"]["segment_starts"], [0, 1])
        self.assertEqual(manifest["episodes"]["4"]["segment_starts"], [0])
        encoder = TorchFrameTokenEncoder(self.vision, device="cpu", batch_size=2)
        expected = encoder([f["observation.images.top_camera"] for f in frames[:2]])
        entry = manifest["episodes"]["0"]
        np.testing.assert_array_equal(np.load(Path(result["directory"]) / entry["file"]), expected)
        self.assertEqual(store.lookup(0, 1).features.shape[-2:], (16, 2048))
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare_features(self.config, device="cpu")

    def test_invalid_source_cannot_silently_reorder_history(self):
        cases = [
            [],
            [frame(0, 1, "task", 0)],
            [frame(0, 0, "task", 0), frame(0, 2, "task", 0)],
            [frame(0, 0, "task", 0), frame(1, 0, "task", 0), frame(0, 0, "task", 0)],
        ]
        for index, frames in enumerate(cases):
            self.config.data.memory_features = str(self.root / f"invalid_{index}")
            with (
                self.subTest(index=index),
                mock.patch("dobot_nexus.models.pi05.data.source.build_source", return_value=frames),
                self.assertRaisesRegex(ValueError, "empty dataset|dense|must not repeat"),
            ):
                prepare_features(self.config, device="cpu", batch_size=2)

    def test_preparation_and_serving_encode_the_same_renamed_or_remapped_camera(self):
        images = {
            name: np.full((224, 224, 3), value, np.uint8)
            for name, value in (("top_camera", 0), ("left_camera", 127), ("right_camera", 255))
        }
        row = {"episode_index": 0, "frame_index": 0, "task": "pick"}
        row.update({f"observation.images.{name}": image for name, image in images.items()})
        for camera, top_name, left_name, expected, language in (
            ("front", "front", "left_wrist", "top_camera", "task"),
            ("base", "left_wrist", "base", "left_camera", None),
        ):
            with self.subTest(camera=camera):
                self.config.model["io"] = {"cameras": {
                    top_name: "base_0_rgb", left_name: "left_wrist_0_rgb",
                    "right_wrist": "right_wrist_0_rgb",
                }}
                self.config.model.memory.camera = camera
                self.config.data.field_mapping.language = language
                self.config.data.memory_features = str(self.root / camera)
                with mock.patch("dobot_nexus.models.pi05.data.source.build_source",
                                return_value=[row]):
                    prepare_features(self.config, device="cpu")
                config, memory = resolve_config(self.config, mode="train")
                config.data.pop("global_batch_size")
                config.data.pop("norm_stats_path")
                pi05 = Pi05Config.from_config(config, mode="train")
                store = MemoryPipeline(config, pi05, memory).store
                self.addCleanup(store.close)
                identity = store.manifest["feature_spec"]
                self.assertEqual(identity["schema_version"], 2)
                self.assertEqual(identity["image_field"], f"observation.images.{expected}")
                self.assertEqual(identity["language_field"], language)
                legacy = self.root / f"{camera}_v1"
                write_episode(legacy, 0, np.zeros((1, 16, 2048)), [0], config=memory,
                              dataset_id="test-data-v1")
                with mock.patch.dict(config.data, memory_features=str(legacy)):
                    with self.assertRaisesRegex(ValueError, "rebuild the feature store"):
                        MemoryPipeline(config, pi05, memory)
                saved = store.lookup(0, 0)
                spec = PolicySpec("pi05", "fixture", pi05.camera_names, pi05.state_dim,
                                  pi05.action_dim, pi05.action_horizon, image_size=pi05.image_size)
                policy = SimpleNamespace(
                    config=config,
                    model=SimpleNamespace(pi05_config=pi05),
                    spec=spec,
                    predict_chunk=mock.Mock(return_value=PolicyOutput(
                        np.ones((1, spec.action_horizon, spec.action_dim), np.float32)
                    )),
                )
                encoder = mock.Mock(wraps=TorchFrameTokenEncoder(self.vision, device="cpu"))
                connection = MemoryConnection(policy, encoder, memory, execute_steps=1)
                connection.infer({"images": images, "state": np.zeros(spec.state_dim),
                                  "prompt": "pick"})
                np.testing.assert_array_equal(encoder.call_args.args[0][0], images[expected])
                online = policy.predict_chunk.call_args.args[0].memory
                np.testing.assert_array_equal(online.features[0], saved.features)
                np.testing.assert_array_equal(online.positions[0], saved.positions)

    def test_default_camera_accepts_confirmed_legacy_or_matching_column_identity(self):
        config, memory = resolve_config(self.config, mode="train")
        config.data.pop("global_batch_size")
        config.data.pop("norm_stats_path")
        pi05 = Pi05Config.from_config(config, mode="train")
        for version, fields in (
            (1, {}),
            (2, {"image_field": "observation.images.top_camera", "language_field": "task"}),
        ):
            path = self.root / f"v{version}"
            config.data.memory_features = str(path)
            write_episode(path, 0, np.zeros((1, 16, 2048)), [0], config=memory,
                          dataset_id="test-data-v1", **fields)
            if version == 1:
                with self.assertRaisesRegex(ValueError, "no producer column provenance"):
                    MemoryPipeline(config, pi05, memory)
                config.data.memory_legacy_columns = {
                    "image_field": "observation.images.top_camera", "language_field": "task",
                }
            pipeline = MemoryPipeline(config, pi05, memory)
            self.assertEqual(pipeline.store.manifest["feature_spec"]["schema_version"], version)
            pipeline.store.close()
            for mapping, changes in (
                (config.data.field_mapping, {"language": None}),
                (config.data.field_mapping.images, {"base": "wrong_camera"}),
            ):
                with (
                    self.subTest(version=version, changes=changes),
                    mock.patch.dict(mapping, changes),
                    self.assertRaisesRegex(ValueError, "rebuild the feature store"),
                ):
                    MemoryPipeline(config, pi05, memory)
            config.data.pop("memory_legacy_columns", None)

    def test_saved_checkpoint_resumes_with_confirmed_legacy_store_columns(self):
        config, memory = resolve_config(self.config, mode="train")
        config.data.pop("global_batch_size")
        config.data.pop("norm_stats_path")
        config.runtime.device = "cpu"
        # Historical training resolved the model recipe and tokenizer before checkpointing.
        Pi05Pipeline(config, Pi05Config.from_config(config, mode="train"))
        write_episode(config.data.memory_features, 0, np.ones((1, 16, 2048)), [0],
                      config=memory, dataset_id=config.data.memory_dataset_id)
        runtime = LocalRuntime(config)
        model = DummyVLAModule(config, mode="train")
        optimizer = torch.optim.AdamW(model.parameters())
        model.scale.square().backward()
        optimizer.step()
        checkpoint = ArtifactManager(self.root).save(
            model=model, runtime=runtime, optimizer=optimizer, scheduler=None,
            config=config, step=1, metrics={},
        )
        resumed = load_config(checkpoint / "config_resolved.yaml")
        self.assertNotIn("memory_legacy_columns", resumed.data)
        resumed.data.memory_cache_bytes = 1024 * 1024
        resumed.data.memory_legacy_columns = {
            "image_field": "observation.images.top_camera", "language_field": "task",
        }
        pi05 = Pi05Config.from_config(resumed, mode="train")
        with (mock.patch.dict(resumed.data.memory_legacy_columns, language_field="wrong"),
              self.assertRaisesRegex(ValueError, "confirm its original")):
            MemoryPipeline(resumed, pi05, memory)
        store = MemoryPipeline(resumed, pi05, memory).store
        self.addCleanup(store.close)
        np.testing.assert_array_equal(store.lookup(0, 0).features, 1)
        restored = DummyVLAModule(resumed, mode="train")
        state = runtime.load_training_state(
            model=restored, optimizer=torch.optim.AdamW(restored.parameters()), scheduler=None,
            checkpoint_dir=checkpoint, expected_fingerprint=resume_fingerprint(resumed),
            trusted_checkpoint=True,
        )
        self.assertEqual(state["step"], 1)
        torch.testing.assert_close(restored.scale, model.scale)

    def test_legacy_left_camera_producer_cannot_silently_supply_default_top_camera(self):
        left, top = "observation.images.left_camera", "observation.images.top_camera"
        row = frame(0, 0, "pick", 0)
        row[left] = np.full_like(row[top], 197)
        producer_columns = {"image_field": left, "language_field": "task"}

        def legacy_write(*args, image_field, language_field, schema_version, **kwargs):
            self.assertEqual((image_field, language_field), (left, "task"))
            write_episode(*args, **kwargs)  # Legacy producer omitted both columns.

        self.config.data.field_mapping.images.base = left
        with (
            mock.patch("dobot_nexus.models.pi05.data.source.build_source", return_value=[row]),
            mock.patch("dobot_nexus.models.pi05.memory.feature_store.write_episode", legacy_write),
        ):
            prepare_features(self.config, device="cpu")
        config, memory = resolve_config(self.config, mode="train")
        config.data.pop("global_batch_size")
        config.data.pop("norm_stats_path")
        pi05 = Pi05Config.from_config(config, mode="train")
        config.data.field_mapping.images.base = top
        for confirmation in (None, True, {}, {"image_field": top}, producer_columns,
                             {"image_field": top, "language_field": None}):
            with self.subTest(confirmation=confirmation):
                config.data.memory_legacy_columns = confirmation
                with self.assertRaisesRegex(ValueError, "confirm its original"):
                    MemoryPipeline(config, pi05, memory)

        config.data.field_mapping.images.base = left
        config.data.memory_legacy_columns = producer_columns
        store = MemoryPipeline(config, pi05, memory).store
        self.addCleanup(store.close)
        worker = pickle.loads(pickle.dumps(store))
        self.addCleanup(worker.close)
        expected = TorchFrameTokenEncoder(self.vision, device="cpu")([row[left]])
        np.testing.assert_array_equal(worker.lookup(0, 0).features, expected[0])
        state = store.__getstate__()
        del state["_confirmed_legacy_columns"]
        with self.assertRaisesRegex(ValueError, "no producer column provenance"):
            FrameFeatureStore.__new__(FrameFeatureStore).__setstate__(state)

    def test_training_rejects_prepared_store_after_only_dataset_image_column_changes(self):
        with mock.patch("dobot_nexus.models.pi05.data.source.build_source",
                        return_value=[frame(0, 0, "pick", 123)]):
            prepare_features(self.config, device="cpu")
        config, memory = resolve_config(self.config, mode="train")
        config.data.pop("global_batch_size")
        config.data.pop("norm_stats_path")
        model = Pi05Config.from_config(config, mode="train")
        original = MemoryPipeline(config, model, memory).store
        self.assertEqual(original.manifest["feature_spec"]["image_field"],
                         "observation.images.top_camera")
        original.close()
        config.data.field_mapping.images.base = "observation.images.different_camera"
        with self.assertRaisesRegex(ValueError, "rebuild the feature store"):
            MemoryPipeline(config, model, memory)


if __name__ == "__main__":
    unittest.main()
