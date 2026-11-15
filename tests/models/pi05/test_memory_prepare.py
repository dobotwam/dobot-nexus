"""CPU contracts for the public Torch history-feature preparation entry."""

from __future__ import annotations

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dobot_nexus.core.config import load_config
from dobot_nexus.models.pi05.memory.configuration import resolve_config
from dobot_nexus.models.pi05.memory.feature_store import FrameFeatureStore
from dobot_nexus.models.pi05.memory.prepare import prepare_features
from dobot_nexus.models.pi05.memory.torch_vision import TorchFrameTokenEncoder
from dobot_nexus.models.pi05.torch_core import siglip
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
        resolved, memory = resolve_config(self.config, mode="train")
        store = FrameFeatureStore(result["directory"], memory, dataset_id="test-data-v1")
        manifest = json.loads((Path(result["directory"]) / "manifest.json").read_text())
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
            with (self.subTest(index=index),
                  mock.patch("dobot_nexus.models.pi05.data.source.build_source", return_value=frames),
                  self.assertRaisesRegex(ValueError, "empty dataset|dense|must not repeat")):
                prepare_features(self.config, device="cpu", batch_size=2)


if __name__ == "__main__":
    unittest.main()
