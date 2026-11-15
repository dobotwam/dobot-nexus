"""CPU coverage for the actual Pi0 dataset and inference codecs."""

import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from dobot_nexus.core.config import load_config
from dobot_nexus.core.contracts import ObservationBatch
from dobot_nexus.models.pi0.configuration_pi0 import Pi0Config
from dobot_nexus.models.pi0.data.pipeline import Pi0Pipeline

ROOT = Path(__file__).resolve().parents[3]


class Pi0PipelineTest(unittest.TestCase):
    def test_inline_images_and_action_codecs_match_train_and_inference(self):
        for mode in ("absolute", "delta"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "meta").mkdir()
                (root / "data").mkdir()
                cameras = {
                    name: f"observation.images.{name}_camera"
                    for name in ("top", "left", "right")
                }
                (root / "meta/info.json").write_text(json.dumps({
                    "codebase_version": "v2.1", "fps": 10,
                    "features": {key: {"dtype": "image"} for key in cameras.values()},
                }))
                (root / "meta/episodes.jsonl").write_text(
                    json.dumps({"episode_index": 0, "length": 2}))
                (root / "meta/tasks.jsonl").write_text(
                    json.dumps({"task_index": 0, "task": "pick object"}))
                encoded_image = io.BytesIO()
                Image.new("RGB", (12, 8), (25, 50, 75)).save(encoded_image, format="PNG")
                rows = [{
                    "episode_index": 0, "frame_index": i, "task_index": 0,
                    "observation.state": [1.0 + i, 10.0 + i],
                    "action": [3.0 + i, 30.0 + i],
                    **{key: {"bytes": encoded_image.getvalue()} for key in cameras.values()},
                } for i in range(2)]
                pq.write_table(pa.Table.from_pylist(rows), root / "data/000.parquet")
                config = load_config(ROOT / "examples/configs/train/pi0.yaml", [
                    "model.checkpoint=unused", "model.state_dim=2", "model.action_dim=2",
                    "model.action_horizon=3", f"data.path={root}",
                    f"data.transforms.action.type={mode}",
                    f"exp.run_dir={root / 'run'}",
                ], expected_task="train")
                if mode == "delta":
                    config.data.transforms.action.delta_mask = [True, False]
                pipeline = Pi0Pipeline(config, Pi0Config.from_config(config, mode="train"))
                logger_name = f"dobot_nexus.{config.exp.run_dir}"
                with self.assertLogs(logger_name, level=logging.INFO):
                    loader, resolved = pipeline.build_dataloader()
                self.assertTrue(Path(resolved.norm_stats_path).is_file())
                self.assertEqual(resolved.format_version, "v2.1")
                self.assertEqual(resolved.fps, 10)
                for index in range(2):
                    raw = loader.dataset.source[index]
                    frame = pipeline.transform_frame(raw)
                    obs = ObservationBatch.from_value({
                        "images": {
                            public: raw[pipeline.image_fields[core]]
                            for public, core in pipeline.model_config.camera_mapping.items()
                        },
                        "state": raw["observation.state"], "prompt": raw["prompt"],
                    })
                    encoded, state = pipeline.transform_runtime_numpy(obs)
                    np.testing.assert_allclose(encoded["state"][0], frame["state"])
                    np.testing.assert_array_equal(
                        encoded["tokenized_prompt"][0], frame["tokenized_prompt"])
                    for camera in frame["image"]:
                        np.testing.assert_array_equal(
                            encoded["image"][camera][0], frame["image"][camera])
                    decoded = pipeline.decode_actions(frame["actions"][None], state)
                    expected = np.array([rows[min(index + j, 1)]["action"] for j in range(3)])
                    np.testing.assert_allclose(decoded[0], expected, rtol=1e-5, atol=1e-5)
                batch = next(iter(loader))
                self.assertEqual(tuple(batch["actions"].shape), (1, 3, 32))
                self.assertEqual(tuple(batch["state"].shape), (1, 32))


if __name__ == "__main__":
    unittest.main()
