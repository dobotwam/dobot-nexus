from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.core.artifacts import load_manifest
from dobot_nexus.core.config import Config
from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.inference import InferencePolicy, SessionOptions
from dobot_nexus.inference.server import _observation_from_json
from dobot_nexus.training import BCTrainer
from dobot_nexus.training.bc_trainer import _build_scheduler


def _wrap(value):
    if isinstance(value, dict):
        return Config({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


class TrainingInferenceTest(unittest.TestCase):


    def test_train_artifact_infer_and_session(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "tokenizer.model"
            asset.write_text("portable-test-asset", encoding="utf-8")
            config = _wrap(
                {
                    "schema_version": 1,
                    "task": "train",
                    "algorithm": "bc",
                    "exp": {
                        "name": "dummy",
                        "output_dir": directory,
                        "seed": 0,
                    },
                    "model": {
                        "name": "dummy",
                        "target": "tests.fakes:DummyVLAModule",
                        "checkpoint": "unused.pt",
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 3,
                        "asset": str(asset),
                    },
                    "data": {"batch_size": 2, "num_workers": 0},
                    "runtime": {"launcher": "local", "device": "cpu", "precision": "fp32"},
                    "optim": {"name": "adamw", "lr": 0.1, "weight_decay": 0.0},
                    "scheduler": {"name": "constant"},
                    "train": {
                        "max_steps": 2,
                        "grad_accum_steps": 1,
                        "grad_clip_norm": 1.0,
                        "log_every": 1,
                        "save_every": 1,
                    },
                }
            )
            experiment = Experiment.from_config(config)
            self.assertEqual(config.exp.run_dir, str(experiment.run_dir))
            result = BCTrainer(config, experiment).run()
            artifact = Path(result.checkpoint)
            self.assertTrue((artifact / "manifest.yaml").is_file())
            self.assertTrue((artifact / "resume" / "dcp" / ".metadata").is_file())
            self.assertTrue(
                (artifact / "deployment" / "pytorch" / "full_weights.pt").is_file()
            )
            self.assertTrue(
                (artifact / "deployment" / "nexus" / "full_weights.pt").is_file()
            )

            infer_config = _wrap(
                {
                    "schema_version": 1,
                    "task": "infer",
                    "exp": {"name": "dummy_infer", "output_dir": directory},
                    "model": {
                        "name": "dummy",
                        "target": "tests.fakes:DummyVLAModule",
                        "artifact": str(artifact),
                        "checkpoint": None,
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 3,
                    },
                    "runtime": {"device": "cpu"},
                    "inference": {"execute_steps": 2, "seed": 1},
                }
            )
            policy = InferencePolicy(infer_config)
            restored_asset = Path(policy.config.model.asset)
            self.assertEqual(restored_asset.read_text(encoding="utf-8"), "portable-test-asset")
            self.assertEqual(restored_asset.parent, artifact / "deployment" / "assets")
            observation = ObservationBatch(
                images={"base": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
                state=np.zeros((1, 2), dtype=np.float32),
                prompt=["test"],
            )
            output = policy.predict_chunk(observation, SamplingOptions(seed=1))
            self.assertEqual(output.actions.shape, (1, 3, 2))

            session = policy.create_session(SessionOptions(execute_steps=2, seed=1))
            first = session.act(observation)
            second = session.act(observation)
            self.assertEqual(first.shape, (1, 2))
            self.assertEqual(session.remaining_steps, 0)
            np.testing.assert_allclose(first, second)

    def test_self_loading_model_skips_duplicate_weights_and_exports_every_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _wrap(
                {
                    "schema_version": 1,
                    "task": "train",
                    "algorithm": "bc",
                    "exp": {"name": "single_weights", "output_dir": directory, "seed": 0},
                    "model": {
                        "name": "self_loading",
                        "target": "tests.fakes:SelfLoadingDeploymentVLAModule",
                        "checkpoint": "unused.pt",
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 3,
                    },
                    "data": {"batch_size": 2, "num_workers": 0},
                    "runtime": {"launcher": "local", "device": "cpu", "precision": "fp32"},
                    "optim": {"name": "adamw", "lr": 0.1, "weight_decay": 0.0},
                    "scheduler": {"name": "constant"},
                    "train": {
                        "max_steps": 2,
                        "grad_accum_steps": 1,
                        "grad_clip_norm": 1.0,
                        "log_every": 1,
                        "save_every": 1,
                    },
                }
            )
            result = BCTrainer(config, Experiment.from_config(config)).run()
            final_artifact = Path(result.checkpoint)
            first_artifact = final_artifact.parent / "step_00000001"

            first_manifest = load_manifest(first_artifact)
            final_manifest = load_manifest(final_artifact)
            self.assertNotIn("nexus_weights", first_manifest["files"])
            self.assertEqual(
                first_manifest["files"]["policy_info"],
                "deployment/policy_info.txt",
            )
            self.assertFalse((first_artifact / "deployment" / "nexus").exists())
            self.assertNotIn("nexus_weights", final_manifest["files"])
            self.assertEqual(
                final_manifest["files"]["policy_info"],
                "deployment/policy_info.txt",
            )
            self.assertFalse((final_artifact / "deployment" / "nexus").exists())
            self.assertEqual(
                (final_artifact / final_manifest["files"]["policy_info"]).read_bytes(),
                b"test-deployment",
            )
            self.assertNotIn("jax_weights", final_manifest["files"])

            infer_config = _wrap(
                {
                    "schema_version": 1,
                    "task": "infer",
                    "model": {
                        "name": "self_loading",
                        "target": "tests.fakes:SelfLoadingDeploymentVLAModule",
                        "artifact": str(final_artifact),
                        "checkpoint": None,
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 3,
                    },
                    "runtime": {"device": "cpu"},
                }
            )
            policy = InferencePolicy(infer_config)
            self.assertTrue(policy.loaded)

    def test_observation_rejects_camera_batch_mismatch(self):
        with self.assertRaises(ValueError):
            ObservationBatch.from_value(
                {
                    "images": {"base": np.zeros((2, 8, 8, 3), dtype=np.uint8)},
                    "state": np.zeros((1, 2), dtype=np.float32),
                    "prompt": ["test"],
                }
            )

    def test_http_observation_restores_json_image_dtype(self):
        observation = _observation_from_json(
            {
                "images": {"base": np.zeros((8, 8, 3), dtype=np.uint8).tolist()},
                "state": [0.0, 0.0],
                "prompt": ["test"],
            }
        )
        self.assertEqual(observation.images["base"].dtype, np.uint8)
        self.assertEqual(observation.images["base"].shape, (1, 8, 8, 3))

    def test_http_observation_rejects_invalid_json_image_values(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            _observation_from_json(
                {
                    "images": {"base": [[[[-1, 0, 0]]]]},
                    "state": [0.0, 0.0],
                    "prompt": ["test"],
                }
            )


class SchedulerTest(unittest.TestCase):
    def test_explicit_initial_ratio_matches_openpi_schedule(self):
        cfg = _wrap({"train": {"max_steps": 10}})
        cfg.scheduler = _wrap(
            {
                "name": "warmup_cosine",
                "warmup_steps": 2,
                "total_steps": 10,
                "min_lr_ratio": 0.1,
                "warmup_initial_ratio": 1 / 3,
            }
        )
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=1)
        scheduler = _build_scheduler(cfg, optimizer)
        expected = {0: 1 / 3, 1: 2 / 3, 2: 1, 6: 0.55, 10: 0.1}
        for step in range(11):
            if step in expected:
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected[step])
            optimizer.step()
            scheduler.step()


if __name__ == "__main__":
    unittest.main()
