from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.core.config import clone_config, save_config
from dobot_nexus.inference.policy import InferencePolicy


class InferenceDimensionsTest(unittest.TestCase):
    def artifact(self, root, state_dim, action_dim):
        model = {"target": "tests.fakes:DummyVLAModule", "state_dim": state_dim,
                 "action_dim": action_dim, "action_horizon": 3}
        save_config({"schema_version": 1, "task": "train", "model": model,
                     "data": {}, "train": {}}, root / "config_resolved.yaml")
        manifest = {"schema_version": 2, "model": dict(model), "files": {"weights": "weights.pt"}}
        save_config(manifest, root / "manifest.yaml")
        torch.save({"scale": torch.tensor(1.0)}, root / "weights.pt")
        return manifest

    def config(self, root):
        return clone_config({"schema_version": 1, "task": "infer", "model": {
            "target": "tests.fakes:DummyVLAModule", "artifact": str(root),
            "state_dim": None, "action_dim": None, "action_horizon": None,
        }, "runtime": {"device": "cpu"}})

    def test_artifact_restores_arbitrary_public_dimensions_before_building_model(self):
        for state_dim, action_dim in ((14, 14), (34, 34), (12, 7)):
            with self.subTest(state=state_dim, action=action_dim), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.artifact(root, state_dim, action_dim)
                config = self.config(root)
                policy = InferencePolicy(config)
                self.assertEqual((policy.spec.state_dim, policy.spec.action_dim), (state_dim, action_dim))
                self.assertEqual(policy.spec.action_horizon, 3)
                self.assertIsNone(config.model.state_dim)  # Caller configuration is not mutated.
                output = policy.predict_chunk({"images": {"base": np.zeros((8, 8, 3), np.uint8)},
                                               "state": np.zeros(state_dim), "prompt": "pick"})
                self.assertEqual(output.actions.shape, (1, 3, action_dim))

    def test_explicit_mismatch_and_inconsistent_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.artifact(root, 34, 34)
            for key in ("state_dim", "action_dim", "action_horizon"):
                with self.subTest(key=key):
                    config = self.config(root)
                    config.model[key] = 7
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        InferencePolicy(config)
                    changed = clone_config(manifest)
                    changed.model[key] = 7
                    save_config(changed, root / "manifest.yaml")
                    with self.assertRaisesRegex(ValueError, "disagrees with its saved configuration"):
                        InferencePolicy(self.config(root))
                    save_config(manifest, root / "manifest.yaml")

    def test_missing_manifest_dimensions_fall_back_to_saved_training_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.artifact(root, 12, 7)
            manifest["model"] = {"target": "tests.fakes:DummyVLAModule"}
            save_config(manifest, root / "manifest.yaml")
            config = self.config(root)
            config.model.pop("state_dim")
            self.assertEqual(InferencePolicy(config).spec.state_dim, 12)
