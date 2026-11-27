import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dobot_nexus.core.artifacts import artifact_weight_path
from dobot_nexus.core.config import load_config
from dobot_nexus.core.import_utils import build_model
from dobot_nexus.inference.policy import InferencePolicy
from tests import test_inference_dimensions


class ArtifactSnapshotTest(unittest.TestCase):
    def test_public_load_binds_weights_and_constructor_config_to_validated_snapshot(self):
        fixture = test_inference_dimensions.InferenceDimensionsTest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture.artifact(root, 2, 2)

            def replace_files(path):
                (root / "manifest.yaml").write_text("schema_version: 2\nfiles: {weights: moved.pt}")
                (root / "config_resolved.yaml").write_text("extends: missing.yaml")
                return artifact_weight_path(path)

            def constructor(config, *, mode):
                self.assertEqual(load_config(root / "config_resolved.yaml").model.state_dim, 2)
                self.assertEqual(config.model.checkpoint, str(root / "weights.pt"))
                return build_model(config, mode=mode)

            with (patch("dobot_nexus.inference.policy.artifact_weight_path", replace_files),
                  patch("dobot_nexus.inference.policy.build_model", constructor)):
                self.assertTrue(InferencePolicy(fixture.config(root)).loaded)
