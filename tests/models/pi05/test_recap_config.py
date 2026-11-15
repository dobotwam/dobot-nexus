from __future__ import annotations

import subprocess
import sys
import unittest

from dobot_nexus.algorithms.recap.configuration import resolve_inference_config
from dobot_nexus.core.config import clone_config
from dobot_nexus.models.pi05.configuration_recap import resolve_policy_config


class ReCapConfigTest(unittest.TestCase):
    def test_policy_requires_the_current_schema_and_recipe(self):
        for field, value in (("schema_version", 999), ("recipe", "unknown")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_policy_config(
                    clone_config({"model": {"recap": {field: value}}}), mode="infer"
                )

    def test_infer_defaults_strict_guidance_and_nonmutation(self):
        original = clone_config({"task": "infer", "model": {}, "inference": {}})
        result = resolve_policy_config(original, mode="infer")
        self.assertNotIn("recap", original.model)
        self.assertEqual(result.model.recap.unconditional_prob, 0.3)
        self.assertEqual(resolve_inference_config(result).condition, "base")
        for block in ({"condition": "invalid"}, {"guidance_scale": 2}, {"cfg": True}):
            with self.assertRaises(ValueError):
                resolve_inference_config(clone_config({"inference": {"recap": block}}))

    def test_policy_import_does_not_import_transformers_or_rlinf(self):
        code = """import sys
sys.modules['transformers'] = None
from dobot_nexus.models.pi05.modeling_recap import Pi05ReCapModel
assert not any(k == 'rlinf' or k.startswith('rlinf.') for k in sys.modules)
assert not any(k.startswith('transformers.') for k in sys.modules)
"""
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
