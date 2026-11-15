import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dobot_nexus.cli.infer import _load_observation
from dobot_nexus.inference.server import _observation_from_json


class JsonInferenceTest(unittest.TestCase):
    def test_offline_and_http_accept_the_same_rgb_json(self):
        for batched in (False, True):
            with self.subTest(batched=batched), tempfile.TemporaryDirectory() as root:
                image = np.array([[[0, 128, 255]]], dtype=np.uint8)
                payload = {"images": {"base": (image[None] if batched else image).tolist()},
                           "state": [0.0, 1.0], "prompt": "test", "env_ids": ["robot"]}
                path = Path(root) / "observation.json"
                path.write_text(json.dumps(payload))
                offline, http = _load_observation(path), _observation_from_json(payload)
                np.testing.assert_array_equal(offline.images["base"], image[None])
                np.testing.assert_array_equal(offline.images["base"], http.images["base"])
                self.assertEqual(offline.images["base"].dtype, np.uint8)
                self.assertEqual(offline.env_ids, ["robot"])

    def test_offline_and_http_reject_invalid_pixels_without_clipping(self):
        for pixel in (-1, 256, 0.5, True, "1"):
            with self.subTest(pixel=pixel), tempfile.TemporaryDirectory() as root:
                payload = {"images": {"base": [[[pixel, pixel, pixel]]]},
                           "state": [0.0], "prompt": "test"}
                path = Path(root) / "observation.json"
                path.write_text(json.dumps(payload))
                for load, value in ((_load_observation, path), (_observation_from_json, payload)):
                    with self.assertRaises((ValueError, TypeError)):
                        load(value)
