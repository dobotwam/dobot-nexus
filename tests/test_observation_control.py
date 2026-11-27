import unittest

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch
from dobot_nexus.inference.server import _observation_from_json


class ObservationControlTest(unittest.TestCase):
    def payload(self):
        return dict(images={"base": np.zeros((2, 4, 4, 3), dtype=np.uint8)},
                    state=np.zeros((2, 3), dtype=np.float32), prompt=["pick", "pick"])

    def test_optional_control_and_json_transport(self):
        payload = self.payload()
        self.assertEqual(ObservationBatch.from_value(payload).control, {})
        payload["images"]["base"] = payload["images"]["base"].tolist()
        payload["control"] = {"phase": [False, True]}
        result = _observation_from_json(payload)
        np.testing.assert_array_equal(result.control["phase"], [False, True])
        self.assertEqual(result.control["phase"].dtype, np.bool_)

    def test_control_requires_explicit_boolean_per_row_and_copies_input(self):
        for value in (True, [True], [[True], [False]], [0, 1], ["false", "true"]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "bool \\[B\\]"):
                ObservationBatch.from_value({**self.payload(), "control": {"phase": value}})
        flags = np.array([False, True])
        result = ObservationBatch.from_value({**self.payload(), "control": {"phase": flags}})
        flags[:] = False
        self.assertTrue(result.control["phase"][1])
