"""Integration with the existing model server; no hardware or model weights."""

import unittest

import numpy as np

from dobot_nexus.inference.transport import msgpack_numpy
from dobot_nexus.models.pi05.deploy.server import build_nexus_observation
from examples.real_robot.atom_common.core import actuator_targets, state_vector
from tests.entrypoints.atom import test_atom_inference_pipeline_regressions as observations


class AtomPolicyTransportTest(unittest.TestCase):
    def contract(self):
        return observations.ObservationPipelineTests().contract()

    def test_full_wire_contract_preserves_all_cameras_channels_and_18_fields(self):
        c = self.contract()
        q = np.linspace(-0.8, 0.8, 17)
        state = state_vector(q, [0.2, 0.8])
        images = {
            k: np.tile(np.array([n, n + 1, n + 2], np.uint8), (224, 224, 1))
            for k, n in [("top_camera", 10), ("left_camera", 80), ("right_camera", 160)]
        }
        packet = c.observation(
            {"state": state, "state_names": list(c.layout.state_names), "images": images},
            "Fold the T-shirt with both arms.",
        )
        wire = msgpack_numpy.unpackb(msgpack_numpy.packb(packet))
        actual = build_nexus_observation(wire, state_dim=18)
        np.testing.assert_array_equal(actual["state"], np.asarray(state, np.float32))
        for wire_name, nexus_name in [
            ("top_camera", "base"),
            ("left_camera", "left_wrist"),
            ("right_camera", "right_wrist"),
        ]:
            np.testing.assert_array_equal(actual["images"][nexus_name], images[wire_name])
        actions = np.tile(np.asarray(state, np.float32), (50, 1))
        actions[:, 0] += np.arange(50) / 1000
        mapped = actuator_targets(c.actions(actions, 50), q, np.tile([-3, 3], (17, 1)))
        np.testing.assert_array_equal(mapped[:, 1:8], actions[:, :7])
        np.testing.assert_array_equal(mapped[:, 8:15], actions[:, 8:15])
        np.testing.assert_array_equal(mapped[:, 15:17], actions[:, 16:18])
        np.testing.assert_array_equal(mapped[:, 17:], actions[:, [7, 15]])
