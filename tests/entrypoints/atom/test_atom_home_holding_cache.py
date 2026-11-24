import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.real_robot.atom_common.core import RuntimeFault
from examples.real_robot.atom_w_0759.native import NativeBridge, load_last_holding


class HomeHoldingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.validation = Path(self.temp.name)
        self.folder = self.validation / "atom-runtime-20261110-225529-304595"
        self.folder.mkdir()
        self.record = {
            "commands_sent": 100,
            "upper_control_returned": True,
            "command_law": "compensated_position",
            "last_motor_reference": [0.0] * 17,
            "last_tau_ff": [0.0] * 17,
        }

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.folder / "result.json").write_text(json.dumps(self.record))

    def bridge(self):
        bridge = NativeBridge("/sdk", self.validation, execute=True)
        bridge.kp = np.full(17, 300.0)
        bridge.effort = np.full(17, 36.0)
        return bridge

    def test_native_sdk_reference_cannot_be_reused_as_a_physical_home_goal(self):
        self.record.update(command_law="sdk_reference")
        self.record["last_motor_reference"][2] = float(np.deg2rad(2.2576575584))
        self.write()
        self.assertIsNone(load_last_holding(np.zeros(17), self.validation))

    def test_even_close_native_pd_references_are_control_law_incompatible(self):
        self.record.update(command_law="sdk_reference")
        self.record["last_motor_reference"][1] = 0.001
        self.write()
        self.assertIsNone(load_last_holding(np.zeros(17), self.validation))

    def test_repositioned_arms_make_a_cache_miss_not_an_angle_limit_fault(self):
        self.record["last_motor_reference"][4] = float(np.deg2rad(20))
        self.write()
        self.assertIsNone(load_last_holding(np.zeros(17), self.validation))

    def test_repositioned_waist_does_not_reuse_an_old_holding_record(self):
        self.record["last_motor_reference"][0] = float(np.deg2rad(0.1))
        self.write()
        self.assertIsNone(load_last_holding(np.zeros(17), self.validation))

    def test_compatible_same_pose_holding_can_still_be_reused(self):
        self.record["last_motor_reference"][1] = float(np.deg2rad(0.2))
        self.record["last_tau_ff"][1] = 2.0
        self.write()
        held = load_last_holding(np.zeros(17), self.validation)
        self.assertEqual(held["source"], str(self.folder))
        self.assertEqual(held["tau_ff"][1], 2.0)

    def test_unconfirmed_handover_is_still_rejected(self):
        self.record.update(command_law="sdk_reference", upper_control_returned=False)
        self.write()
        with self.assertRaisesRegex(RuntimeFault, "handover"):
            load_last_holding(np.zeros(17), self.validation)

    def test_cross_law_home_seeds_current_reference_and_measured_support_without_writing(self):
        self.record.update(command_law="sdk_reference")
        self.record["last_motor_reference"][2] = float(np.deg2rad(2.26))
        self.write()
        bridge = self.bridge()
        q = np.linspace(-0.2, 0.4, 17)
        tau = np.zeros(17)
        tau[2] = 11.79
        bridge.seed_holding(q, tau)
        np.testing.assert_array_equal(bridge.reference, q)
        self.assertEqual(bridge.tau_ff[2], 11.79)
        self.assertEqual(bridge.command_count, 0)
        self.assertIsNone(bridge.session)

    def test_current_support_still_obeys_sdk_rated_effort(self):
        self.record.update(command_law="sdk_reference")
        self.write()
        tau = np.zeros(17)
        tau[2] = 36.01
        with self.assertRaisesRegex(RuntimeError, "SDK rated effort"):
            self.bridge().seed_holding(np.zeros(17), tau)


if __name__ == "__main__":
    unittest.main()
