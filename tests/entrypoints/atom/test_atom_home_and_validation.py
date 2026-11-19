import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.atom_robot.core import RuntimeFault, Trajectory
from examples.atom_robot.home import HOME_CONFIG, home_targets, tracking_playback_scale
from examples.atom_robot.validation import (
    require_live_validation,
    source_fingerprint,
    validate_read_only_report,
)


class HomeContractTest(unittest.TestCase):
    def test_tracking_clock_waits_before_the_unchanged_joint_error_guard(self):
        actual = np.zeros(17)
        reference = np.zeros(17)
        self.assertEqual(tracking_playback_scale(actual, reference), 1.0)
        reference[4] = np.deg2rad(0.85)
        self.assertAlmostEqual(tracking_playback_scale(actual, reference), 0.5)
        reference[4] = np.deg2rad(1.2)
        self.assertAlmostEqual(tracking_playback_scale(actual, reference), 0.0)
        reference[4] = np.deg2rad(1.9)
        self.assertEqual(tracking_playback_scale(actual, reference), 0.0)
        actual[:] = reference
        self.assertEqual(tracking_playback_scale(actual, reference), 1.0)

    def test_head_reference_bias_is_not_added_to_the_physical_goal_or_accumulated(self):
        config = json.loads(HOME_CONFIG.read_text())
        q = np.zeros(17)
        physical, motor = home_targets(q, config)
        self.assertAlmostEqual(motor[16] - physical[16], np.deg2rad(-0.6))
        again_physical, again_motor = home_targets(physical, config)
        np.testing.assert_allclose(again_physical, physical)
        np.testing.assert_allclose(again_motor, motor)

    def test_home_motion_envelope_cannot_be_relaxed_by_configuration(self):
        config = json.loads(HOME_CONFIG.read_text())
        for key, value in [
            ("physical_tolerance_deg", 2),
            ("reference_speed_deg_s", 6.0),
            ("head_pitch_motor_reference_bias_deg", -2),
        ]:
            other = dict(config)
            other[key] = value
            with self.assertRaises(RuntimeFault):
                home_targets(np.zeros(17), other)

    def test_home_uses_normal_reference_speed_without_the_old_point375_cap(self):
        config = json.loads(HOME_CONFIG.read_text())
        state = np.asarray(config["physical_state"])
        q = np.r_[0.0, state[:7], state[8:15], state[16:]]
        q[16] -= np.deg2rad(20)
        _, target = home_targets(q, config)
        path = Trajectory(
            np.r_[q, 1.0, 1.0],
            np.r_[target, 1.0, 1.0][None, :],
            joint_speed=np.deg2rad(config["reference_speed_deg_s"]),
        )
        self.assertLess(path.duration, 10.0)
        self.assertGreater(path.duration, 3.0)


class ValidationContractTest(unittest.TestCase):
    def test_live_gate_rejects_receipt_for_a_different_negotiated_horizon(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = {
                "status": "passed",
                "sources": source_fingerprint(),
                "repeat_runs": 2,
                "unit_tests_passed": True,
                "inference_horizon": 50,
            }
            (Path(folder) / "atom-runtime-validation.json").write_text(json.dumps(receipt))
            self.assertEqual(
                require_live_validation(folder, expected_horizon=50)["inference_horizon"], 50
            )
            with self.assertRaises(RuntimeFault):
                require_live_validation(folder, expected_horizon=20)

    def valid(self):
        return {
            "status": "completed",
            "sources": source_fingerprint(),
            "execute": False,
            "runtime_identity": {"head_mode": "hold", "camera": "test"},
            "horizon": 20,
            "commands_sent": 0,
            "gripper_commands_sent": 0,
            "motor_writer_created": False,
            "dds_session_initializations": 1,
            "completed_horizons": 3,
            "consumed_actions": 60,
            "command_law": "sdk_reference",
            "action_fps": 20.0,
            "dispatched_action_rows": 60,
            "gripper_action_rows_consumed": 60,
            "maximum_row_dispatch_lateness_s": 0.005,
            "maximum_gripper_row_lateness_s": 0.01,
            "maximum_scheduler_gap_s": 0.01,
            "worker_exit_codes": [0, 0],
            "maximum_shadow_reference_step_deg": 0.01,
            "server_metadata": {
                "policy_spec": {"action_horizon": 20},
                "output_schema": {"actions": [20, 18]},
            },
            "worker_events": [
                {
                    "kind": "closed",
                    "component": "perception",
                    "connection_count": 1,
                    "camera_stats": {
                        name: {"starts": 1, "stops": 1, "frames": 100}
                        for name in ["top_camera", "left_camera", "right_camera"]
                    },
                },
                {"kind": "closed", "component": "grippers", "writes": 0, "writer_creations": 0},
            ],
        }

    def test_only_complete_zero_writer_reused_lifecycles_pass(self):
        report = self.valid()
        self.assertEqual(validate_read_only_report(report)["actions"], 60)
        for field, value in [
            ("commands_sent", 1),
            ("motor_writer_created", True),
            ("dds_session_initializations", 3),
            ("consumed_actions", 3),
            ("maximum_scheduler_gap_s", 0.2),
            ("sources", {}),
        ]:
            other = copy.deepcopy(report)
            other[field] = value
            with self.assertRaises(RuntimeFault):
                validate_read_only_report(other)
        report["worker_events"][0]["camera_stats"]["left_camera"]["starts"] = 2
        with self.assertRaises(RuntimeFault):
            validate_read_only_report(report)

    def test_50_step_receipt_requires_all_150_actions_in_three_horizons(self):
        report = self.valid()
        report["horizon"] = 50
        report["consumed_actions"] = 150
        report["dispatched_action_rows"] = 150
        report["gripper_action_rows_consumed"] = 150
        report["server_metadata"]["policy_spec"]["action_horizon"] = 50
        report["server_metadata"]["output_schema"]["actions"] = [50, 18]
        self.assertEqual(validate_read_only_report(report)["actions"], 150)
        report["consumed_actions"] = 60
        with self.assertRaises(RuntimeFault):
            validate_read_only_report(report)
        report["consumed_actions"] = 150
        report["server_metadata"]["output_schema"]["actions"] = [20, 18]
        with self.assertRaises(RuntimeFault):
            validate_read_only_report(report)


if __name__ == "__main__":
    unittest.main()
