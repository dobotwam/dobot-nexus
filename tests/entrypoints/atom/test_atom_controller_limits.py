import copy
import types
import unittest
from unittest.mock import patch

import numpy as np

from examples.real_robot.atom_common.controller_limits import (
    apply_controller_limits,
    assert_controller_limits_unchanged,
    controller_bounds,
    read_controller_settings,
)
from examples.real_robot.atom_common.core import STATE_NAMES, RuntimeFault, actuator_targets
from examples.real_robot.atom_w_0759.home import home_targets, tracking_playback_scale
from examples.real_robot.atom_w_0759.native import NativeBridge
from examples.real_robot.atom_common.torque import OuterTorqueCompensator


def fixture():
    positions = {
        "waist": [-2.9653, 2.9653],
        "left_arm": [
            [-2.9706, 2.9706],
            [-0.4294, 3.1398],
            [-2.9706, 2.9706],
            [-0.8692, 3.1398],
            [-2.9706, 2.9706],
            [-1.5708, 1.5708],
            [-1.5708, 1.5708],
        ],
        "right_arm": [
            [-2.9706, 2.9706],
            [-3.1398, 0.4294],
            [-2.9706, 2.9706],
            [-0.8692, 3.1398],
            [-2.9706, 2.9706],
            [-1.5708, 1.5708],
            [-1.5708, 1.5708],
        ],
        "head": [[-1.5708, 1.5708], [-0.8692, 1.5708]],
    }
    snapshot = {
        "version": {"controller": "1.2.0.0-beta8-f4993b2b"},
        "posLimit": {"jointsPosLimit": positions},
        "jointOffset": {
            "offset": {
                "waist": [0.0],
                "left_arm": [0.0] * 7,
                "right_arm": [0.0] * 7,
                "head": [0.0] * 2,
            }
        },
    }
    names = ("torso_joint",) + STATE_NAMES[:7] + STATE_NAMES[8:15] + STATE_NAMES[16:]
    settings = {
        "joints": [
            {
                "index": i,
                "name": n.replace("wrist_yaw_joint", "wrist_roll_joint"),
                "limits": [-0.5, 0.5],
                "kp": 200,
                "kd": 8,
            }
            for i, n in enumerate(names)
        ]
    }
    return settings, snapshot


class ControllerLimitsTest(unittest.TestCase):
    def test_recorded_motor_reference_does_not_fake_measured_feedback_or_change_wire_target(self):
        def fake_bridge(enabled):
            b = NativeBridge("/sdk", "/state", execute=True, dataset_reference_test=enabled)
            b.limits = np.tile([-3.0, 3.0], (17, 1))
            b.kp = np.full(17, 200.0)
            b.kd = np.full(17, 8.0)
            b.effort = np.full(17, 36.0)
            b.reference = np.zeros(17)
            b.comp = types.SimpleNamespace(step=lambda ref, q, dt: np.zeros(17))
            b.session = types.SimpleNamespace(
                pump=lambda: None,
                check=lambda owner: None,
                q=lambda: np.zeros(17),
                matches={"upper": 1},
                base=np.zeros(17),
                last_ref=None,
                last={
                    "upper": types.SimpleNamespace(motor_state=[types.SimpleNamespace(dq=0.0)] * 17)
                },
                dds=types.SimpleNamespace(
                    MotorCmd_=lambda **kw: types.SimpleNamespace(**kw), UpperCmd_=lambda **kw: kw
                ),
                writer=types.SimpleNamespace(write=lambda value: sent.append(value)),
            )
            return b

        reference = np.zeros(17)
        reference[4] = np.deg2rad(-3.0)
        sent = []
        ordinary = fake_bridge(False)
        with self.assertRaises(RuntimeFault):
            ordinary.send(reference)
        with self.assertRaises(RuntimeFault):
            ordinary.send(reference, expected_feedback=np.zeros(17))
        diagnostic = fake_bridge(True)
        diagnostic.send(reference, expected_feedback=np.zeros(17))
        self.assertAlmostEqual(sent[-1]["motor_cmd"][4].q, reference[4])
        bad = np.zeros(17)
        bad[4] = np.deg2rad(3.0)
        with self.assertRaises(RuntimeFault):
            diagnostic.send(reference, expected_feedback=bad)

    def test_head_clock_uses_physical_reference_without_accumulating_motor_bias(self):
        reference = np.zeros(17)
        feedback = reference.copy()
        feedback[16] = np.deg2rad(1.19)
        original = reference.copy()
        self.assertLess(tracking_playback_scale(feedback, reference), 0.01)
        self.assertGreater(
            tracking_playback_scale(feedback, reference, head_pitch_bias_deg=-0.6), 0.9
        )
        np.testing.assert_array_equal(reference, original)

    def test_live_limits_replace_custom_ranges_without_mutating_sdk_or_gains(self):
        settings, snapshot = fixture()
        original = copy.deepcopy(settings)
        result = apply_controller_limits(settings, snapshot, pc1_ip="192.168.8.234")
        self.assertEqual(settings, original)
        bounds = controller_bounds(snapshot)
        np.testing.assert_array_equal([j["limits"] for j in result["joints"]], bounds)
        self.assertEqual(result["joints"][4]["limits"], [-0.8692, 3.1398])
        self.assertEqual(result["joints"][16]["limits"], [-0.8692, 1.5708])
        for before, after in zip(settings["joints"], result["joints"], strict=True):
            for key in ("name", "index", "kp", "kd"):
                self.assertEqual(before[key], after[key])

    def test_current_elbow_and_head_feedback_are_valid_in_controller_ranges(self):
        _, snapshot = fixture()
        q = np.zeros(17)
        q[4], q[16] = np.deg2rad([86.9322, 43.0388])
        actions = np.zeros((1, 18))
        actions[0, 3] = q[4]
        actions[0, 17] = q[16]
        points = actuator_targets(actions, q, controller_bounds(snapshot))
        self.assertAlmostEqual(points[0, 4], q[4])
        self.assertAlmostEqual(points[0, 16], q[16])

    def test_hardware_boundary_is_enforced_for_arm_and_head_action_targets(self):
        _, snapshot = fixture()
        bounds = controller_bounds(snapshot)
        q, action = np.zeros(17), np.zeros((1, 18))
        action[0, 3] = bounds[4, 1]
        actuator_targets(action, q, bounds)
        action[0, 3] += 0.0001
        with self.assertRaises(RuntimeFault):
            actuator_targets(action, q, bounds)
        action[0, 3] = 0
        action[0, 17] = bounds[16, 1] + 0.0001
        with self.assertRaises(RuntimeFault):
            actuator_targets(action, q, bounds)

    def test_malformed_controller_ranges_and_calibration_are_rejected(self):
        _, original = fixture()
        for kind in ("length", "nan", "reversed", "offset"):
            value = copy.deepcopy(original)
            if kind == "length":
                value["posLimit"]["jointsPosLimit"]["left_arm"].pop()
            if kind == "nan":
                value["posLimit"]["jointsPosLimit"]["head"][0][0] = float("nan")
            if kind == "reversed":
                value["posLimit"]["jointsPosLimit"]["waist"] = [1, -1]
            if kind == "offset":
                value["jointOffset"]["offset"]["head"].pop()
            with self.subTest(kind=kind), self.assertRaises(RuntimeFault):
                controller_bounds(value)

    def test_different_joint_order_is_not_silently_mapped(self):
        settings, snapshot = fixture()
        settings["joints"][1]["name"] = "right_shoulder_pitch_joint"
        with self.assertRaises(RuntimeFault):
            apply_controller_limits(settings, snapshot, pc1_ip="192.168.8.234")

    def test_settings_failure_has_no_fallback_to_old_or_unbounded_limits(self):
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError("offline")
            with self.assertRaises(RuntimeFault):
                read_controller_settings("192.168.8.234")

    def test_limit_or_calibration_change_is_detected_before_writer_creation(self):
        settings, snapshot = fixture()
        bridge = NativeBridge("/sdk", "/state", execute=True)
        bridge.settings = apply_controller_limits(settings, snapshot, pc1_ip="192.168.8.234")
        for group in ("limit", "calibration", "version"):
            changed = copy.deepcopy(snapshot)
            if group == "limit":
                changed["posLimit"]["jointsPosLimit"]["waist"][1] -= 0.1
            if group == "calibration":
                changed["jointOffset"]["offset"]["head"][0] += 0.01
            if group == "version":
                changed["version"]["controller"] = "different"
            with (
                self.subTest(group=group),
                patch(
                    "examples.real_robot.atom_common.controller_limits.read_controller_settings",
                    return_value=changed,
                ),
                self.assertRaises(RuntimeFault),
            ):
                bridge.acquire()
            self.assertEqual(bridge.command_count, 0)
            self.assertIsNone(bridge.session)
        with patch(
            "examples.real_robot.atom_common.controller_limits.read_controller_settings", return_value=snapshot
        ):
            assert_controller_limits_unchanged(bridge.settings, pc1_ip="192.168.8.234")

    def test_home_has_no_extra_ninety_degree_displacement_box(self):
        config = {
            "schema_version": 1,
            "robot_type": "ATOM W",
            "robot_id": "0759",
            "joint_units": "rad",
            "physical_state": [0.0] * 18,
            "waist": "hold",
            "grippers": "hold",
            "physical_tolerance_deg": 0.8,
            "reference_speed_deg_s": 3.0,
            "head_pitch_motor_reference_bias_deg": 0.0,
        }
        q = np.zeros(17)
        q[1] = -2.7
        physical, motor = home_targets(q, config)
        self.assertEqual(motor[1], 0.0)
        self.assertEqual(physical[0], q[0])

    def test_static_support_above_old_quarter_effort_is_preserved(self):
        comp = OuterTorqueCompensator(np.full(17, 200.0), np.full(17, 36.0), max_bias_deg=3.0)
        tau = np.zeros(17)
        tau[4], tau[11] = -9.45, -9.16
        comp.seed_static(tau)
        np.testing.assert_allclose(comp.current_tau(), tau)
        np.testing.assert_allclose(comp.step(np.zeros(17), np.zeros(17), 0.01), tau)
        # Static support is independent of the small additional PI correction.
        tau[4] = -20.0
        comp.seed_static(tau)
        np.testing.assert_allclose(comp.current_tau(), tau)
        tau[4] = -36.001
        with self.assertRaises(RuntimeError):
            comp.seed_static(tau)

    def test_latched_support_has_no_handover_jump_and_total_stays_rated(self):
        comp = OuterTorqueCompensator(np.full(17, 200.0), np.full(17, 18.6), max_bias_deg=3.0)
        q = np.zeros(17)
        ref = q.copy()
        ref[4] = np.deg2rad(1.5)
        tau = np.zeros(17)
        tau[4] = 17.0
        comp.seed_latched(ref, q, tau)
        np.testing.assert_allclose(comp.step(ref, q, 0.0), tau, atol=1e-12)
        for _ in range(2000):
            out = comp.step(ref, q, 0.01)
            self.assertLessEqual(abs(out[4]), 18.6)
            np.testing.assert_array_equal(out[[0, 15, 16]], 0.0)


if __name__ == "__main__":
    unittest.main()
