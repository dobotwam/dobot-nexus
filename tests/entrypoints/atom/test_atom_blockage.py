import io
import unittest

import numpy as np

from examples.real_robot.atom_common.core import BlockedDirectionGuard
from examples.real_robot.atom_w_0759.kinematics import BlockageMonitor, JointBlockageMonitor, Kinematics


class BlockageTest(unittest.TestCase):
    def test_uncalibrated_joint_stall_requires_next_horizon_before_stopping(self):
        monitor = JointBlockageMonitor()
        moving = JointBlockageMonitor()
        evidence = None
        actual = np.zeros(7)
        for now in np.linspace(0, 0.5, 11):
            reference = np.zeros(7)
            reference[3] = np.deg2rad(now * 3)
            evidence = monitor.observe(float(now), "left", actual, reference, 0.2 + now * 0.2)
            self.assertIsNone(moving.observe(float(now), "left", reference, reference, 0.2))
        self.assertIsNotNone(evidence)
        gate = BlockedDirectionGuard(minimum_push_m=np.deg2rad(0.3), dimension=7)
        gate.record_blockage(4, evidence["feedback_q_rad"], evidence["reference_q_rad"])
        continued = np.tile(reference, (20, 1))
        self.assertTrue(gate.next_horizon_pushes_into_obstruction(5, continued))
        self.assertFalse(gate.next_horizon_pushes_into_obstruction(5, -continued))

    def test_stall_is_confirmed_but_normal_motion_is_not(self):
        stalled = BlockageMonitor()
        moving = BlockageMonitor()
        evidence = None
        for now in np.linspace(0, 0.5, 11):
            target = np.array([0.0, 0.0, 0.7 - now * 0.03])
            evidence = stalled.observe(
                float(now), "left", [0, 0, 0.7], target, 1.0, 0.1 + now * 0.2
            )
            self.assertIsNone(moving.observe(float(now), "left", target, target, 0.05, 0.1))
        self.assertIsNotNone(evidence)
        self.assertLessEqual(evidence["feedback_progress_m"], 0.001)

    def test_no_contact_is_invented_during_a_stationary_hold(self):
        monitor = BlockageMonitor()
        for now in np.linspace(0, 1, 20):
            self.assertIsNone(
                monitor.observe(float(now), "right", [0, 0, 0.7], [0, 0, 0.7], 1.0, 0.2)
            )

    def test_fk_uses_rotation_origin_and_tool_offset(self):
        text = '<robot name="test">'
        for side in ["left", "right"]:
            text += (
                f'<joint name="{side}_shoulder_pitch_joint" type="revolute">'
                f'<parent link="torso_link"/><child link="{side}_wrist_yaw_link"/>'
                '<origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>'
            )
        text += "</robot>"
        names = (
            ["torso_joint", "left_shoulder_pitch_joint"]
            + [f"x{i}" for i in range(6)]
            + ["right_shoulder_pitch_joint"]
            + [f"y{i}" for i in range(8)]
        )
        fk = Kinematics(
            io.StringIO(text), names, tool_offsets={"left": [1, 0, 0], "right": [1, 0, 0]}
        )
        q = np.zeros(17)
        q[1] = np.pi / 2
        np.testing.assert_allclose(fk.position(q, "left"), [1, 1, 0], atol=1e-12)
        np.testing.assert_allclose(fk.position(q, "right"), [2, 0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
