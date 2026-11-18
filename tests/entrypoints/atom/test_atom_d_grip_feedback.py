"""Exercise inference feedback without constructing a robot driver or command publisher."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from examples.atom_robot import atom_d_inference as inference


def hand(status, opening):
    return SimpleNamespace(
        grip_state=SimpleNamespace(value=status),
        init_result_state=SimpleNamespace(value=1),
        realtime_position=opening,
    )


class GripFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.left = hand(1, 1.0)
        self.right = hand(3, 0.0)
        self.fsm = 2
        self.now = 1.0
        self.feedback = Mock()
        hands = SimpleNamespace(left=self.left, right=self.right)
        motor = SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0)
        self.feedback.messages = {
            "rt/hands/state": hands,
            "rt/upper/state": SimpleNamespace(motor_state=[motor] * 17),
        }
        self.feedback.arrivals = {}

        def sample(cancelled):
            self.feedback.arrivals["rt/hands/state"] = self.now
            return None, self.fsm, hands

        self.feedback.sample.side_effect = sample
        self.robot = Mock()
        self.robot.has_error.return_value = False

    def wait(self, sides, opening=False):
        def sleep(seconds):
            self.now += seconds

        with tempfile.TemporaryDirectory() as folder, patch.object(
            inference.time, "sleep", sleep
        ), patch.object(inference.time, "monotonic", lambda: self.now):
            return inference.wait_for_hand_state(
                self.feedback,
                self.robot,
                Mock(),
                Path(folder),
                sides,
                self.now + 5.0,
                opening=opening,
            )

    def test_drop_is_telemetry_but_bad_mode_initialization_and_sdk_fault_still_stop(self):
        self.assertEqual(inference.measure(self.feedback)[0][15], 0.0)
        self.fsm = 0
        with self.assertRaisesRegex(RuntimeError, "FSM=0"):
            inference.measure(self.feedback)
        self.fsm = 2
        self.left.init_result_state.value = 0
        with self.assertRaisesRegex(RuntimeError, "initialization"):
            inference.measure(self.feedback)
        self.left.init_result_state.value = 1
        self.robot.has_error.return_value = True
        with self.assertRaisesRegex(RuntimeError, "SDK drive or bus fault"):
            self.wait([1])

    def test_drop_in_a_two_hand_check_is_attributed_to_the_reporting_side(self):
        self.left.grip_state.value = 2
        self.left.realtime_position = 0.05
        result = self.wait([0, 1])
        self.assertEqual(result["states"], [2, 3])
        self.assertEqual(result["dropped_sides"], [1])
        self.assertGreaterEqual(result["wait_s"], 0.4)

    def test_drop_during_release_still_requires_measured_open_position(self):
        with self.assertRaisesRegex(RuntimeError, "Unconfirmed opening"):
            self.wait([1], opening=True)
        self.right.realtime_position = 1.0
        self.assertEqual(self.wait([1], opening=True)["dropped_sides"], [1])

    def test_empty_close_remains_distinct_from_a_reported_drop(self):
        self.right.grip_state.value = 1
        with self.assertRaisesRegex(RuntimeError, "Unconfirmed grasp"):
            self.wait([1])
        self.right.grip_state.value = 2
        self.assertEqual(self.wait([1])["dropped_sides"], [])
