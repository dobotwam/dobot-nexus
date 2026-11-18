import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from examples.atom_robot.atom_d_home import prepare_home_grippers


def hand(init=1, grip=1, opening=1.0):
    init_names = ["NOT_DONE", "SUCCESS", "EXECUTING"]
    grip_names = ["MOVING", "IN_PLACE", "HOLDING_OBJECT", "DROPPED"]
    return SimpleNamespace(
        init_result_state=SimpleNamespace(value=init, name=init_names[init]),
        grip_state=SimpleNamespace(value=grip, name=grip_names[grip]),
        realtime_position=opening,
    )


class GripperStartupTest(unittest.TestCase):
    def setUp(self):
        self.now = 1.0
        self.robot = Mock()
        self.robot.has_error.return_value = False
        self.feedback = Mock()
        self.feedback.arrivals = {"rt/hands/state": self.now}

    def run_setup(self, status, *, fresh=True, cancelled=lambda: False):
        def sample(check_cancel):
            if check_cancel():
                raise InterruptedError()
            fsm, left, right = status()
            if fresh:
                self.feedback.arrivals["rt/hands/state"] = self.now
            return None, fsm, SimpleNamespace(left=left, right=right)

        def sleep(seconds):
            self.now += seconds

        self.feedback.sample.side_effect = sample
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_home_grippers(
                self.robot, self.feedback, cancelled, clock=lambda: self.now, sleep=sleep
            )

    def test_full_open_but_holding_requires_explicit_position_command(self):
        result = self.run_setup(
            lambda: (2, hand(grip=1 if self.robot.robot.open_gripper.called else 2), hand())
        )
        self.assertEqual(result["before"]["left"]["grip"], "HOLDING_OBJECT")
        self.assertEqual(result["after"]["left"]["grip"], "IN_PLACE")
        self.assertGreaterEqual(result["opening_seconds"], 0.3)
        self.robot.robot.open_gripper.assert_called_once_with()
        self.robot.home_grippers.assert_not_called()
        self.robot.set_fsm_state.assert_not_called()

    def test_mode_then_homing_then_retry_lost_open(self):
        def status():
            return (
                2 if self.robot.set_fsm_state.called else 0,
                hand(
                    init=int(self.robot.home_grippers.called),
                    opening=1.0 if self.robot.robot.open_gripper.call_count >= 2 else 0.4,
                ),
                hand(),
            )

        result = self.run_setup(status)
        self.assertEqual(result["homing_commands"], {"left": 1, "right": 0})
        self.assertEqual(result["open_commands"], 2)
        self.robot.home_grippers.assert_called_once_with(["left"])
        self.robot.set_fsm_state.assert_called_once_with()

    def test_executing_homing_is_not_restarted(self):
        result = self.run_setup(lambda: (2, hand(init=2 if self.now < 4.0 else 1), hand()))
        self.assertEqual(result["homing_commands"], {"left": 0, "right": 0})

    def test_executing_left_hand_blocks_other_homing_and_open(self):
        with self.assertRaisesRegex(RuntimeError, "gripper homing timed out.*EXECUTING"):
            self.run_setup(lambda: (2, hand(init=2, opening=1.006), hand(init=0)))
        self.robot.home_grippers.assert_not_called()
        self.robot.robot.open_gripper.assert_not_called()

    def test_cold_start_retries_only_the_unacknowledged_hand(self):
        def status():
            sends = self.robot.home_grippers.call_count
            return 2, hand(init=int(sends >= 1)), hand(init=int(sends >= 2))

        result = self.run_setup(status)
        self.assertEqual(result["homing_commands"], {"left": 1, "right": 2})
        self.assertEqual(
            [c.args[0] for c in self.robot.home_grippers.call_args_list],
            [["left", "right"], ["right"]],
        )

    def test_cold_start_with_no_ack_times_out(self):
        with self.assertRaisesRegex(RuntimeError, "gripper homing timed out.*NOT_DONE"):
            self.run_setup(lambda: (2, hand(), hand(init=0)))
        self.assertEqual(self.robot.home_grippers.call_count, 2)
        self.robot.robot.open_gripper.assert_not_called()

    def test_contradictory_right_status_is_reported_and_left_is_identified(self):
        with self.assertRaisesRegex(
            RuntimeError, 'gripper opening timed out.*"left".*"right".*HOLDING_OBJECT'
        ):
            self.run_setup(lambda: (2, hand(), hand(grip=2)))
        self.assertEqual(self.robot.robot.open_gripper.call_count, 3)
        self.robot.command_joint_state.assert_not_called()

    def test_stale_endpoint_cannot_pass(self):
        with self.assertRaisesRegex(RuntimeError, "gripper opening timed out"):
            self.run_setup(lambda: (2, hand(), hand()), fresh=False)

    def test_no_hand_command_if_mode_or_safety_check_fails(self):
        with self.assertRaisesRegex(RuntimeError, "FSM transition timed out"):
            self.run_setup(lambda: (3, hand(), hand()))
        self.robot.robot.open_gripper.assert_not_called()
        self.robot.has_error.return_value = True
        with self.assertRaisesRegex(RuntimeError, "SDK fault"):
            self.run_setup(lambda: (2, hand(), hand()))
        self.robot.robot.open_gripper.assert_not_called()
        with self.assertRaises(InterruptedError):
            self.run_setup(lambda: (2, hand(), hand()), cancelled=lambda: True)
        self.robot.robot.open_gripper.assert_not_called()
