import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from examples.atom_robot import gripper_reset, home, run
from examples.atom_robot.core import RuntimeFault


class FakeDevice:
    def __init__(self, *, dropped=False, cached=False, home_ok=True, open_ok=True):
        self.hands = {
            s: {"init_state": 1, "grip_state": 3, "opening": 1.0} for s in ("left", "right")
        }
        self.dropped, self.cached = dropped, cached
        self.home_ok, self.open_ok = home_ok, open_ok
        self.calls, self.token, self.writes = [], 0, 0

    def read(self):
        self.token += not self.cached
        return copy.deepcopy(self.hands), self.token

    def home(self, side, timeout):
        self.calls.append(("home", side))
        self.writes += 1
        return self.home_ok

    def open_hand(self, side, timeout):
        self.calls.append(("open", side))
        self.writes += 1
        if not self.dropped:
            self.hands[side]["grip_state"] = 1
        return self.open_ok

    def neutral(self):
        self.calls.append(("neutral",))
        self.writes += 1


def clock():
    state = NS(now=0.0)

    def advance():
        state.now += 0.1
        return state.now

    return advance


class ResetContractTest(unittest.TestCase):
    def test_initialized_dropped_devices_are_rehomed_and_opened(self):
        device = FakeDevice()
        result = gripper_reset.reset_device(device, clock=clock(), sleep=lambda value: None)
        self.assertEqual(
            device.calls, [("home", "left"), ("open", "left"), ("home", "right"), ("open", "right")]
        )
        self.assertEqual(result["before"]["left"]["grip_state"], 3)
        self.assertEqual(result["after"]["left"]["grip_state"], 1)
        self.assertEqual(result["status"], "reset_confirmed")

    def test_persistent_drop_or_cached_success_cannot_pass(self):
        for device in (FakeDevice(dropped=True), FakeDevice(cached=True)):
            with self.subTest(device=device), self.assertRaisesRegex(RuntimeFault, "timed out"):
                gripper_reset.reset_device(
                    device, timeout=1, clock=clock(), sleep=lambda value: None
                )

    def test_failed_home_or_open_aborts_before_other_device(self):
        for device in (FakeDevice(home_ok=False), FakeDevice(open_ok=False)):
            with self.subTest(device=device), self.assertRaises(RuntimeFault):
                gripper_reset.reset_device(device, clock=clock(), sleep=lambda value: None)
            self.assertNotIn(("home", "right"), device.calls)

    def test_invalid_feedback_is_rejected(self):
        normal = {s: {"init_state": 1, "grip_state": 1, "opening": 1.0} for s in ("left", "right")}
        for key, value in [
            ("init_state", 0),
            ("grip_state", 3),
            ("grip_state", 0),
            ("opening", 0.5),
            ("opening", float("nan")),
        ]:
            hands = copy.deepcopy(normal)
            hands["left"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeFault):
                gripper_reset.validate_ready(hands)

    def test_missing_explicit_gates_never_load_hardware(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(gripper_reset, "manufacturer_device") as factory,
        ):
            for flags in ([], ["--execute"], ["--site-clear"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    gripper_reset.main(
                        [
                            "--validation",
                            folder,
                            "--report",
                            folder + "/result.json",
                            "--lock-fd",
                            "0",
                            *flags,
                        ]
                    )
            factory.assert_not_called()

    def test_worker_neutralizes_homing_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            report = path / "result.json"
            with (
                patch.object(
                    gripper_reset, "hardware_lock_path", return_value=path / "model-arm-step.lock"
                ),
                (path / "model-arm-step.lock").open("a") as lock,
            ):
                device = FakeDevice(home_ok=False)
                with (
                    patch.object(gripper_reset, "manufacturer_device", return_value=device),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    code = gripper_reset.main(
                        [
                            "--execute",
                            "--site-clear",
                            "--validation",
                            folder,
                            "--report",
                            str(report),
                            "--lock-fd",
                            str(lock.fileno()),
                        ]
                    )
            self.assertEqual(code, 2)
            self.assertEqual(device.calls[-1], ("neutral",))
            self.assertEqual(json.loads(report.read_text())["status"], "reset_failed")

    def test_parent_cancel_stops_child_and_keeps_report_failure_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            child = Mock()
            child.poll.return_value = None
            with (
                patch.object(gripper_reset.subprocess, "Popen", return_value=child) as launch,
                self.assertRaisesRegex(RuntimeFault, "interrupted"),
            ):
                gripper_reset.reset_grippers(folder, folder, 42, cancelled=lambda: True)
            child.terminate.assert_called_once()
            child.wait.assert_called_once()
            self.assertEqual(launch.call_args.kwargs["pass_fds"], (42,))
            self.assertEqual(
                launch.call_args.kwargs["cwd"], Path(gripper_reset.__file__).resolve().parents[2]
            )

    def test_child_failure_or_timeout_never_reports_reset_success(self):
        with tempfile.TemporaryDirectory() as folder:
            child = Mock(returncode=2)
            child.poll.return_value = 2
            with (
                patch.object(gripper_reset.subprocess, "Popen", return_value=child),
                self.assertRaisesRegex(RuntimeFault, "Device reset failed"),
            ):
                gripper_reset.reset_grippers(folder, folder, 42)

    def test_foreign_hardware_lease_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "model-arm-step.lock").touch()
            with (
                patch.object(
                    gripper_reset, "hardware_lock_path", return_value=path / "model-arm-step.lock"
                ),
                (path / "other.lock").open("a") as lock,
                patch.object(gripper_reset, "manufacturer_device") as factory,
                self.assertRaises(RuntimeFault),
            ):
                gripper_reset.main(
                    [
                        "--execute",
                        "--site-clear",
                        "--validation",
                        folder,
                        "--report",
                        folder + "/result.json",
                        "--lock-fd",
                        str(lock.fileno()),
                    ]
                )
            factory.assert_not_called()


class HomeResetIntegrationTest(unittest.TestCase):
    def exercise(self, *, reset=True, failure=False):
        calls = []
        healthy = {s: {"init_state": 1, "grip_state": 1, "opening": 1.0} for s in ("left", "right")}
        bridge = NS(reference=np.zeros(17), limits=np.tile([-2.0, 2.0], (17, 1)), session=None)
        bridge.open = lambda: calls.append("open")
        bridge.observe = lambda: (
            calls.append("observe") or {"q": np.zeros(17), "openings": np.ones(2)}
        )
        bridge.acquire = lambda: calls.append("acquire")
        bridge.send = lambda value: calls.append("send")
        bridge.snapshot = lambda: {"q": np.zeros(17)}
        bridge.close = lambda: calls.append("close")
        bridge.result = lambda: {"commands_sent": calls.count("send")}

        def resetter(*a, **k):
            calls.append("reset")
            if failure:
                raise RuntimeFault("left drop persists")
            return {"status": "reset_confirmed", "writes": 7, "after": healthy}

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(home, "hardware_lock_path", return_value=Path(folder) / "hardware.lock"),
            patch.object(home, "NativeBridge", return_value=bridge) as factory,
            patch.object(home, "home_targets", return_value=(np.zeros(17), np.zeros(17))),
            patch.object(home.time, "monotonic", side_effect=clock()),
            patch.object(home.time, "sleep"),
            patch.object(gripper_reset, "reset_grippers", side_effect=resetter) as device_reset,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = home.return_home("/sdk", folder, fresh_hold=True, reset_devices=reset)
        return calls, result, device_reset, factory

    def test_reset_and_fresh_feedback_precede_arm_control(self):
        calls, result, reset, factory = self.exercise()
        self.assertEqual(result["status"], "physical_home_completed")
        self.assertEqual(result["gripper_commands_sent"], 7)
        self.assertLess(calls.index("reset"), calls.index("acquire"))
        self.assertEqual(calls.count("observe"), 2)
        self.assertTrue(factory.call_args.kwargs["fresh_hold"])

    def test_failed_reset_never_acquires_or_sends_arm_commands(self):
        calls, result, reset, factory = self.exercise(failure=True)
        self.assertEqual(result["status"], "home_fault")
        self.assertNotIn("acquire", calls)
        self.assertEqual(result["commands_sent"], 0)

    def test_normal_return_home_holds_grippers(self):
        calls, result, reset, factory = self.exercise(reset=False)
        reset.assert_not_called()
        self.assertEqual(result["gripper_commands_sent"], 0)

    def test_direct_runtime_home_failure_prevents_workers(self):
        args = run.parse_args(
            ["--execute", "--site-clear", "--sdk-root", "/sdk", "--gripper-sdk-root", "/dh"]
        )
        with (
            patch.object(run, "preflight_policy", return_value=({}, 50)),
            patch("examples.atom_robot.validation.require_live_validation"),
            patch.object(home, "return_home", return_value={"status": "home_fault"}) as returning,
            patch.object(run.mp, "get_context") as context,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeFault):
                run.run(args)
        self.assertTrue(returning.call_args.kwargs["reset_devices"])
        self.assertTrue(returning.call_args.kwargs["fresh_hold"])
        context.assert_not_called()


if __name__ == "__main__":
    unittest.main()
