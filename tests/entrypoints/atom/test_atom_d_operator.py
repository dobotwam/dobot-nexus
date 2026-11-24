"""Exercise the real terminal and process transitions with a motion-free worker."""

import json
import os
import pty
import select
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from examples.real_robot.atom_d.console import CANCELLED, OperatorCancelled

ROOT = Path(__file__).resolve().parents[3]


def fake_worker(directory, operation, check_control):
    directory = Path(directory)
    print("started-" + operation, flush=True)
    try:
        while not (directory / ("finish-" + operation)).exists():
            check_control()
            time.sleep(0.01)
        return 0
    except OperatorCancelled:
        return 2 if (directory / "fault").exists() else CANCELLED
    finally:
        print("finished-" + operation, flush=True)


class OperatorTest(unittest.TestCase):
    def launch(self, live_entry=False):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        self.master = master
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        # Attach a real controlling terminal so the Ctrl+C byte generates SIGINT.
        wrapper = (
            "import os, fcntl, termios, sys, signal, subprocess; "
            "os.setsid(); fcntl.ioctl(0, termios.TIOCSCTTY, 0); "
            "saved = termios.tcgetattr(0); signal.signal(signal.SIGINT, lambda *_: None); "
            "code = subprocess.call(sys.argv[1:]); "
            # macOS sets the kernel's pending-input bit on return to canonical mode.
            "restored = termios.tcgetattr(0); restored[3] &= ~getattr(termios, 'PENDIN', 0); "
            "saved[3] &= ~getattr(termios, 'PENDIN', 0); assert saved == restored; sys.exit(code)"
        )
        if live_entry:
            command = [
                "examples/atom_robot/run_atom_d.sh",
                "--execute",
                "--site-clear",
                "--home-start",
                "--full-episode",
            ]
        else:
            command = [
                sys.executable,
                "-u",
                "-c",
                "import sys; from examples.real_robot.atom_d.console import operator_session; "
                "from tests.entrypoints.atom.test_atom_d_operator import fake_worker; "
                "sys.exit(operator_session(sys.argv[1], fake_worker))",
                temp.name,
            ]
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", wrapper, *command],
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env={**os.environ, "ATOM_PYTHON": sys.executable},
        )
        self.addCleanup(self.proc.stdout.close)
        self.addCleanup(self.proc.wait, 5)
        self.addCleanup(self.proc.kill)
        self.buffer = b""
        self.history = b""
        self.wait_for(b"WAITING")

    def wait_for(self, marker, timeout=15):
        deadline = time.monotonic() + timeout
        while marker not in self.buffer:
            ready = select.select([self.proc.stdout], [], [], max(0, deadline - time.monotonic()))[
                0
            ]
            self.assertTrue(ready, (marker, self.history))
            block = os.read(self.proc.stdout.fileno(), 4096)
            self.assertTrue(block, (marker, self.history))
            self.history += block
            self.buffer += block
        self.buffer = self.buffer.split(marker, 1)[1]

    def exit_console(self):
        os.write(self.master, b"\x03")
        self.wait_for(b"EXIT")
        self.assertEqual(self.proc.wait(10), 130, self.proc.stdout.read())

    def test_live_launcher_stays_idle_until_enter(self):
        self.launch(live_entry=True)
        self.exit_console()
        self.assertNotIn(b"EPISODE_FOLDER", self.history)

    def test_stop_home_then_restart(self):
        self.launch()
        os.write(self.master, b"\r")
        self.wait_for(b"started-inference")
        os.write(self.master, b"\rh")
        self.wait_for(b"finished-inference")
        self.wait_for(b"started-home")
        os.write(self.master, b"\rhh")
        (self.folder / "finish-home").touch()
        self.wait_for(b"WAITING")
        os.write(self.master, b"\r")
        self.wait_for(b"started-inference")
        self.exit_console()
        self.assertIn(b"finished-inference", self.history.split(b"started-inference")[-1])
        self.assertEqual(self.history.count(b"started-home"), 1)

    def test_idle_home_can_be_cancelled(self):
        self.launch()
        os.write(self.master, b"h")
        self.wait_for(b"started-home")
        self.exit_console()
        self.assertIn(b"finished-home", self.history)
        self.assertNotIn(b"started-inference", self.history)

    def test_fault_during_stop_requires_a_fresh_command(self):
        self.launch()
        (self.folder / "fault").touch()
        os.write(self.master, b"\r")
        self.wait_for(b"started-inference")
        os.write(self.master, b"h")
        self.wait_for(b"WAITING")
        self.exit_console()
        self.assertNotIn(b"started-home", self.history)

    def test_normal_completion_waits_for_enter(self):
        self.launch()
        (self.folder / "finish-inference").touch()
        os.write(self.master, b"\r")
        self.wait_for(b"finished-inference")
        self.wait_for(b"WAITING")
        self.exit_console()
        self.assertEqual(self.history.count(b"started-inference"), 1)


class TrialRoutingTest(unittest.TestCase):
    def test_manual_home_uses_sdk_home_without_starting_policy(self):
        from examples.real_robot.atom_d import inference as driver

        args = SimpleNamespace(execute=True, site_clear=True, home_start=False)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            driver, "ROOT", Path(tmp)
        ), patch.object(driver, "configure_sdk"), patch.object(
            driver, "atom_d_control_lease"
        ) as lease, patch.object(
            driver.home, "run_home", return_value={"status": "home_start_completed"}
        ) as home, patch.object(driver, "policy_session") as policy:
            self.assertEqual(driver.perform_trial(args, "home", lambda: None), 0)
            lease.assert_called_once()
            self.assertTrue(home.call_args.kwargs["execute"])
            self.assertFalse(home.call_args.kwargs["cancelled"]())
            policy.assert_not_called()
            report = json.loads(next(Path(tmp).rglob("result.json")).read_text())
            self.assertEqual(report["operation"], "home")
            self.assertEqual(report["status"], "home_start_completed")
