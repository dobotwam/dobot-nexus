"""Contracts between the ATOM D runner, the shared actuator map and the policy."""

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.real_robot.atom_d import inference as atom_d_inference, policy as atom_d_policy
from examples.real_robot.atom_common.core import RuntimeFault, actuator_targets

ROOT = Path(atom_d_inference.__file__).resolve().parents[3]


class AtomDReleaseTest(unittest.TestCase):
    def test_only_commanded_axes_can_be_saturated(self):
        limits = np.tile([-1.0, 1.0], (17, 1))
        reference = np.zeros(17)
        actions = np.full((3, 18), 2.0)
        with self.assertRaises(RuntimeFault):
            actuator_targets(actions, reference, limits)
        result = actuator_targets(actions, reference, limits, saturate_joints=True)
        np.testing.assert_array_equal(result[:, 1:17], np.ones((3, 16)))
        np.testing.assert_array_equal(result[:, 0], np.zeros(3))
        np.testing.assert_array_equal(actions, np.full((3, 18), 2.0))
        for axis in (0, 15):
            held = reference.copy()
            held[axis] = 2.0
            with self.assertRaises(RuntimeFault):
                actuator_targets(actions, held, limits, saturate_joints=True, head_mode="hold")

    def test_launcher_resolves_sdk_relative_to_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            caller = Path(directory)
            sdk = caller / "sdk-python"
            sdk.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
            sdk.chmod(0o755)
            runner = Path(atom_d_inference.__file__).with_name("run_atom_d.sh")
            result = subprocess.run(
                [str(runner), "--help"],
                cwd=caller,
                env={**os.environ, "ATOM_PYTHON": "./sdk-python"},
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            self.assertIn("--observation-npz", result.stdout)

    def test_legacy_imports_share_the_canonical_runtime(self):
        from examples.atom_robot import atom_d_home, atom_d_inference as legacy, core
        from examples.real_robot.atom_common.core import RuntimeFault
        from examples.real_robot.atom_d import home

        self.assertIs(legacy, atom_d_inference)
        self.assertIs(atom_d_home, home)
        self.assertIs(core.RuntimeFault, RuntimeFault)
        self.assertTrue(home.CONFIG.is_file())

    def test_recorded_run_rejects_motion_flags_before_loading_policy(self):
        for flag in ("--execute", "--full-episode", "--home-start"):
            with self.subTest(flag=flag), redirect_stderr(StringIO()), self.assertRaises(
                SystemExit
            ) as exc:
                atom_d_inference.main(["--observation-npz", "sample.npz", "--site-clear", flag])
            self.assertEqual(exc.exception.code, 2)

    def test_policy_metadata_binds_requested_artifact_and_horizon(self):
        try:
            from websockets.sync import client
        except ImportError:
            self.skipTest("websockets.sync is optional in a CPU-only installation")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            model = Path(directory).resolve()
            meta = {
                "artifact": str(model),
                "input_schema": {"state": [18]},
                "output_schema": {"actions": [50, 18]},
            }
            with patch.object(client, "connect") as connect, patch.dict(
                "os.environ", NEXUS_POLICY_TOKEN="local-test"
            ):
                socket = connect.return_value.__enter__.return_value
                socket.recv.side_effect = lambda **kw: atom_d_policy.msgpack_numpy.packb(meta)
                self.assertIs(atom_d_policy._connect_policy(stack, 8001, model), socket)
                self.assertEqual(
                    connect.call_args.kwargs["additional_headers"]["Authorization"],
                    "Bearer local-test",
                )
                self.assertIsNone(atom_d_policy._connect_policy(stack, 8001, model.parent))
                meta["output_schema"]["actions"] = [20, 18]
                with self.assertRaises(ValueError):
                    atom_d_policy._connect_policy(stack, 8001, model)
