"""CPU checks for example configuration, execution gates and package isolation."""

import contextlib
import copy
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.real_robot.atom_w_0759 import home, run, validate, validation, verify
from examples.real_robot.atom_common.core import RuntimeFault, actuator_targets
from examples.real_robot.atom_common.policy import policy_contract, policy_identity
from examples.real_robot.atom_common.sensors import DEFAULT_CAMERA_CONFIG, load_camera_specs, resize_with_pad
from tests.entrypoints.atom import test_atom_home_and_validation as validation_tests

ROOT = Path(__file__).resolve().parents[3]


def metadata():
    return {
        "artifact": "/models/step_1000",
        "policy_spec": {"state_dim": 18, "action_dim": 18, "action_horizon": 50},
        "input_schema": {
            "state": [18],
            "images": {
                name: [3, 224, 224] for name in ("top_camera", "left_camera", "right_camera")
            },
        },
        "output_schema": {"actions": [50, 18]},
    }


class DeploymentContractTest(unittest.TestCase):
    def test_refusal_does_not_claim_zero_commands_after_a_failed_home(self):
        output = io.StringIO()
        with (
            patch.object(run, "run", side_effect=RuntimeFault("initial Home failed")),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(run.main([]), 2)
        result = json.loads(output.getvalue().split(" ", 1)[1])
        self.assertIsNone(result["commands_sent"])

    def test_head_hold_preserves_initial_reference_without_mutating_prediction(self):
        reference = np.linspace(-0.2, 0.2, 17)
        actions = np.zeros((50, 18))
        actions[:, 16:] = 1.0
        before = actions.copy()
        bounds = np.tile([-2.0, 2.0], (17, 1))
        held = actuator_targets(actions, reference, bounds, head_mode="hold")
        active = actuator_targets(actions, reference, bounds, head_mode="policy")
        np.testing.assert_array_equal(held[:, 15:17], np.tile(reference[15:17], (50, 1)))
        np.testing.assert_array_equal(active[:, 15:17], actions[:, 16:])
        np.testing.assert_array_equal(held[:, 0], np.full(50, reference[0]))
        np.testing.assert_array_equal(actions, before)

    def test_unsupported_policy_fails_before_actuator_setup(self):
        for change in (
            lambda m: m.update(history_requires_dense_frames=True),
            lambda m: m["policy_spec"].update(action_dim=24),
            lambda m: m["input_schema"]["images"].update(top_camera=[3, 256, 256]),
            lambda m: m["policy_spec"].update(action_horizon=20),
        ):
            other = metadata()
            change(other)
            with self.assertRaises((RuntimeFault, ValueError)):
                policy_contract(other)
        contract, horizon = policy_contract(metadata(), 50)
        self.assertEqual(horizon, 50)
        self.assertEqual(contract.state_dim, 18)

    def test_camera_config_rejects_duplicate_devices_and_insufficient_warmup(self):
        original = json.loads(DEFAULT_CAMERA_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cameras.json"
            for key, value in (
                ("serial", original["cameras"][1]["serial"]),
                ("warmup_frames", 1),
                ("fps", True),
                ("name", "unknown"),
            ):
                data = copy.deepcopy(original)
                data["cameras"][0][key] = value
                path.write_text(json.dumps(data))
                with self.subTest(key=key), self.assertRaises(RuntimeFault):
                    load_camera_specs(path)

    def test_camera_preprocessing_preserves_rgb_and_aspect_ratio(self):
        rgb = np.full((2, 4, 3), [255, 31, 7], dtype=np.uint8)
        image = resize_with_pad(rgb, 8)
        self.assertEqual(image.shape, (8, 8, 3))
        self.assertTrue((image[:2] == 0).all())
        self.assertTrue((image[6:] == 0).all())
        np.testing.assert_array_equal(image[2:6], np.tile([255, 31, 7], (4, 8, 1)))

    def test_receipt_rejects_changed_camera_head_mode_task_and_policy(self):
        args = run.parse_args(["--sdk-root", "/sdk"])
        runtime = run.runtime_identity(args)
        policy = policy_identity(metadata())
        with tempfile.TemporaryDirectory() as folder:
            receipt = {
                "status": "passed",
                "sources": validation.source_fingerprint(),
                "repeat_runs": 2,
                "unit_tests_passed": True,
                "runtime_identity": runtime,
                "policy_identity": policy,
            }
            (Path(folder) / "atom-runtime-validation.json").write_text(json.dumps(receipt))
            validation.require_live_validation(
                folder, expected_runtime=runtime, expected_policy=policy
            )
            for key, value in (("head_mode", "policy"), ("task", "another task"), ("cameras", [])):
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(RuntimeFault, "settings changed"),
                ):
                    validation.require_live_validation(
                        folder, expected_runtime={**runtime, key: value}
                    )
            with self.assertRaisesRegex(RuntimeFault, "Policy changed"):
                validation.require_live_validation(
                    folder, expected_policy={**policy, "artifact": "/new"}
                )

    def test_invalid_receipt_prevents_home_workers_and_control(self):
        with tempfile.TemporaryDirectory() as folder:
            args = run.parse_args(
                [
                    "--sdk-root",
                    "/sdk",
                    "--gripper-sdk-root",
                    "/dh",
                    "--validation",
                    folder,
                    "--execute",
                    "--site-clear",
                ]
            )
            with (
                patch.object(run, "preflight_policy", return_value=(metadata(), 50)),
                patch.object(home, "return_home") as returning,
                patch.object(run.mp, "get_context") as context,
            ):
                with self.assertRaisesRegex(RuntimeFault, "read-only validation"):
                    run.run(args)
                returning.assert_not_called()
                context.assert_not_called()

    def test_validation_never_dispatches_an_execute_command(self):
        with (
            patch.object(validate, "preflight_policy") as connect,
            patch.object(validate.subprocess, "run") as child,
        ):
            with self.assertRaisesRegex(RuntimeFault, "never accepts"):
                validate.main(["--execute", "--site-clear"])
            connect.assert_not_called()
            child.assert_not_called()

    def test_copied_reports_do_not_prove_two_independent_runs(self):
        report = validation_tests.ValidationContractTest().valid()
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("one.json", "two.json")]
            for path in paths:
                path.write_text(json.dumps(report))
            with patch.object(
                verify.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()
            ):
                with self.assertRaisesRegex(SystemExit, "independent"):
                    verify.main(
                        [
                            "--report",
                            str(paths[0]),
                            "--report",
                            str(paths[1]),
                            "--validation",
                            folder,
                        ]
                    )

    def test_missing_zero_command_evidence_is_rejected(self):
        for field in ("commands_sent", "gripper_commands_sent", "motor_writer_created"):
            report = validation_tests.ValidationContractTest().valid()
            del report[field]
            with self.subTest(field=field), self.assertRaises(RuntimeFault):
                validation.validate_read_only_report(report)

    def test_runtime_defaults_hold_head_and_reject_unvalidated_action_frequency(self):
        self.assertEqual(run.parse_args([]).head_mode, "hold")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.parse_args(["--action-fps", "30"])

    def test_examples_import_without_torch_or_hardware_sdks(self):
        script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'pyrealsense2', 'cyclonedds', 'atom_training'}:
            raise AssertionError('Unexpected optional dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from examples.real_robot.atom_w_0759 import run, home, validate, verify
from examples.real_robot.shared.robot_contract import load_robot_layout
assert run.parse_args([]).head_mode == 'hold'
assert len(load_robot_layout('atom18').state_names) == 18
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)},
            capture_output=True,
            timeout=20,
        )

    def test_launcher_retains_relative_python_and_works_outside_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            caller = Path(folder)
            (caller / "python").write_text(
                "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
            )
            (caller / "python").chmod(0o755)
            for launcher in (
                "examples/real_robot/atom_w_0759/start_atom_runtime.sh",
                "examples/atom_robot/start_atom_runtime.sh",
            ):
                with self.subTest(launcher=launcher):
                    result = subprocess.run(
                        ["bash", str(ROOT / launcher), "--help"],
                        cwd=caller,
                        capture_output=True,
                        text=True,
                        timeout=20,
                        env={**os.environ, "ATOM_PYTHON": "./python"},
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--head-mode", result.stdout)

    def test_recipe_restores_artifact_horizon_and_uses_current_model_target(self):
        from dobot_nexus.core.config import load_config

        config = load_config(
            ROOT / "examples/recipes/infer/pi05_atom18.yaml", expected_task="infer"
        )
        self.assertEqual(config.model.state_dim, 18)
        self.assertEqual(config.model.action_dim, 18)
        self.assertIsNone(config.model.action_horizon)
        self.assertEqual(config.model.target, "dobot_nexus.models.pi05.modeling_pi05:Pi05Model")
        self.assertEqual(config.model.inference_parameter_precision, "bf16")
        self.assertEqual(config.runtime.autocast, "auto")
        self.assertEqual(config.runtime.compile_mode, "legacy")
        self.assertTrue(config.runtime.cuda_graph)
