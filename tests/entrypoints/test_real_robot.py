from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from examples.real_robot.shared.resolve_model import resolve_model_path

ROOT = Path(__file__).resolve().parents[2]


class RealRobotEntrypointsTest(unittest.TestCase):
    def test_policy_launcher_keeps_the_callers_python_after_changing_directory(self):
        with tempfile.TemporaryDirectory(prefix="nexus policy caller ") as directory:
            caller = Path(directory).resolve()
            artifact = self._artifact(caller)
            probe = caller / "python-probe"
            probe.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1] == '-c':\n"
                "    print('torch=launcher-probe')\n"
                "elif sys.argv[1].endswith('resolve_model.py'):\n"
                "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
                "else:\n"
                "    print(json.dumps({'python': sys.argv[0], 'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n"
            )
            probe.chmod(0o755)
            executable = caller / ".venv/bin/python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(probe)
            # Exercise the actual exec path, stubbing only the Torch dependency
            # probe and server process; artifact resolution still runs normally.
            for selection in (".venv/bin/python", "python", str(executable)):
                with self.subTest(selection=selection):
                    result = subprocess.run(
                        ["bash", str(ROOT / "examples/pi05_real_robot/start_policy_server.sh"),
                         str(artifact.relative_to(caller))],
                        cwd=caller, capture_output=True, text=True, timeout=30,
                        env={**os.environ, "POLICY_PYTHON": selection,
                             "PATH": ".venv/bin:" + os.environ["PATH"],
                             "TOKENIZER_PATH": "", "OPENPI_TOKENIZER_PATH": ""},
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    child = json.loads(result.stdout)
                    self.assertEqual(child["python"], str(executable))
                    self.assertEqual(child["cwd"], str(ROOT))
                    self.assertEqual(child["args"][:2], ["-m", "dobot_nexus.models.pi05.deploy.server"])
                    index = child["args"].index("--artifact")
                    self.assertEqual(child["args"][index + 1], str(artifact))

    def test_robot_launchers_work_from_a_standalone_directory(self):
        # The robot receives only this directory, without the Nexus source tree.
        # --help exercises both imports and CLI assembly without opening hardware.
        for source in ("examples/real_robot/dobot_dual_arm_14d", "examples/pi05_real_robot"):
            with self.subTest(source=source):
                self.check_standalone_launchers(ROOT / source)

    def check_standalone_launchers(self, source):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "robot copy" / "pi05_real_robot"
            shutil.copytree(
                source,
                copied,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            for script in copied.glob("start_*.sh"):
                subprocess.run(["bash", "-n", str(script)], check=True)
            environment = {
                "PATH": os.environ["PATH"],
                "ROBOT_PYTHON": sys.executable,
                "CLIENT_PYTHON": sys.executable,
                "ENV_SRC": str(copied / "unused_sdk"),
                "TASK": "press the yellow button",
                "TOP_ROI_XYWH": "none",
            }
            for mode in ("standard", "memory"):
                for script_name in ("start_dobot_dual_arm_14d_server.sh", "start_robot_client.sh"):
                    with self.subTest(mode=mode, script=script_name):
                        result = subprocess.run(
                            ["bash", str(copied / script_name), "--help"],
                            cwd=directory,
                            env={**environment, "BRIDGE_MODE": mode},
                            capture_output=True,
                            text=True,
                            timeout=20,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn(
                            "--history-fps" if mode == "memory" else "--task", result.stdout
                        )

    def test_standard_bridge_rejects_memory_before_warmup_or_robot_connection(self):
        from examples.real_robot.dobot_dual_arm_14d._runtime import dobot_dual_arm_14d as bridge

        args = bridge.parse_args(["openpi-client"])
        policy = mock.Mock()
        policy.get_server_metadata.return_value = {"history_requires_dense_frames": True}
        with (
            mock.patch(
                "dobot_nexus.inference.transport.client.WebsocketPolicyClient", return_value=policy
            ),
            mock.patch.object(bridge, "ClientHealthServer"),
            mock.patch.object(bridge, "warmup_openpi_policy") as warmup,
            mock.patch.object(bridge.socket, "create_connection") as connect,
        ):
            with self.assertRaisesRegex(RuntimeError, "BRIDGE_MODE=memory"):
                bridge.run_openpi_client(args)
            warmup.assert_not_called()
            connect.assert_not_called()
            policy.infer.assert_not_called()
            policy.close.assert_called_once()

    def _artifact(self, root: Path) -> Path:
        artifact = root / "run/checkpoints/step_00000042"
        for relative in (
            "backend/full_weights.pt",
            "assets/norm_stats.json",
            "openpi_pytorch/model.safetensors",
        ):
            path = artifact / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
        (artifact / "config_resolved.yaml").write_text(
            "schema_version: 1\ntask: train\n", encoding="utf-8"
        )
        (artifact / "manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "files": {
                        "weights": "backend/full_weights.pt",
                        "resolved_config": "config_resolved.yaml",
                        "norm_stats": "assets/norm_stats.json",
                        "jax_weights": "openpi_pytorch/model.safetensors",
                    },
                }
            ),
            encoding="utf-8",
        )
        (artifact.parent / "latest.txt").write_text(f"{artifact.name}\n", encoding="utf-8")
        return artifact

    def test_client_rejects_explicit_dimension_mismatch_before_robot_connection(self):
        from examples.real_robot.dobot_dual_arm_14d._runtime import dobot_dual_arm_14d as bridge

        policy = mock.Mock()
        policy.get_server_metadata.return_value = {"state_dim": 34, "action_dim": 34}
        with (
            mock.patch("dobot_nexus.inference.transport.client.WebsocketPolicyClient",
                       return_value=policy),
            mock.patch.object(bridge, "ClientHealthServer"),
            mock.patch.object(bridge, "warmup_openpi_policy") as warmup,
            mock.patch.object(bridge.socket, "create_connection") as connect,
        ):
            with self.assertRaisesRegex(ValueError, "Configured state_dim=18 does not match policy 34"):
                bridge.run_openpi_client(bridge.parse_args(["openpi-client", "--state-dim", "18"]))
            warmup.assert_not_called()
            connect.assert_not_called()
            policy.close.assert_called_once()

    def test_model_path_resolves_artifact_run_and_weight_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            expected_weights = artifact / "backend/full_weights.pt"
            for supplied in (
                artifact,
                artifact / "openpi_pytorch",
                artifact / "openpi_pytorch/model.safetensors",
                expected_weights,
                artifact.parent / "latest.txt",
                artifact.parents[1],
            ):
                with self.subTest(supplied=supplied):
                    resolved = resolve_model_path(supplied)
                    self.assertEqual(resolved.artifact_dir, artifact.resolve())
                    self.assertEqual(resolved.torch_checkpoint, expected_weights.resolve())

    def test_standalone_weights_are_rejected_without_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.safetensors"
            weights.write_bytes(b"test")
            with self.assertRaisesRegex(FileNotFoundError, "complete Nexus Artifact"):
                resolve_model_path(weights)

    def test_schema_v2_uses_deployment_pointer_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            artifact = run / "checkpoints" / "step_00000007"
            for relative in (
                "deployment/pytorch/full_weights.pt",
                "deployment/assets/norm_stats.json",
            ):
                path = artifact / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            (artifact / "config_resolved.yaml").write_text(
                "schema_version: 1\ntask: train\n", encoding="utf-8"
            )
            (artifact / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 2,
                        "components": {"deployment": {"status": "ready"}},
                        "files": {
                            "weights": "deployment/pytorch/full_weights.pt",
                            "norm_stats": "deployment/assets/norm_stats.json",
                            "resolved_config": "config_resolved.yaml",
                        },
                    }
                ),
                encoding="utf-8",
            )
            pointer = run / "checkpoints" / "latest_deployment.txt"
            pointer.write_text(f"{artifact.name}\n", encoding="utf-8")

            resolved = resolve_model_path(run)
            self.assertEqual(resolved.artifact_dir, artifact.resolve())
            self.assertEqual(
                resolved.torch_checkpoint,
                (artifact / "deployment/pytorch/full_weights.pt").resolve(),
            )

    def test_legacy_jax_manifest_entry_is_not_required_for_torch_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            (artifact / "openpi_pytorch/model.safetensors").unlink()
            resolved = resolve_model_path(artifact)
            self.assertEqual(resolved.torch_checkpoint, artifact / "backend/full_weights.pt")

    def test_deployment_requires_packaged_tokenizer_and_memory_vision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            path = artifact / "manifest.yaml"
            manifest = yaml.safe_load(path.read_text())
            manifest["files"]["model_assets"] = {"data.tokenizer_path": "assets/tokenizer.model"}
            manifest["files"]["memory_vision"] = "assets/vision.npz"
            path.write_text(yaml.safe_dump(manifest))
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer"):
                resolve_model_path(artifact)
            (artifact / "assets/tokenizer.model").write_bytes(b"test")
            with self.assertRaisesRegex(FileNotFoundError, "memory_vision"):
                resolve_model_path(artifact)
            (artifact / "assets/vision.npz").write_bytes(b"test")
            self.assertEqual(resolve_model_path(artifact).artifact_dir, artifact)


if __name__ == "__main__":
    unittest.main()
