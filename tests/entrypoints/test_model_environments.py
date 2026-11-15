from __future__ import annotations

import io
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.env import check_model_env, create_model_env

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "src" / "dobot_nexus" / "models"


class ModelEnvironmentTest(unittest.TestCase):
    def test_model_requirements_keep_incompatible_dependencies_separate(self):
        for model in create_model_env.MODEL_NAMES:
            with self.subTest(model=model):
                content = (MODEL_ROOT / model / "requirements.txt").read_text()
                self.assertIn("-r ../../../../requirements.txt", content)
                self.assertNotIn("transformers", content)
                if model in {"pi0", "pi05"}:
                    self.assertIn("datasets", content)
                    self.assertIn("lerobot", content)

    def test_creator_plans_all_models_and_extras_without_changing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "environment with spaces"
            for model in create_model_env.MODEL_NAMES:
                with self.subTest(model=model):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/env/create_model_env.py"),
                            "--model",
                            model,
                            "--venv",
                            str(venv),
                            "--python",
                            "/unused/python with spaces",
                            "--dry-run",
                            "--extra",
                            "websocket",
                            "--extra",
                            "serve",
                            "--extra",
                            "websocket",
                        ],
                        cwd=directory,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    commands = [
                        shlex.split(line[2:])
                        for line in result.stdout.splitlines()
                        if line.startswith("+ ")
                    ]
                    self.assertEqual(
                        commands[0], ["/unused/python with spaces", "-m", "venv", str(venv)]
                    )
                    self.assertIn(str(MODEL_ROOT / model / "requirements.txt"), commands[-2])
                    self.assertEqual(commands[-1][-1], f"{ROOT}[serve,websocket]")
                    self.assertNotIn("--no-deps", commands[-1])
                    validation = shlex.split(result.stdout.split("Validate with: ")[1])
                    self.assertEqual(validation[-4:], ["--extra", "serve", "--extra", "websocket"])
                    self.assertFalse(venv.exists())

    def test_creator_reuses_existing_environment_and_honors_skip_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory)
            python = create_model_env.environment_python(venv)
            python.parent.mkdir()
            python.touch()
            argv = [
                "create_model_env.py",
                "--model",
                "pi05",
                "--venv",
                str(venv),
                "--system-site-packages",
                "--skip-tool-upgrade",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(create_model_env.subprocess, "run") as run,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(create_model_env.main(), 0)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0][:2], [str(python), "-c"])
            self.assertFalse(
                any("venv" in command or "--upgrade" in command for command in commands)
            )
            self.assertEqual(commands[-1][-3:], ["--no-deps", "-e", str(ROOT)])
            self.assertIn("only applies when creating", stderr.getvalue())

    def test_checker_only_imports_selected_model_and_service_dependencies(self):
        for model in create_model_env.MODEL_NAMES:
            with self.subTest(model=model):
                argv = ["check_model_env.py", "--model", model, "--extra", "websocket"]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(
                        check_model_env.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0, "", ""),
                    ),
                    mock.patch.object(check_model_env.importlib, "import_module") as imports,
                    mock.patch.object(
                        check_model_env.importlib.metadata,
                        "version",
                        side_effect={"transformers": "4.57.3", "tokenizers": "0.22.2"}.__getitem__,
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(check_model_env.main(), 0)
                names = [call.args[0] for call in imports.call_args_list]
                self.assertEqual(len(names), len(set(names)))
                self.assertIn("dobot_nexus", names)
                self.assertIn("websockets.sync.client", names)
                self.assertNotIn("fastapi", names)
                self.assertNotIn("jax", names)
                self.assertNotIn("openpi", names)
                self.assertEqual("transformers" in names, False)

    def test_checker_reports_pip_import_and_version_failures_together(self):
        def import_module(name):
            if name == "fastapi":
                raise ImportError("not installed")

        with (
            mock.patch.object(
                sys, "argv", ["check_model_env.py", "--model", "pi05", "--extra", "serve"]
            ),
            mock.patch.object(
                check_model_env.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1, "dependency conflict", ""),
            ),
            mock.patch.object(
                check_model_env.importlib, "import_module", side_effect=import_module
            ),
            mock.patch.object(check_model_env.importlib.metadata, "version", return_value="0.0"),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(check_model_env.main(), 1)
        errors = stderr.getvalue()
        self.assertIn("pip check: dependency conflict", errors)
        self.assertIn("cannot import fastapi", errors)


if __name__ == "__main__":
    unittest.main()
