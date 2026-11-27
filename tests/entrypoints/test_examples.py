from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dobot_nexus.core.config import load_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TRAIN_CONFIGS = EXAMPLES / "configs" / "train"
INFER_CONFIGS = EXAMPLES / "configs" / "infer"
RECIPES = EXAMPLES / "recipes"
BC_CONFIGS = {
    "pi0": TRAIN_CONFIGS / "pi0.yaml",
    "pi05": TRAIN_CONFIGS / "pi05.yaml",
}


class UserExamplesTest(unittest.TestCase):
    def test_documented_pi0_memory_infer_command(self):
        template = INFER_CONFIGS / "pi0_memory.yaml"
        command, = [shlex.split(line[2:]) for line in template.read_text().splitlines()
                    if line.startswith("# python -m ")]
        self.assertEqual(command[:3], ["python", "-m", "dobot_nexus.models.pi0.memory.infer"])
        self.assertEqual(ROOT / command[command.index("--config") + 1], template)
        result = subprocess.run([sys.executable, *command[1:], "--help"], cwd=ROOT,
                                capture_output=True, text=True, check=True, timeout=30)
        for flag in ("--config", "--input", "--output"):
            self.assertIn(flag, result.stdout)

    def test_launchers_keep_the_callers_python_after_changing_directory(self):
        with tempfile.TemporaryDirectory(prefix="nexus caller ") as directory:
            caller = Path(directory).resolve()
            probe = caller / "python-probe"
            probe.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1] == '-c':\n"
                "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
                "print(json.dumps({'python': sys.argv[0], 'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n"
            )
            probe.chmod(0o755)
            executable = caller / ".venv/bin/python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(probe)
            (caller / "artifact").mkdir()
            (caller / "observation.npz").touch()
            launches = [
                ("train.sh", ["--config", str(BC_CONFIGS["pi05"]), "--gpus", "0"],
                 ["-m", "dobot_nexus.cli.main", "train"]),
                ("train.sh", ["--config", str(BC_CONFIGS["pi05"]), "--gpus", "0,1"],
                 ["-m", "torch.distributed.run"]),
                ("infer.sh", ["--model", "pi05", "--artifact", "artifact", "--input", "observation.npz"],
                 ["-m", "dobot_nexus.cli.main", "infer"]),
                ("infer.sh", ["--model", "pi05", "--artifact", "artifact", "--serve"],
                 ["-m", "dobot_nexus.cli.main", "serve"]),
            ]
            for script, arguments, prefix in launches:
                for selection in ("argument", "environment", "path", "absolute"):
                    with self.subTest(script=script, arguments=prefix, selection=selection):
                        env = {**os.environ, "PYTHON_BIN": "python"}
                        options = []
                        if selection == "argument":
                            options = ["--python", ".venv/bin/python"]
                        elif selection == "environment":
                            env["PYTHON_BIN"] = ".venv/bin/python"
                        elif selection == "path":
                            env["PATH"] = ".venv/bin:" + env["PATH"]
                        else:
                            options = ["--python", str(executable)]
                        result = subprocess.run(
                            [str(EXAMPLES / script), *arguments, *options],
                            cwd=caller, env=env, capture_output=True, text=True, timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        child = json.loads(result.stdout)
                        self.assertEqual(child["python"], str(executable))
                        self.assertEqual(child["cwd"], str(ROOT))
                        self.assertEqual(child["args"][:len(prefix)], prefix)

    def test_single_gpu_child_does_not_inherit_torchrun_process_identity(self):
        inherited = {
            "WORLD_SIZE": "8", "RANK": "3", "LOCAL_RANK": "3",
            "LOCAL_WORLD_SIZE": "8", "GROUP_RANK": "1", "ROLE_RANK": "3",
            "ROLE_WORLD_SIZE": "8", "MASTER_ADDR": "stale-master.invalid",
            "MASTER_PORT": "29599",
        }
        with tempfile.TemporaryDirectory() as directory:
            # Execute the launcher's real child path, replacing only training
            # with a config probe so this contract test needs no CUDA or weights.
            probe = Path(directory) / "python-probe"
            probe.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1] == '-c':\n"
                "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
                "assert sys.argv[1:4] == ['-m', 'dobot_nexus.cli.main', 'train']\n"
                "from dobot_nexus.core.config import load_config, resolve_training_runtime\n"
                "index = sys.argv.index('--config')\n"
                "config = load_config(sys.argv[index + 1], sys.argv[index + 2:])\n"
                "runtime = resolve_training_runtime(config)\n"
                "print(json.dumps({'launcher': runtime.launcher, 'fsdp': runtime.fsdp.enabled,\n"
                f"    'inherited': {{key: os.environ.get(key) for key in {tuple(inherited)!r}}},\n"
                "    'devices': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
            )
            probe.chmod(0o755)
            for model in ("pi0", "pi05"):
                for launcher in ("auto", "local", "accelerate"):
                    with self.subTest(model=model, launcher=launcher):
                        result = subprocess.run(
                            [str(EXAMPLES / "train.sh"), "--python", str(probe),
                             "--config", str(BC_CONFIGS[model]), "--gpus", "7", "--",
                             f"runtime.launcher={launcher}"],
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, **inherited},
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        child = json.loads(result.stdout)
                        expected = "local" if launcher == "auto" else launcher
                        self.assertEqual(child["launcher"], expected)
                        self.assertIs(child["fsdp"], False)
                        self.assertEqual(child["inherited"], dict.fromkeys(inherited))
                        self.assertEqual(child["devices"], "7")
                        self.assertIn(f"launcher={expected} fsdp=false", result.stderr)

    def test_same_training_template_selects_single_or_multiple_gpu_runtime(self):
        for model in ("pi0", "pi05"):
            for gpus, launcher, fsdp in (("0", "local", "false"),
                                         ("0,1", "accelerate", "true")):
                with self.subTest(model=model, gpus=gpus):
                    result = subprocess.run(
                        [str(EXAMPLES / "train.sh"), "--python", sys.executable,
                         "--config", str(BC_CONFIGS[model]), "--gpus", gpus, "--dry-run"],
                        capture_output=True, text=True, check=True,
                        # Preflight must use --gpus, even inside another torchrun environment.
                        env={**os.environ, "WORLD_SIZE": "8"},
                    )
                    self.assertIn(f"launcher={launcher} fsdp={fsdp}", result.stderr)
                    self.assertEqual("torch.distributed.run" in result.stdout, gpus == "0,1")
                    self.assertNotIn("runtime.launcher=", result.stdout)
                    self.assertNotIn("runtime.fsdp.enabled=", result.stdout)

    def test_training_templates_enable_local_tensorboard(self):
        for path in [*TRAIN_CONFIGS.glob('*.yaml'), *RECIPES.glob('*.yaml')]:
            with self.subTest(path=path.name):
                config = load_config(path, expected_task='train')
                self.assertTrue(config.tracking.tensorboard.enabled)
                self.assertFalse(config.tracking.wandb.enabled)

    def test_examples_are_grouped_by_task_with_single_entrypoints(self) -> None:
        for script_name in ("train.sh", "infer.sh"):
            script = EXAMPLES / script_name
            self.assertTrue(script.is_file())
            self.assertNotEqual(script.stat().st_mode & 0o111, 0)
            subprocess.run(["bash", "-n", str(script)], check=True)

        expected_configs = {
            "pi0.yaml",
            "pi0_recap.yaml",
            "pi0_memory.yaml",
            "pi05.yaml",
            "pi05_recap.yaml",
            "pi05_memory.yaml",
        }
        self.assertEqual(
            {path.name for path in TRAIN_CONFIGS.glob("*.yaml")},
            expected_configs,
        )
        for path in TRAIN_CONFIGS.glob("*.yaml"):
            self.assertEqual(load_config(path, expected_task="train").algorithm, "bc")
        self.assertEqual(
            {path.name for path in RECIPES.glob("*.yaml")},
            {
                "pi05_usb_bc.yaml",
                "pi05_usb_recap.yaml",
                "pi05_button_bc.yaml",
                "pi05_button_memory.yaml",
                "pi05_kbd_pick_0916_full34.yaml",
            },
        )
        for path in RECIPES.glob("*.yaml"):
            config = load_config(path, expected_task="train")
            self.assertEqual(config.algorithm, "bc")
            self.assertNotIn("extends", config)
            self.assertTrue(config.runtime.fsdp.enabled)
        self.assertEqual(
            {path.name for path in INFER_CONFIGS.glob("*.yaml")},
            expected_configs | {"pi05_rltoken.yaml"},
        )
        for path in INFER_CONFIGS.glob("*.yaml"):
            config = load_config(path, expected_task="infer")
            self.assertEqual(config.task, "infer")
            if path.name in {"pi0_recap.yaml", "pi0_memory.yaml"}:
                self.assertEqual(config.server.host, "127.0.0.1")

        self.assertEqual(list(EXAMPLES.rglob("norm_stats.json")), [])

    def test_pi05_generic_configs_and_task_recipes_have_distinct_semantics(self):
        for name in ("pi05", "pi05_recap", "pi05_memory"):
            config = load_config(TRAIN_CONFIGS / f"{name}.yaml")
            self.assertEqual(config.data.default_prompt, "")
            self.assertNotIn("global_batch_size", config.data)
            self.assertNotIn("openpi_root", config.model)
            self.assertNotIn("jax_export_base_checkpoint", config.model)
        button = load_config(RECIPES / "pi05_button_bc.yaml")
        self.assertEqual(button.data.transforms.action.type, "delta")
        self.assertEqual(
            button.data.transforms.action.delta_mask, [True] * 6 + [False] + [True] * 6 + [False]
        )
        self.assertEqual(button.train.max_steps, 10000)
        memory = load_config(RECIPES / "pi05_button_memory.yaml")
        self.assertTrue(memory.model.target.endswith(":Pi05MemoryModel"))
        self.assertEqual(memory.data.transforms.action.type, "absolute")
        self.assertEqual(memory.data.normalization, "quantile")
        self.assertEqual(memory.data.batch_size * 4 * memory.train.grad_accum_steps, 64)
        self.assertEqual(memory.data.global_batch_size, 64)
        self.assertEqual(memory.scheduler.total_steps, memory.train.max_steps)
        self.assertEqual(memory.train.max_steps, 1990)
        self.assertEqual(memory.model.memory.budget, 512)
        kbd = load_config(RECIPES / "pi05_kbd_pick_0916_full34.yaml")
        self.assertEqual((kbd.model.state_dim, kbd.model.action_dim), (34, 34))
        self.assertEqual(kbd.model.model_action_dim, 64)
        self.assertEqual(kbd.runtime.fsdp.version, 2)
        self.assertEqual(kbd.runtime.distributed_timeout_seconds, 7200)
        self.assertEqual(kbd.data.transforms.action.type, "absolute")
        self.assertEqual(kbd.data.field_mapping.images.base, "observation.images.head")
        self.assertFalse(kbd.tracking.wandb.enabled)

    def test_recipe_launcher_resolves_templates_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    str(EXAMPLES / "train.sh"),
                    "--config",
                    str(RECIPES / "pi05_usb_bc.yaml"),
                    "--gpus",
                    "0,1",
                    "--dry-run",
                    "--",
                    "data.batch_size=4",
                    "train.grad_accum_steps=84",
                ],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("launcher=accelerate", result.stderr)
        self.assertIn("torch.distributed.run", result.stdout)
        self.assertIn("train.grad_accum_steps=84", result.stdout)

    def test_pi_training_launchers_resolve_relative_configs_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            checkpoint = root / "checkpoint"
            dataset.mkdir()
            checkpoint.mkdir()
            for model in ("pi0", "pi05"):
                with self.subTest(model=model):
                    result = subprocess.run(
                        [
                            str(EXAMPLES / "train.sh"),
                            "--config",
                            f"examples/configs/train/{model}.yaml",
                            "--gpus",
                            "0,1",
                            "--dry-run",
                            "--",
                            f"data.path={dataset}",
                            f"model.checkpoint={checkpoint}",
                        ],
                        cwd=root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertIn("CUDA_VISIBLE_DEVICES=0,1", result.stdout)
                    self.assertIn("torch.distributed.run", result.stdout)
                    self.assertIn(f"data.path={dataset}", result.stdout)
                    self.assertIn(f"model.checkpoint={checkpoint}", result.stdout)
                    self.assertNotIn("dobot_nexus_v1_path", result.stdout)

    def test_training_launcher_is_selected_from_config(self) -> None:
        local_result = subprocess.run(
            [
                str(EXAMPLES / "train.sh"),
                "--config",
                str(BC_CONFIGS["pi0"]),
                "--gpus",
                "0,1",
                "--dry-run",
                "--",
                "runtime.launcher=local",
                "runtime.fsdp.enabled=false",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(local_result.returncode, 2)
        self.assertIn("runtime.launcher=local", local_result.stderr)

        fsdp_result = subprocess.run(
            [
                str(EXAMPLES / "train.sh"),
                "--config",
                str(BC_CONFIGS["pi05"]),
                "--gpus",
                "0",
                "--dry-run",
                "--",
                "runtime.fsdp.enabled=true",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(fsdp_result.returncode, 2)
        self.assertIn("enables FSDP", fsdp_result.stderr)

    def test_unified_inference_resolves_each_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            observation = root / "observation.npz"
            artifact.mkdir()
            observation.touch()

            for model in ("pi05", "pi0"):
                with self.subTest(model=model):
                    command = [
                        str(EXAMPLES / "infer.sh"),
                        "--model",
                        model,
                        "--artifact",
                        str(artifact),
                        "--input",
                        str(observation),
                        "--dry-run",
                    ]
                    result = subprocess.run(
                        command,
                        cwd=root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertIn(f"examples/configs/infer/{model}.yaml", result.stdout)
                    self.assertIn(f"model.artifact={artifact}", result.stdout)
                    self.assertIn(f"--input {observation}", result.stdout)


if __name__ == "__main__":
    unittest.main()
