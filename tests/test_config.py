from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dobot_nexus.core.config import (
    Config, load_config, parse_override, resolve_training_runtime, save_config, validate_config,
)


class ConfigTest(unittest.TestCase):
    def test_auto_runtime_uses_process_count_without_mutating_template(self):
        config = Config(runtime=Config(launcher="auto", precision="bf16", fsdp=Config(
            enabled="auto", version=2, param_dtype="bf16",
        )), data=Config(batch_size=2), train=Config(grad_accum_steps=4))
        for count, launcher, fsdp in ((1, "local", False), (2, "accelerate", True),
                                      (8, "accelerate", True)):
            with self.subTest(count=count):
                resolved = resolve_training_runtime(config, world_size=count)
                self.assertEqual(resolved.launcher, launcher)
                self.assertIs(resolved.fsdp.enabled, fsdp)
                self.assertEqual(resolved.fsdp.version, 2)
                self.assertEqual(resolved.fsdp.param_dtype, "bf16")
                self.assertEqual(resolved.precision, "bf16")
        self.assertEqual(config.runtime.launcher, "auto")
        self.assertEqual(config.runtime.fsdp.enabled, "auto")
        self.assertEqual(config.data.batch_size, 2)
        self.assertEqual(config.train.grad_accum_steps, 4)

    def test_runtime_resolution_preserves_explicit_choices_and_old_configs(self):
        for count, launcher, enabled in ((1, "local", False), (1, "accelerate", False),
                                        (2, "accelerate", False), (2, "accelerate", True)):
            config = {"runtime": {"launcher": launcher, "fsdp": {"enabled": enabled}}}
            with self.subTest(config=config):
                self.assertEqual(resolve_training_runtime(config, world_size=count), config["runtime"])
        self.assertEqual(resolve_training_runtime({}, world_size=1), {})
        config = {"runtime": {"launcher": "auto", "fsdp": {"enabled": False}}}
        resolved = resolve_training_runtime(config, world_size=2)
        self.assertEqual(resolved.launcher, "accelerate")
        self.assertFalse(resolved.fsdp.enabled)

    def test_runtime_resolution_uses_world_size_not_visible_devices(self):
        config = {"runtime": {"launcher": "auto", "fsdp": {"enabled": "auto"}}}
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "CUDA_VISIBLE_DEVICES": "0"}):
            self.assertTrue(resolve_training_runtime(config).fsdp.enabled)
            self.assertFalse(resolve_training_runtime(config, world_size=1).fsdp.enabled)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "CUDA_VISIBLE_DEVICES": "0,1"}):
            self.assertEqual(resolve_training_runtime(config).launcher, "local")

    def test_runtime_resolution_rejects_incompatible_or_invalid_launches(self):
        for runtime, count, message in (
            ({"launcher": "local"}, 2, "exactly one"),
            ({"launcher": "auto", "fsdp": {"enabled": True}}, 1, "at least two"),
            ({"launcher": "accelerate", "fsdp": {"enabled": "false"}}, 2, "enabled"),
            ({"launcher": "unknown"}, 1, "launcher"),
            ({"fsdp": {"enabled": 1}}, 1, "enabled"),
        ):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, message):
                resolve_training_runtime({"runtime": runtime}, world_size=count)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "world_size"):
                resolve_training_runtime({}, world_size=count)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "invalid"}):
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
                resolve_training_runtime({})

    def test_dimensions_can_only_be_omitted_for_artifact_inference(self):
        config = {"schema_version": 1, "task": "infer", "model": {
            "target": "tests.fakes:DummyVLAModule", "artifact": "unused", "state_dim": None,
        }}
        validate_config(config)
        for value in (True, 2.5, "14", 0, -1):
            config["model"]["state_dim"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "state_dim"):
                validate_config(config)
        config["model"].pop("state_dim")
        config["model"].pop("artifact")
        config["model"]["checkpoint"] = "weights.pt"
        with self.assertRaisesRegex(ValueError, "state_dim"):
            validate_config(config)
        config["task"] = "train"
        config["model"]["artifact"] = "unused"
        with self.assertRaisesRegex(ValueError, "state_dim"):
            validate_config(config)

    def test_scalar_overrides(self):
        self.assertIs(parse_override("true"), True)
        self.assertIsNone(parse_override("null"))
        self.assertEqual(parse_override("[1, 2]"), [1, 2])

    def test_environment_and_dot_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """schema_version: 1
task: infer
model:
  target: tests.fakes:DummyVLAModule
  checkpoint: ${env:DUMMY_CHECKPOINT,default.pt}
  state_dim: 2
  action_dim: 2
  action_horizon: 3
runtime:
  device: cpu
""",
                encoding="utf-8",
            )
            os.environ["DUMMY_CHECKPOINT"] = "weights.pt"
            try:
                config = load_config(path, ["model.action_horizon=4"])
            finally:
                os.environ.pop("DUMMY_CHECKPOINT", None)
            self.assertEqual(config.model.checkpoint, "weights.pt")
            self.assertEqual(config.model.action_horizon, 4)

    def test_resume_limit_must_be_non_negative(self):
        config = {
            "schema_version": 1,
            "task": "train",
            "model": {
                "target": "tests.fakes:DummyVLAModule",
                "state_dim": 2,
                "action_dim": 2,
                "action_horizon": 3,
            },
            "data": {},
            "train": {"resume_limit": -1},
        }
        with self.assertRaisesRegex(ValueError, "resume_limit"):
            validate_config(config)

    def test_recipe_merges_relative_bases_before_environment_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "recipes").mkdir()
            (root / "base.yaml").write_text(
                "schema_version: 1\ntask: infer\n"
                "model:\n  target: tests.fakes:DummyVLAModule\n"
                "  checkpoint: ${env:RECIPE_UNUSED_REQUIRED_PATH}\n"
                "  state_dim: 2\n  action_dim: 2\n  action_horizon: 3\n"
                "runtime:\n  device: cpu\n  fsdp:\n    enabled: true\n"
                "    transformer_layer_cls_to_wrap: [Block, Encoder1DBlock]\n"
                "inference:\n  seed: 42\n"
            )
            (root / "variant.yaml").write_text(
                "extends: base.yaml\nruntime:\n  fsdp:\n    enabled: false\n"
            )
            recipe = root / "recipes" / "task.yaml"
            recipe.write_text(
                "extends: ../variant.yaml\n"
                "model:\n  checkpoint: ${env:RECIPE_WEIGHTS,weights.pt}\n"
                "runtime:\n  fsdp:\n    transformer_layer_cls_to_wrap: [CustomBlock]\n"
                "inference:\n  seed: null\n"
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                config = load_config(recipe, ["model.action_horizon=8"], expected_task="infer")
            self.assertEqual(config.model.checkpoint, "weights.pt")
            self.assertEqual(config.model.action_horizon, 8)
            self.assertEqual(config.model.state_dim, 2)
            self.assertFalse(config.runtime.fsdp.enabled)
            self.assertEqual(config.runtime.fsdp.transformer_layer_cls_to_wrap, ["CustomBlock"])
            self.assertIsNone(config.inference.seed)
            self.assertEqual(config.exp.config_path, str(recipe))
            self.assertNotIn("extends", config)

            # Artifacts remain portable after both recipe and template disappear.
            resolved = root / "config_resolved.yaml"
            save_config(config, resolved)
            recipe.unlink()
            (root / "variant.yaml").unlink()
            (root / "base.yaml").unlink()
            self.assertEqual(load_config(resolved), config)

    def test_recipe_rejects_cycles_missing_bases_and_invalid_declarations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a.yaml", root / "b.yaml"
            first.write_text("extends: b.yaml\n")
            second.write_text("extends: a.yaml\n")
            with self.assertRaisesRegex(ValueError, "Circular.*a.yaml.*b.yaml"):
                load_config(first)
            second.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "b.yaml"):
                load_config(first)
            for value in ("[]", "false", "null", '""'):
                first.write_text(f"extends: {value}\n")
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "one YAML"):
                    load_config(first)
            first.write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "root must be a mapping"):
                load_config(first)

    def test_recipe_templates_do_not_leak_mutations_between_loads(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "examples/configs/train/pi05.yaml"
        first = load_config(source)
        first.runtime.fsdp.enabled = False
        self.assertEqual(load_config(source).runtime.fsdp.enabled, "auto")
        with self.assertRaisesRegex(ValueError, "command-line override"):
            load_config(source, ["extends=another.yaml"])


if __name__ == "__main__":
    unittest.main()
