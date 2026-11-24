from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.artifacts import ArtifactManager, load_manifest
from dobot_nexus.core.config import Config, load_config
from dobot_nexus.core.contracts import ObservationBatch, PolicyOutput, SamplingOptions
from dobot_nexus.inference import InferencePolicy
from dobot_nexus.models.pi0.configuration_pi0 import Pi0Config
from dobot_nexus.models.pi0.data import Pi0Pipeline
from dobot_nexus.models.pi0.modeling_pi0 import Pi0Model
from dobot_nexus.models.pi0.training.checkpoint import normalize_state_dict
from dobot_nexus.models.pi0.training.source import load_core_source
from dobot_nexus.training import BCTrainer
from dobot_nexus.training.runtime import LocalRuntime

ROOT = Path(__file__).resolve().parents[3]


def _wrap(value):
    if isinstance(value, dict):
        return Config({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _write_fake_core(root: Path) -> None:
    package = root / "rlinf/models/embodiment/openpi_rlinf/pi0_model"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "model.py").write_text(
        """from dataclasses import dataclass
@dataclass
class Observation:
    images: dict
    image_masks: dict
    state: object
    tokenized_prompt: object = None
    tokenized_prompt_mask: object = None
    token_ar_mask: object = None
    token_loss_mask: object = None
    pcd_xyz: object = None
    @classmethod
    def from_dict(cls, value):
        return cls(
            images=value['image'], image_masks=value['image_mask'],
            state=value['state'], tokenized_prompt=value.get('tokenized_prompt'),
            tokenized_prompt_mask=value.get('tokenized_prompt_mask'))
""",
        encoding="utf-8",
    )
    (package / "pi0_config.py").write_text(
        """class Pi0Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def create(self):
        from .pi0 import Pi0
        model = Pi0(self.kwargs['action_dim'], self.kwargs['action_horizon'])
        model.config_kwargs = self.kwargs
        return model
""",
        encoding="utf-8",
    )
    (package / "pi0.py").write_text(
        """import torch
class Pi0(torch.nn.Module):
    def __init__(self, action_dim, action_horizon):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.gradient_checkpointing = False
    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True
    def compute_loss(self, observation, actions, train=True):
        return ((actions * self.weight) ** 2).mean(dim=-1)
    def sample_actions(self, observation, num_steps=10, noise=None, rng=None):
        shape = (observation.state.shape[0], self.action_horizon, self.action_dim)
        if noise is None:
            noise = torch.rand(
                shape, device=self.weight.device, dtype=self.weight.dtype, generator=rng)
        return noise * self.weight
""",
        encoding="utf-8",
    )


def _model_config(root: Path, checkpoint: Path, stats: Path, tokenizer: Path) -> Config:
    return _wrap(
        {
            "task": "train",
            "dependencies": {"rlinf_path": str(root)},
            "model": {
                "name": "pi0",
                "target": "dobot_nexus.models.pi0.modeling_pi0:Pi0Model",
                "checkpoint": str(checkpoint),
                "precision": "fp32",
                "parameter_precision": "fp32",
                "state_dim": 2,
                "action_dim": 2,
                "action_horizon": 2,
                "image_size": [224, 224],
            },
            "data": {
                "norm_stats_path": str(stats),
                "tokenizer_path": str(tokenizer),
                "field_mapping": {
                    "images": {
                        "base": "base_image",
                        "left_wrist": "left_image",
                        "right_wrist": "right_image",
                    },
                    "state": "robot_state",
                    "action": "robot_action",
                    "language": "task",
                },
                "transforms": {"action": {"type": "absolute"}},
            },
            "runtime": {"precision": "fp32"},
            "train": {"gradient_checkpointing": True},
        }
    )


class Pi0DirectCoreTest(unittest.TestCase):
    def test_explicit_projection_casts_keep_fp32_parameters_and_gradients(self):
        from dobot_nexus.models.pi0.numerics import configure_input_dtype_linears

        core = torch.nn.Sequential(torch.nn.Linear(8, 4))
        weight = core[0].weight
        keys = list(core.state_dict())
        configure_input_dtype_linears(core)
        self.assertIs(core[0].weight, weight)
        self.assertEqual(list(core.state_dict()), keys)
        with torch.autocast('cpu', enabled=False):
            output = core(torch.randn(2, 8, dtype=torch.bfloat16))
            self.assertEqual(output.dtype, torch.bfloat16)
            output.float().square().mean().backward()
        self.assertEqual(weight.dtype, torch.float32)
        self.assertEqual(weight.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_supplied_noise_preserves_fp32_independently_of_parameter_dtype(self) -> None:
        class NoiseCore(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))

            def sample_actions(self, observation, *, noise, **kwargs):
                return noise

        observation = ObservationBatch.from_value({
            "images": {"base": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": [0.0, 0.0], "prompt": "pick",
        })
        noise = np.random.default_rng(42).standard_normal((2, 32)).astype(np.float32)
        config = load_config(ROOT / "examples/configs/infer/pi0.yaml", [
            "model.artifact=null", "model.checkpoint=unused",
            "model.state_dim=2", "model.action_dim=2", "model.action_horizon=2",
        ], expected_task="infer")
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(parameter_dtype=dtype):
                model = Pi0Model.__new__(Pi0Model)
                torch.nn.Module.__init__(model)
                model.mode = "infer"
                model.core_model = NoiseCore(dtype)
                model.pi0_config = Pi0Config.from_config(config, mode="infer")
                received = []

                def decode(actions, aux):
                    received.append(actions)
                    return PolicyOutput(actions[:, :, :2].float().numpy())

                model.adapter = SimpleNamespace(
                    encode_observation=lambda obs: obs,
                    move_encoded=lambda obs, device: SimpleNamespace(
                        observation=obs, raw_state=obs.state),
                    decode_actions=decode,
                )
                for value in (noise, noise[None]):
                    model.predict_action_chunk(observation, SamplingOptions(noise=value, num_steps=1))
                    self.assertEqual(received[-1].dtype, torch.float32)
                    np.testing.assert_array_equal(received[-1].numpy(), noise[None])

    def test_fp32_inference_parameters_support_bf16_operator_inputs(self) -> None:
        class ProjectionCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(32, 32, dtype=torch.float32)

            def sample_actions(self, observation, **kwargs):
                del observation, kwargs
                return self.proj(torch.ones((1, 2, 32), dtype=torch.bfloat16))

        model = Pi0Model.__new__(Pi0Model)
        torch.nn.Module.__init__(model)
        model.mode = "infer"
        model.core_model = ProjectionCore()
        config = load_config(ROOT / "examples/configs/infer/pi0.yaml", [
            "model.checkpoint=unused", "model.inference_parameter_precision=fp32",
            "model.state_dim=2", "model.action_dim=2", "model.action_horizon=2",
        ], expected_task="infer")
        model.pi0_config = Pi0Config.from_config(config, mode="infer")
        model.adapter = SimpleNamespace(
            encode_observation=lambda obs: obs,
            move_encoded=lambda obs, device: SimpleNamespace(observation=obs, raw_state=obs.state),
            decode_actions=lambda actions, aux: PolicyOutput(actions[:, :, :2].float().numpy()),
        )
        observation = ObservationBatch.from_value({
            "images": {"base": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": [0.0, 0.0], "prompt": "pick",
        })
        result = model.predict_action_chunk(observation, SamplingOptions(num_steps=1))
        self.assertEqual(result.actions.shape, (1, 2, 2))
        self.assertTrue(np.isfinite(result.actions).all())
        self.assertEqual(model.core_model.proj.weight.dtype, torch.float32)

        seen = []
        def sample(observation, **kwargs):
            seen.append(torch.is_autocast_enabled("cpu"))
            return torch.ones(1, 2, 32)
        with mock.patch.object(model.core_model, "sample_actions", side_effect=sample):
            for policy, enabled in (("auto", True), ("off", False)):
                model.pi0_config = replace(model.pi0_config, autocast=policy)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    model.predict_action_chunk(observation, SamplingOptions(num_steps=1))
                self.assertEqual(seen[-1], enabled)
            model.pi0_config = replace(model.pi0_config, autocast="auto")
            model.to(torch.bfloat16)
            model.predict_action_chunk(observation, SamplingOptions(num_steps=1))
            self.assertFalse(seen[-1])
        config.runtime.autocast = False
        self.assertFalse(Pi0Config.from_config(config, mode="infer").inference_autocast_enabled)

    def test_inference_precision_combinations_are_validated_before_loading(self) -> None:
        config = load_config(ROOT / "examples/configs/infer/pi0.yaml", [
            "model.artifact=null", "model.checkpoint=unused",
            "model.state_dim=2", "model.action_dim=2", "model.action_horizon=2",
        ], expected_task="infer")
        # Training-only precision must not select the inference parameter dtype.
        config.model.parameter_precision = "fp32"
        for compute, parameters in (("bf16", "bf16"), ("bf16", "fp32"), ("fp32", "fp32")):
            with self.subTest(compute=compute, parameters=parameters):
                config.model.precision = compute
                config.model.inference_parameter_precision = parameters
                resolved = Pi0Config.from_config(config, mode="infer")
                self.assertEqual(resolved.precision, compute)
                self.assertEqual(resolved.parameter_precision, parameters)
        for compute in ("bf16", "fp32"):
            config.model.precision = compute
            config.model.pop("inference_parameter_precision", None)
            self.assertEqual(Pi0Config.from_config(config, mode="infer").parameter_precision, compute)
        for parameters in ("bf16", "float16"):
            config.model.precision = "float32"
            config.model.inference_parameter_precision = parameters
            with self.subTest(parameters=parameters), self.assertRaisesRegex(
                ValueError, "FP32 inference compute requires"
            ):
                Pi0Config.from_config(config, mode="infer")

    def test_user_configs_hide_operator_backend_details(self) -> None:
        train = load_config(ROOT / "examples/configs/train/pi0.yaml", expected_task="train")
        infer = load_config(ROOT / "examples/configs/infer/pi0.yaml", expected_task="infer")
        self.assertNotIn("dependencies", train)
        self.assertNotIn("backend", train.model)
        self.assertNotIn("dependencies", infer)
        self.assertNotIn("backend", infer.model)
        self.assertEqual(
            set(train.data.field_mapping.images), {"base", "left_wrist", "right_wrist"}
        )
        self.assertEqual(train.model.precision, "bf16")
        self.assertEqual(train.model.parameter_precision, "fp32")
        self.assertEqual(train.runtime.fsdp.sharding_strategy, "full_shard")

        config = Pi0Config.from_config(train, mode="train")
        self.assertEqual(config.max_token_len, 48)
        self.assertFalse(config.discrete_state_input)
        self.assertFalse(config.core_kwargs()["pi05"])
        pipeline = Pi0Pipeline(train, config)
        self.assertEqual(pipeline.tokenizer_path.name, "pi0_tokenizer.model")
        self.assertGreater(pipeline.tokenizer_path.stat().st_size, 1_000_000)

    def test_config_rejects_invalid_precision_and_camera_mapping(self) -> None:
        config = _wrap(
            {
                "model": {
                    "checkpoint": "unused.pt",
                    "precision": "bf16",
                    "parameter_precision": "bf16",
                    "state_dim": 2,
                    "action_dim": 2,
                    "action_horizon": 2,
                    "image_size": [224, 224],
                },
                "runtime": {"precision": "bf16"},
            }
        )
        with self.assertRaisesRegex(ValueError, "parameter_precision='fp32'"):
            Pi0Config.from_config(config, mode="train")

        config.model.parameter_precision = "fp32"
        config.model.io = Config({"cameras": Config({"base": "base_0_rgb"})})
        with self.assertRaisesRegex(ValueError, "must populate"):
            Pi0Config.from_config(config, mode="train")

    def test_checkpoint_prefixes_become_bare_core_keys(self) -> None:
        state = normalize_state_dict(
            {
                "_fsdp_wrapped_module.module.core_model.llm.weight": torch.ones(1),
                "backend_model.model.img.bias": torch.zeros(1),
            }
        )
        self.assertEqual(set(state), {"llm.weight", "img.bias"})

    def test_wrapped_official_checkpoint_is_converted_locally(self) -> None:
        vision_prefix = (
            "model.paligemma_with_expert.paligemma.model.vision_tower."
            "vision_model.embeddings.patch_embedding"
        )
        state = normalize_state_dict(
            {
                vision_prefix + ".weight": torch.ones(1),
                "model.state_proj.weight": torch.zeros(1),
            }
        )
        self.assertEqual(set(state), {"img.stem.weight", "state_proj.weight"})

    def test_leaf_loader_does_not_import_rlinf_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fake_core(root)
            before = set(sys.modules)
            source = load_core_source(repo_root=root)
            added = set(sys.modules) - before
            self.assertFalse(source.embedded)
            self.assertIs(
                source.Observation, sys.modules[source.package_name + ".model"].Observation
            )
            self.assertFalse(any(name == "rlinf" or name.startswith("rlinf.") for name in added))
            self.assertFalse(any(name == "ray" or name.startswith("ray.") for name in added))

    def test_official_pi0_conversion_preserves_all_action_expert_rmsnorms(self) -> None:
        prefix = "model.paligemma_with_expert.gemma_expert.model."
        original = {prefix + "norm.weight": torch.arange(4, dtype=torch.float32)}
        expected = {"llm.final_norms.1.scale": original[prefix + "norm.weight"]}
        for index in range(18):
            for source_name, target_name in (
                ("input_layernorm", "pre_attention_norms"),
                ("post_attention_layernorm", "pre_ffw_norms"),
            ):
                value = torch.full((4,), float(index + 1))
                original[f"{prefix}layers.{index}.{source_name}.weight"] = value
                expected[f"llm.layers.{index}.{target_name}.1.scale"] = value
        converted = normalize_state_dict(original)
        self.assertEqual(set(converted), set(expected))
        for key, value in expected.items():
            torch.testing.assert_close(converted[key], value, rtol=0, atol=0)

    def test_default_core_is_pi0_owned_and_ignores_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DOBOT_PI0_CORE_PATH": "/missing/rlinf", "RLINF_PATH": "/also/missing"},
        ):
            source = load_core_source()
        self.assertTrue(source.embedded)
        self.assertEqual(source.revision, "89b0cd5fce52")
        self.assertIn("models/pi0/training/vendor", source.core_dir.as_posix())
        self.assertTrue((source.core_dir / "LICENSE").is_file())
        self.assertTrue((source.core_dir / "NOTICE.md").is_file())

    def test_model_runs_direct_core_train_and_infer_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fake_core(root)
            checkpoint = root / "weights.pt"
            torch.save({"weight": torch.tensor(2.0)}, checkpoint)
            stats = root / "norm_stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                        "actions": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    }
                ),
                encoding="utf-8",
            )
            tokenizer = root / "paligemma_tokenizer.model"
            tokenizer.write_bytes(b"test")
            config = _model_config(root, checkpoint, stats, tokenizer)

            train_model = Pi0Model(config, mode="train")
            self.assertFalse(train_model.core_model.config_kwargs["pi05"])
            self.assertFalse(train_model.core_model.config_kwargs["discrete_state_input"])
            self.assertEqual(train_model.core_model.config_kwargs["max_token_len"], 48)
            self.assertTrue(train_model.core_model.gradient_checkpointing)
            batch = {
                "image": {
                    name: torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
                    for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                },
                "image_mask": {
                    name: torch.ones(1, dtype=torch.bool)
                    for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                },
                "state": torch.zeros((1, 32)),
                "tokenized_prompt": torch.zeros((1, 48), dtype=torch.long),
                "tokenized_prompt_mask": torch.ones((1, 48), dtype=torch.bool),
                "actions": torch.ones((1, 2, 32)),
            }
            encoded_batch = train_model.adapter.encode_train_batch(batch, torch.device("cpu"))
            train_output = train_model.training_step(encoded_batch)
            self.assertAlmostEqual(float(train_output.loss.detach()), 4.0)
            self.assertTrue(train_output.loss.requires_grad)

            parameter = next(train_model.parameters())
            optimizer = torch.optim.AdamW(train_model.parameters(), lr=0.1)
            train_output.loss.backward()
            optimizer.step()
            self.assertIn(parameter, optimizer.state)

            config.task = "infer"
            infer_model = Pi0Model(config, mode="infer")
            infer_model.adapter.pipeline.tokenizer._processor = SimpleNamespace(
                encode=lambda _text, add_bos=False: [1, 2]
            )
            image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            observation = ObservationBatch(
                images={"base": image, "left_wrist": image, "right_wrist": image},
                state=np.zeros((1, 2), dtype=np.float32),
                prompt=["pick"],
            )
            global_rng = torch.random.get_rng_state().clone()
            output = infer_model.predict_action_chunk(
                observation,
                options=SimpleNamespace(seed=7, noise=None, num_steps=2),
            )
            self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
            self.assertEqual(output.actions.shape, (1, 2, 2))
            self.assertEqual(output.actions.dtype, np.float32)

            with self.assertRaisesRegex(ValueError, "Expected state_dim=2"):
                infer_model.predict_action_chunk(
                    ObservationBatch(
                        images={
                            "base": image,
                            "left_wrist": image,
                            "right_wrist": image,
                        },
                        state=np.zeros((1, 3), dtype=np.float32),
                        prompt=["pick"],
                    ),
                    options=SimpleNamespace(seed=7, noise=None, num_steps=2),
                )
            with self.assertRaisesRegex(KeyError, "missing cameras"):
                infer_model.predict_action_chunk(
                    ObservationBatch(
                        images={"base": image, "left_wrist": image},
                        state=np.zeros((1, 2), dtype=np.float32),
                        prompt=["pick"],
                    ),
                    options=SimpleNamespace(seed=7, noise=None, num_steps=2),
                )

            with self.assertRaisesRegex(ValueError, "noise must have shape"):
                infer_model.predict_action_chunk(
                    observation,
                    options=SimpleNamespace(
                        seed=7, noise=np.zeros((1, 2, 2), dtype=np.float32), num_steps=2
                    ),
                )

    def test_local_pipeline_uses_pi0_prompt_and_round_trips_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "norm_stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "state": {"mean": [1.0, 2.0], "std": [2.0, 4.0]},
                        "actions": {"mean": [10.0, 20.0], "std": [2.0, 5.0]},
                    }
                ),
                encoding="utf-8",
            )
            tokenizer = root / "paligemma_tokenizer.model"
            tokenizer.write_bytes(b"test")
            config = _wrap(
                {
                    "task": "train",
                    "model": {},
                    "data": {
                        "norm_stats_path": str(stats),
                        "tokenizer_path": str(tokenizer),
                        "field_mapping": {
                            "images": {
                                "base": "base_image",
                                "left_wrist": "left_image",
                                "right_wrist": "right_image",
                            },
                            "state": "robot_state",
                            "action": "robot_action",
                            "language": "task",
                        },
                        "transforms": {
                            "action": {"type": "delta", "delta_mask": [True, False]}
                        },
                    },
                }
            )
            model_config = SimpleNamespace(
                state_dim=2,
                action_dim=2,
                action_horizon=2,
                model_action_dim=4,
                max_token_len=5,
                camera_mapping={
                    "base": "base_0_rgb",
                    "left_wrist": "left_wrist_0_rgb",
                    "right_wrist": "right_wrist_0_rgb",
                },
            )
            pipeline = Pi0Pipeline(config, model_config)

            calls = []

            def encode(text, add_bos=False):
                calls.append((text, add_bos))
                return [1, 2]

            pipeline.tokenizer._processor = SimpleNamespace(encode=encode)
            raw_actions = np.asarray([[3.0, 7.0], [5.0, 9.0]], dtype=np.float32)
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            transformed = pipeline.transform_frame(
                {
                    "base_image": image,
                    "left_image": image,
                    "right_image": image,
                    "robot_state": np.asarray([1.0, 2.0], dtype=np.float32),
                    "robot_action": raw_actions,
                    "task": "pick_cube",
                }
            )
            self.assertEqual(calls, [("pick cube", True), ("\n", False)])
            self.assertNotIn("State:", calls[0][0])
            decoded = pipeline.decode_actions(
                transformed["actions"][None],
                np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
            np.testing.assert_allclose(decoded[0], raw_actions, atol=1e-5)
            self.assertEqual(transformed["state"].shape, (4,))
            self.assertEqual(transformed["actions"].shape, (2, 4))

    def test_artifact_export_uses_bare_fp32_operator_keys(self) -> None:
        module = Pi0Model.__new__(Pi0Model)
        torch.nn.Module.__init__(module)
        state = module.export_backend_state_dict(
            {
                "_fsdp_wrapped_module.core_model.weight": torch.ones(1),
                "other.weight": torch.ones(1),
            }
        )
        self.assertEqual(set(state), {"weight"})
        self.assertFalse(module.save_nexus_weights())

    def test_legacy_bc_token_limit_is_restored_for_inference_and_checked_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fake_core(root)
            weights, stats, tokenizer = root / "base.pt", root / "stats.json", root / "tokenizer"
            torch.save({"weight": torch.tensor(1.0)}, weights)
            stats.write_text(json.dumps(dict.fromkeys(
                ("state", "actions"), {"mean": [0, 0], "std": [1, 1]}
            )))
            tokenizer.write_bytes(b"test tokenizer")
            train = _model_config(root, weights, stats, tokenizer)
            train.schema_version = 1
            train.exp = Config(name="legacy", output_dir=str(root))
            train.train.max_steps = 1

            # Reproduce the old loader: the model used 48, but 12 was saved and fingerprinted.
            model = Pi0Model(train, mode="train")
            train.model.max_token_len = 12
            artifact = ArtifactManager(root / "legacy").save(
                model=model, runtime=LocalRuntime(train),
                optimizer=torch.optim.AdamW(model.parameters()), scheduler=None,
                config=train, step=1, metrics={},
            )
            self.assertEqual(load_manifest(artifact)["model"]["max_token_len"], 48)
            self.assertEqual(load_config(artifact / "config_resolved.yaml").model.max_token_len, 12)
            infer = _wrap({
                "schema_version": 1,
                "task": "infer",
                "model": {
                    "target": train.model.target, "artifact": str(artifact), "precision": "fp32",
                },
                "runtime": {"device": "cpu"},
            })
            for override in (None, 48):
                with self.subTest(override=override):
                    if override is not None:
                        infer.model.max_token_len = override
                    policy = InferencePolicy(infer)
                    self.assertEqual(policy.model.adapter.pipeline.tokenizer.max_length, 48)
            infer.model.max_token_len = 12
            with self.assertRaisesRegex(ValueError, "Cannot override trained model.max_token_len"):
                InferencePolicy(infer)

            for selector in (str(artifact), "auto"):
                train.train.resume_from = selector
                with (
                    self.subTest(resume_from=selector),
                    mock.patch("dobot_nexus.models.pi0.modeling_pi0.load_core_source") as load_core,
                    self.assertRaisesRegex(
                        ValueError, r"max_token_len .*checkpoint=48, configured=12"
                    ),
                ):
                    BCTrainer(train, runtime=LocalRuntime(train)).run()
                load_core.assert_not_called()

    def test_artifact_restores_weights_stats_and_tokenizer_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fake_core(root)
            checkpoint = root / "weights.pt"
            torch.save({"weight": torch.tensor(2.0)}, checkpoint)
            stats = root / "norm_stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                        "actions": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    }
                ),
                encoding="utf-8",
            )
            tokenizer = root / "paligemma_tokenizer.model"
            tokenizer.write_bytes(b"portable-tokenizer")
            config = _model_config(root, checkpoint, stats, tokenizer)
            config.schema_version = 1
            config.algorithm = "bc"
            config.exp = Config({"name": "pi0-test", "output_dir": str(root)})
            config.train.resume_limit = 0
            config.train.grad_accum_steps = 1
            model = Pi0Model(config, mode="train")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            artifact = ArtifactManager(root / "run").save(
                model=model,
                runtime=LocalRuntime(config),
                optimizer=optimizer,
                scheduler=None,
                config=config,
                step=1,
                metrics={"loss": 0.0},
            )

            manifest = load_manifest(artifact)
            self.assertNotIn("nexus_weights", manifest["files"])
            copied_stats = artifact / manifest["files"]["norm_stats"]
            tokenizer_relative = manifest["files"]["model_assets"]["data.tokenizer_path"]
            copied_tokenizer = artifact / tokenizer_relative
            self.assertTrue(copied_stats.is_file())
            self.assertEqual(copied_tokenizer.read_bytes(), b"portable-tokenizer")

            infer_config = _wrap(
                {
                    "schema_version": 1,
                    "task": "infer",
                    "exp": {"name": "pi0-infer", "output_dir": str(root)},
                    "model": {
                        "name": "pi0",
                        "target": "dobot_nexus.models.pi0.modeling_pi0:Pi0Model",
                        "artifact": str(artifact),
                        "checkpoint": None,
                        "precision": "fp32",
                        "state_dim": 2,
                        "action_dim": 2,
                        "action_horizon": 2,
                        "image_size": [224, 224],
                    },
                    "runtime": {"device": "cpu"},
                    "inference": {"num_steps": 2, "execute_steps": 2, "seed": 3},
                }
            )
            policy = InferencePolicy(infer_config)
            policy.model.adapter.pipeline.tokenizer._processor = SimpleNamespace(
                encode=lambda _text, add_bos=False: [1, 2]
            )
            self.assertEqual(
                Path(policy.config.data.norm_stats_path), copied_stats.resolve()
            )
            self.assertEqual(
                Path(policy.config.data.tokenizer_path), copied_tokenizer.resolve()
            )
            image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            output = policy.predict_chunk(
                ObservationBatch(
                    images={"base": image, "left_wrist": image, "right_wrist": image},
                    state=np.zeros((1, 2), dtype=np.float32),
                    prompt=["pick"],
                ),
                SamplingOptions(seed=3, num_steps=2),
            )
            self.assertEqual(output.actions.shape, (1, 2, 2))
            self.assertEqual(output.actions.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
