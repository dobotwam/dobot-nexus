from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.config import Config, load_config
from dobot_nexus.core.contracts import ObservationBatch, PolicyOutput, SamplingOptions
from dobot_nexus.models.pi05.configuration_pi05 import Pi05Config
from dobot_nexus.models.pi05.data import Pi05Pipeline
from dobot_nexus.models.pi05.modeling_pi05 import Pi05Model
from dobot_nexus.models.pi05.torch_core.source import load_core_source
from dobot_nexus.models.pi05.training.checkpoint import normalize_state_dict

ROOT = Path(__file__).resolve().parents[3]


def _wrap(value):
    if isinstance(value, dict):
        return Config({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


class _FakeCore(torch.nn.Module):
    """Small model for lifecycle contracts using the real config and Observation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def compute_loss(self, observation, actions, train=True, rng=None):
        return ((actions * self.weight) ** 2).mean(dim=-1)

    def sample_actions(self, observation, num_steps=10, noise=None, rng=None):
        shape = (observation.state.shape[0], self.action_horizon, self.action_dim)
        return torch.ones(shape, device=self.weight.device) * self.weight


def _create_fake_core(config):
    return _FakeCore(config)


class Pi05DirectCoreTest(unittest.TestCase):
    def test_fp32_inference_parameters_support_bf16_operator_inputs(self) -> None:
        class ProjectionCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(32, 32, dtype=torch.float32)

            def sample_actions(self, observation, **kwargs):
                return self.proj(torch.ones((1, 2, 32), dtype=torch.bfloat16))

        model = Pi05Model.__new__(Pi05Model)
        torch.nn.Module.__init__(model)
        model.mode = "infer"
        model.core_model = ProjectionCore()
        model.torch_engine = None
        config = load_config(ROOT / "examples/configs/infer/pi05.yaml", [
            "model.artifact=null", "model.checkpoint=unused",
            "model.inference_parameter_precision=fp32", "runtime.compile=false",
            "runtime.cuda_graph=false",
            "model.state_dim=2", "model.action_dim=2", "model.action_horizon=2",
        ], expected_task="infer")
        model.pi05_config = Pi05Config.from_config(config, mode="infer")
        model.adapter = SimpleNamespace(
            encode_observation=lambda obs: obs,
            move_encoded=lambda obs, device: SimpleNamespace(
                observation=obs, raw_state=obs.state, model_kwargs={},
            ),
            decode_actions=lambda actions, aux: PolicyOutput(actions[:, :, :2]),
        )
        observation = ObservationBatch.from_value({
            "images": {"base": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": [0.0, 0.0], "prompt": "pick",
        })
        output = model.predict_action_chunk(observation, SamplingOptions(num_steps=1))
        self.assertEqual(output.actions.shape, (1, 2, 2))
        self.assertTrue(np.isfinite(output.actions).all())
        self.assertEqual(model.core_model.proj.weight.dtype, torch.float32)

    def test_inference_precision_combinations_are_validated_before_loading(self) -> None:
        config = load_config(ROOT / "examples/configs/infer/pi05.yaml", [
            "model.artifact=null", "model.checkpoint=unused",
            "model.state_dim=2", "model.action_dim=2", "model.action_horizon=2",
        ], expected_task="infer")
        # Training-only precision must not select the inference parameter dtype.
        config.model.parameter_precision = "fp32"
        for compute, parameters in (("bf16", "bf16"), ("bf16", "fp32"), ("fp32", "fp32")):
            with self.subTest(compute=compute, parameters=parameters):
                config.model.precision = compute
                config.model.inference_parameter_precision = parameters
                resolved = Pi05Config.from_config(config, mode="infer")
                self.assertEqual(resolved.precision, compute)
                self.assertEqual(resolved.parameter_precision, parameters)
        for compute in ("bf16", "fp32"):
            config.model.precision = compute
            config.model.pop("inference_parameter_precision", None)
            self.assertEqual(Pi05Config.from_config(config, mode="infer").parameter_precision, compute)
        for parameters in ("bf16", "float16"):
            config.model.precision = "float32"
            config.model.inference_parameter_precision = parameters
            with self.subTest(parameters=parameters), self.assertRaisesRegex(
                ValueError, "FP32 inference compute requires"
            ):
                Pi05Config.from_config(config, mode="infer")

    def test_user_configs_hide_operator_backend_details(self) -> None:
        train = load_config(
            ROOT / "examples/configs/train/pi05.yaml", expected_task="train"
        )
        infer = load_config(
            ROOT / "examples/configs/infer/pi05.yaml", expected_task="infer"
        )
        self.assertNotIn("dependencies", train)
        self.assertNotIn("backend", train.model)
        self.assertNotIn("dependencies", infer)
        self.assertNotIn("backend", infer.model)
        self.assertEqual(
            set(train.data.field_mapping.images),
            {"base", "left_wrist", "right_wrist"},
        )
        self.assertEqual(train.model.precision, "bf16")
        self.assertEqual(train.model.parameter_precision, "fp32")
        self.assertEqual(train.runtime.fsdp.sharding_strategy, "full_shard")
        self.assertTrue(train.runtime.fsdp.use_orig_params)
        self.assertEqual(
            train.runtime.fsdp.transformer_layer_cls_to_wrap,
            ["Block", "Encoder1DBlock"],
        )

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
                "model.action_in_proj.weight": torch.zeros(1),
            }
        )
        self.assertEqual(set(state), {"img.stem.weight", "action_in_proj.weight"})

    def test_legacy_external_core_config_requires_explicit_migration(self) -> None:
        for mode in ("train", "infer"):
            with self.subTest(mode=mode):
                config = load_config(
                    ROOT / f"examples/configs/{mode}/pi05.yaml", expected_task=mode
                )
                config["dependencies"] = Config({"rlinf_path": "/old/rlinf"})
                with self.assertRaisesRegex(ValueError, "dependencies.rlinf_path.*Remove"):
                    Pi05Model(config, mode=mode)

    def test_obsolete_jax_export_path_is_discarded_from_resolved_config(self) -> None:
        config = load_config(ROOT / "examples/configs/train/pi05.yaml", expected_task="train")
        config.model.jax_export_base_checkpoint = "/missing/old-base"
        parsed = Pi05Config.from_config(config, mode="train")
        self.assertNotIn("jax_export_base_checkpoint", config.model)
        self.assertFalse(hasattr(parsed, "jax_export_base_checkpoint"))

    def test_production_rejects_reference_backends_before_loading_weights(self) -> None:
        for mode in ("train", "infer"):
            for backend in ("jax", "tensorflow"):
                with self.subTest(mode=mode, backend=backend):
                    config = load_config(ROOT / f"examples/configs/{mode}/pi05.yaml")
                    config.model.inference_backend = backend
                    with self.assertRaisesRegex(ValueError, "reference_jax"):
                        Pi05Model(config, mode=mode)

    def test_historical_reference_paths_are_ignored_without_changing_resume_identity(self) -> None:
        from dobot_nexus.core.artifacts import resume_fingerprint

        for mode in ("train", "infer"):
            with self.subTest(mode=mode):
                config = load_config(ROOT / f"examples/configs/{mode}/pi05.yaml")
                config.model.pop("artifact", None)
                config.model.checkpoint = "/unused/torch/weights.pt"
                config.model.update(state_dim=14, action_dim=14, action_horizon=50)
                Pi05Config.from_config(config, mode=mode)
                config.model.openpi_root = "/missing/reference/checkout"
                config.model.jax_checkpoint = "/missing/reference/weights.safetensors"
                before = resume_fingerprint(config) if mode == "train" else None
                parsed = Pi05Config.from_config(config, mode=mode)
                self.assertEqual(parsed.inference_backend, "torch")
                self.assertFalse(hasattr(parsed, "openpi_root"))
                self.assertFalse(hasattr(parsed, "jax_checkpoint"))
                if mode == "train":
                    self.assertEqual(resume_fingerprint(config), before)
                else:
                    self.assertNotIn("openpi_root", config.model)
                    self.assertNotIn("jax_checkpoint", config.model)

    def test_default_core_is_model_owned_and_ignores_rlinf_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DOBOT_PI05_CORE_PATH": "/missing/rlinf",
                "RLINF_PATH": "/also/missing",
            },
        ):
            source = load_core_source()
        self.assertTrue(source.embedded)
        self.assertEqual(source.revision, "89b0cd5fce52")
        self.assertEqual(source.core_dir.name, "torch_core")
        self.assertEqual(source.package_name, "dobot_nexus.models.pi05.torch_core")
        self.assertTrue((source.core_dir / "LICENSE").is_file())
        self.assertTrue((source.core_dir / "NOTICE.md").is_file())
        self.assertNotIn("RLinf", str(source.core_dir))

    @mock.patch("dobot_nexus.models.pi05.torch_core.pi0_config.Pi0Config.create",
                new=_create_fake_core)
    def test_pi05_model_runs_direct_core_train_and_infer_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            config = _wrap(
                {
                    "model": {
                        "checkpoint": str(checkpoint),
                        "precision": "fp32",
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
            train_model = Pi05Model(config, mode="train")
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
                "tokenized_prompt": torch.zeros((1, 200), dtype=torch.long),
                "tokenized_prompt_mask": torch.ones((1, 200), dtype=torch.bool),
                "actions": torch.ones((1, 2, 32)),
            }
            encoded_batch = train_model.adapter.encode_train_batch(
                batch, torch.device("cpu")
            )
            train_output = train_model.training_step(encoded_batch)
            self.assertAlmostEqual(float(train_output.loss.detach()), 4.0)

            infer_model = Pi05Model(config, mode="infer")
            infer_model.adapter.pipeline.tokenizer._processor = SimpleNamespace(
                encode=lambda _text, add_bos: [1, 2]
            )
            image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            observation = ObservationBatch(
                images={"base": image, "left_wrist": image, "right_wrist": image},
                state=np.zeros((1, 2), dtype=np.float32),
                prompt=["pick"],
            )
            output = infer_model.predict_action_chunk(
                observation,
                options=SimpleNamespace(seed=1, noise=None, num_steps=2),
            )
            self.assertEqual(output.actions.shape, (1, 2, 2))
            self.assertGreater(output.timing["model_ms"], 0)
            self.assertGreaterEqual(output.timing["total_ms"], output.timing["model_ms"])
            acceleration = infer_model.torch_engine
            infer_model.to("cpu")
            self.assertIsNone(infer_model.torch_engine)
            self.assertEqual(acceleration.metadata["cuda_graph_entries"], 0)
            infer_model.bfloat16()
            with mock.patch.object(infer_model.core_model, "sample_actions",
                                   wraps=infer_model.core_model.sample_actions) as sample:
                infer_model.predict_action_chunk(observation, options=SimpleNamespace(
                    seed=1, noise=np.full((2, 32), 0.123456, dtype=np.float32), num_steps=2,
                ))
            noise = sample.call_args.kwargs["noise"]
            self.assertEqual(noise.dtype, torch.float32)
            self.assertEqual(tuple(noise.shape), (1, 2, 32))
            self.assertAlmostEqual(noise[0, 0, 0].item(), 0.123456, places=7)
            self.assertIsNotNone(infer_model.torch_engine)
            infer_model.load_state_dict(infer_model.state_dict())
            self.assertIsNone(infer_model.torch_engine)
            infer_model.predict_action_chunk(observation, options=SimpleNamespace(
                seed=1, noise=None, num_steps=2,
            ))
            self.assertIsNotNone(infer_model.torch_engine)
            infer_model.train()
            self.assertIsNone(infer_model.torch_engine)

    @mock.patch("dobot_nexus.models.pi05.torch_core.pi0_config.Pi0Config.create",
                new=_create_fake_core)
    def test_training_keeps_fp32_master_weights_and_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.pt"
            torch.save({"weight": torch.tensor(2.0, dtype=torch.bfloat16)}, checkpoint)
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
            config = _wrap(
                {
                    "model": {
                        "checkpoint": str(checkpoint),
                        "precision": "bf16",
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
                    "runtime": {"precision": "bf16"},
                    "train": {"gradient_checkpointing": False},
                }
            )
            model = Pi05Model(config, mode="train")
            parameter = next(model.core_model.parameters())
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(model.core_model.config.dtype, "bfloat16")
            self.assertEqual(model.pi05_config.torch_dtype, torch.bfloat16)
            self.assertEqual(model.pi05_config.parameter_torch_dtype, torch.float32)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            (parameter.square()).backward()
            optimizer.step()
            state = optimizer.state[parameter]
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)

            infer_model = Pi05Model(config, mode="infer")
            self.assertEqual(next(infer_model.core_model.parameters()).dtype, torch.bfloat16)

    def test_training_rejects_non_fp32_master_weights(self) -> None:
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
            Pi05Config.from_config(config, mode="train")

    def test_training_rejects_mismatched_compute_precision(self) -> None:
        config = _wrap(
            {
                "model": {
                    "checkpoint": "unused.pt",
                    "precision": "bf16",
                    "parameter_precision": "fp32",
                    "state_dim": 2,
                    "action_dim": 2,
                    "action_horizon": 2,
                    "image_size": [224, 224],
                },
                "runtime": {"precision": "fp32"},
            }
        )
        with self.assertRaisesRegex(ValueError, "must select the same compute dtype"):
            Pi05Config.from_config(config, mode="train")


    def test_local_pipeline_round_trips_delta_actions(self) -> None:
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
            config = Config(
                {
                    "model": Config({}),
                    "data": Config(
                        {
                            "norm_stats_path": str(stats),
                            "tokenizer_path": str(tokenizer),
                            "field_mapping": Config(
                                {
                                    "images": Config(
                                        {
                                            "base": "base_image",
                                            "left_wrist": "left_image",
                                            "right_wrist": "right_image",
                                        }
                                    ),
                                    "state": "robot_state",
                                    "action": "robot_action",
                                    "language": "task",
                                }
                            ),
                            "transforms": Config(
                                {
                                    "action": Config(
                                        {"type": "delta", "delta_mask": [True, False]}
                                    )
                                }
                            ),
                        }
                    ),
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
            pipeline = Pi05Pipeline(config, model_config)
            pipeline.tokenizer._processor = SimpleNamespace(
                encode=lambda _text, add_bos: [1, 2]
            )
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
            decoded = pipeline.decode_actions(
                transformed["actions"][None],
                np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
            np.testing.assert_allclose(decoded[0], raw_actions, atol=1e-5)
            self.assertEqual(transformed["state"].shape, (4,))
            self.assertEqual(transformed["actions"].shape, (2, 4))

            observation = ObservationBatch(
                images={
                    "base": image[None],
                    "left_wrist": image[None],
                    "right_wrist": image[None],
                },
                state=np.asarray([[1.0, 2.0]], dtype=np.float32),
                prompt=["pick cube"],
            ).validate()
            encoded, raw_state = pipeline.transform_runtime(observation)
            self.assertEqual(tuple(encoded["state"].shape), (1, 4))
            np.testing.assert_allclose(raw_state, [[1.0, 2.0]])


class RecipeTest(unittest.TestCase):
    def test_global_batch_is_checked_for_actual_world_size(self):
        from dobot_nexus.models.pi05.configuration_pi05 import validate_training_recipe

        for name in ("pi05_usb_bc", "pi05_usb_recap"):
            config = load_config(ROOT / f"examples/recipes/{name}.yaml")
            with self.subTest(recipe=name), mock.patch.dict(os.environ, WORLD_SIZE="7"):
                validate_training_recipe(config)
                self.assertEqual(config.data.global_batch_size, 672)
                self.assertEqual(config.data.batch_size, 96)
                self.assertEqual(config.train.grad_accum_steps, 1)
            with mock.patch.dict(os.environ, WORLD_SIZE="2"):
                with self.assertRaisesRegex(ValueError, "192"):
                    validate_training_recipe(config)
                # The documented two-rank memory-saving launch keeps the same batch.
                config.data.batch_size = 4
                config.train.grad_accum_steps = 84
                validate_training_recipe(config)


class Pi05ArtifactTest(unittest.TestCase):
    def test_pi05_artifact_uses_bare_operator_keys(self):
        module = Pi05Model.__new__(Pi05Model)
        torch.nn.Module.__init__(module)
        exported = module.export_backend_state_dict(
            {
                "core_model.llm.weight": torch.ones(1),
                "wrapper_head.weight": torch.zeros(1),
            }
        )
        self.assertEqual(set(exported), {"llm.weight"})
        self.assertFalse(module.save_nexus_weights())
        with self.assertRaisesRegex(RuntimeError, "FP32 training weights"):
            module.export_backend_state_dict(
                {"core_model.llm.weight": torch.ones(1, dtype=torch.bfloat16)}
            )

    def test_pi05_and_recap_need_no_base_checkpoint_for_deployment(self):
        from dobot_nexus.models.pi05.modeling_recap import Pi05ReCapModel

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            source = artifact / "deployment/pytorch/full_weights.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"weights")
            for model_type in (Pi05Model, Pi05ReCapModel):
                with self.subTest(model=model_type.__name__):
                    module = model_type.__new__(model_type)
                    torch.nn.Module.__init__(module)
                    produced = module.export_deployment_artifacts(
                        source_weights=source, artifact_dir=artifact
                    )
                    self.assertEqual(produced, {})
                    self.assertFalse((artifact / "deployment/openpi_jax").exists())
                    self.assertEqual(source.read_bytes(), b"weights")


if __name__ == "__main__":
    unittest.main()
