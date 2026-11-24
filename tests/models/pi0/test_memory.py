"""FrameSample contracts for the Pi0 operator, data pipeline and deployment Artifact."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.artifacts import ArtifactManager, load_manifest
from dobot_nexus.core.config import clone_config
from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions
from dobot_nexus.core.import_utils import build_model
from dobot_nexus.inference import InferencePolicy
from dobot_nexus.models.pi0.configuration_pi0 import Pi0Config
from dobot_nexus.models.pi0.data.pipeline import _Tokenizer
from dobot_nexus.models.pi0.data_adapter_pi0 import Pi0DataAdapter
from dobot_nexus.models.pi0.memory.checkpoint import is_memory_key, load_memory_checkpoint
from dobot_nexus.models.pi0.memory.configuration import MemoryConfig, resolve_config
from dobot_nexus.models.pi0.memory.data import MemoryPipeline
from dobot_nexus.models.pi0.memory.feature_store import FrameFeatureStore, content_sha256
from dobot_nexus.models.pi0.memory.infer import main as infer_memory
from dobot_nexus.models.pi0.memory.infer import open_memory_session
from dobot_nexus.models.pi0.memory.modeling_memory import Pi0MemoryModel
from dobot_nexus.models.pi0.memory.prepare import prepare_features
from dobot_nexus.models.pi0.memory.torch import MemoryAttention, install_memory
from dobot_nexus.models.pi0.memory.torch_vision import TorchFrameTokenEncoder
from dobot_nexus.models.pi0.training.source import load_core_source
from dobot_nexus.models.pi0.training.vendor.rlinf_core import siglip
from dobot_nexus.training.runtime import LocalRuntime
from tests.algorithms.recap.fakes import FakeTokenizer
from tests.models.pi0.fixtures import ROOT, infer_config, small_backbone, train_config

VARIANT = dict(width=8, depth=2, num_heads=2, mlp_dim=12, patch_size=(14, 14))


def vision_archive():
    """Small standalone scanned checkpoint in the published vision.npz schema."""
    shapes = {
        "embedding/kernel": (14, 14, 3, 8),
        "embedding/bias": (8,),
        "pos_embedding": (1, 256, 8),
        "head/kernel": (8, 2048),
        "head/bias": (2048,),
        "Transformer/encoder_norm/scale": (8,),
        "Transformer/encoder_norm/bias": (8,),
    }
    prefix = "Transformer/encoderblock"
    for norm in ("LayerNorm_0", "LayerNorm_1"):
        for leaf in ("scale", "bias"):
            shapes[f"{prefix}/{norm}/{leaf}"] = (2, 8)
    for dense, inputs, outputs in (("Dense_0", 8, 12), ("Dense_1", 12, 8)):
        shapes[f"{prefix}/MlpBlock_0/{dense}/kernel"] = (2, inputs, outputs)
        shapes[f"{prefix}/MlpBlock_0/{dense}/bias"] = (2, outputs)
    attention = f"{prefix}/MultiHeadDotProductAttention_0"
    for projection in ("query", "key", "value"):
        shapes[f"{attention}/{projection}/kernel"] = (2, 8, 2, 4)
        shapes[f"{attention}/{projection}/bias"] = (2, 2, 4)
    shapes[f"{attention}/out/kernel"] = (2, 2, 4, 8)
    shapes[f"{attention}/out/bias"] = (2, 8)
    rng = np.random.default_rng(10)
    return {
        key: (
            np.ones(shape, np.float32)
            if key.endswith("/scale")
            else rng.normal(0, 0.1, shape).astype(np.float32)
        )
        for key, shape in shapes.items()
    }


class FrameSource(list):
    dataset_root = Path("test-data")
    detected_format_version = "v2.1"
    resolved_fps = 10


class Pi0MemoryTest(unittest.TestCase):
    def test_import_frozen_vision_is_eager_in_a_fresh_interpreter(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from unittest.mock import patch\n"
                "with patch('torch.compile', side_effect=AssertionError('compiled')) as compile:\n"
                "    import dobot_nexus.models.pi0.memory.torch_vision\n"
                "    compile.assert_not_called()",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_vision_decodes_the_verified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision.npz"
            archive = vision_archive()
            np.savez(path, **archive)
            digest = content_sha256(path)
            read = Path.read_bytes

            def replace_after_read(asset):
                raw = read(asset)
                asset.write_bytes(b"changed")
                return raw

            with (
                mock.patch.object(
                    Path, "read_bytes", autospec=True, side_effect=replace_after_read
                ) as load,
                mock.patch.object(siglip, "_decode_variant", return_value=VARIANT),
            ):
                encoder = TorchFrameTokenEncoder(path, expected_sha256=digest)
            load.assert_called_once_with(path)
            np.testing.assert_array_equal(
                encoder.model.stem.bias.numpy(), archive["embedding/bias"]
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                TorchFrameTokenEncoder(path, expected_sha256=digest)

    def test_install_preserves_pi0_parameters_on_meta(self):
        source = load_core_source()
        with torch.device("meta"):
            core = source.Pi0Config(pi05=False).create()
            before = dict(core.named_parameters())
            self.assertFalse(any(is_memory_key(name) for name in core.state_dict()))
            install_memory(core, torch.bfloat16)
        self.assertFalse(core.pi05)
        self.assertIn("action_time_mlp_in.weight", before)
        self.assertIn("state_proj.weight", before)
        self.assertEqual(set(before), {k for k in core.state_dict() if not is_memory_key(k)})
        params = dict(core.named_parameters())
        for name, parameter in before.items():
            self.assertIs(parameter, params[name])
        for block in core.llm.layers:
            self.assertIs(block.memory_modulation.attention.rope, block.attn.rope)
            self.assertFalse(block.pre_ffw_norms[-1].adaptive)
        self.assertTrue(all(parameter.is_meta for parameter in params.values()))
        with mock.patch.object(core.llm.configs[-1], "width", 512):
            with self.assertRaisesRegex(ValueError, "1024"):
                install_memory(core, torch.bfloat16)

    def test_load_from_base_or_strict_trained_checkpoint(self):
        def core():
            with torch.device("meta"):
                module = torch.nn.Module()
                module.base = torch.nn.Linear(2, 2)
                module.memory_encoder = torch.nn.Linear(2, 2)
                module.llm = torch.nn.Module()
                block = torch.nn.Module()
                block.memory_modulation = torch.nn.Linear(2, 2)
                module.llm.layers = torch.nn.ModuleList([block])
            return module

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            base = {"base.weight": torch.eye(2), "base.bias": torch.zeros(2)}
            torch.save(base, path)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_memory_checkpoint(core(), path)
            initialized = core()
            load_memory_checkpoint(initialized, path, allow_initialization=True)
            trained = initialized.state_dict()
            for key, value in base.items():
                torch.testing.assert_close(trained[key], value)
            self.assertTrue(all(value.isfinite().all() for value in trained.values()))
            torch.save(trained, path)
            restored = core()
            load_memory_checkpoint(restored, path)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, trained[key], rtol=0, atol=0)
            for partial in (
                {**base, "memory_encoder.bias": trained["memory_encoder.bias"]},
                {**trained, "base.bias": torch.zeros(1)},
            ):
                torch.save(partial, path)
                with self.assertRaisesRegex(ValueError, "Incomplete|shape mismatch"):
                    load_memory_checkpoint(core(), path, allow_initialization=True)

    def test_deployment_bundles_vision_without_policy_conversion_or_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "vision.npz"
            np.savez(source, kernel=np.arange(6, dtype=np.float32))
            digest = content_sha256(source)
            model = Pi0MemoryModel.__new__(Pi0MemoryModel)
            torch.nn.Module.__init__(model)
            model.memory_config = MemoryConfig(vision_checkpoint=str(source), vision_sha256=digest)
            artifact = root / "artifact"
            produced = model.export_deployment_artifacts(
                source_weights=root / "unused-policy-weights.pt", artifact_dir=artifact
            )
            self.assertEqual(produced, {"memory_vision": "deployment/memory/vision.npz"})
            bundled = artifact / produced["memory_vision"]
            self.assertEqual(content_sha256(bundled), digest)
            self.assertFalse((artifact / "deployment/openpi_jax").exists())

            # A changed encoder must not overwrite the deployment's validated copy.
            np.savez(source, kernel=np.ones(6, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "encoder changed"):
                model.export_deployment_artifacts(source_weights=None, artifact_dir=artifact)
            self.assertEqual(content_sha256(bundled), digest)
            source.unlink()
            with np.load(bundled) as loaded:
                np.testing.assert_array_equal(loaded["kernel"], np.arange(6, dtype=np.float32))

    def test_history_kv_cache_preserves_attention_and_refreshes_per_request(self):
        gemma = importlib.import_module(load_core_source().package_name + ".gemma")

        torch.manual_seed(42)
        attention = MemoryAttention(gemma._apply_rope).eval()
        mask = torch.tensor([[True, True, False, True, False]])
        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16):
                query = torch.randn(1, 3, 1024).to(dtype)
                outputs = []
                for _ in range(2):
                    history = torch.randn(1, 5, 1024).to(dtype)
                    cached = attention.prepare_memory(history)
                    for delta in (0, 1):
                        expected = attention(query + delta, history, mask)
                        actual = attention(query + delta, history, mask, cached_kv=cached)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    outputs.append(actual)
                self.assertFalse(torch.equal(*outputs))

    def test_extension_tensor_transfer_preserves_containers_and_legacy_observations(self):
        observation = load_core_source().Observation(
            images={}, image_masks={}, state=torch.zeros(2)
        )
        encoded = SimpleNamespace(observation=observation, raw_state=np.zeros((1, 2)))
        adapter = Pi0DataAdapter.__new__(Pi0DataAdapter)
        device = torch.device("meta")
        self.assertEqual(adapter.move_encoded(encoded, device).model_kwargs, {})
        encoded.model_kwargs = {"nested": ({"values": [torch.ones(1), 7]},), "flag": True}
        moved = adapter.move_encoded(encoded, device)
        self.assertIs(moved.raw_state, encoded.raw_state)
        self.assertTrue(moved.observation.state.is_meta)
        self.assertIsInstance(moved.model_kwargs["nested"], tuple)
        values = moved.model_kwargs["nested"][0]["values"]
        self.assertIsInstance(values, list)
        self.assertTrue(values[0].is_meta)
        self.assertEqual(values[1], 7)
        self.assertIs(moved.model_kwargs["flag"], True)
        self.assertFalse(encoded.model_kwargs["nested"][0]["values"][0].is_meta)

    def test_sampler_builds_history_cache_once_per_layer_and_request(self):
        module = importlib.import_module(load_core_source().package_name + ".pi0")
        observation = SimpleNamespace(state=torch.zeros(1, 2))
        attentions = [
            SimpleNamespace(prepare_memory=mock.Mock(side_effect=lambda x: (x + 1, x))),
            SimpleNamespace(prepare_memory=mock.Mock(side_effect=lambda x: (2 * x, x))),
        ]
        layers = [
            SimpleNamespace(memory_modulation=SimpleNamespace(attention=a)) for a in attentions
        ]
        core = SimpleNamespace(
            training=False,
            action_horizon=2,
            action_dim=2,
            llm=SimpleNamespace(layers=layers),
            _encode_memory=lambda inputs: {"memory_tokens": inputs, "memory_mask": torch.ones(1)},
            build_prefix_cache=lambda obs: (None, None, None),
            run_suffix=mock.Mock(
                side_effect=lambda *args, memory_kv, **kw: memory_kv[0][0] + memory_kv[1][0]
            ),
            velocity_from_suffix=lambda value: value,
        )
        with mock.patch.object(
            module.model, "preprocess_observation", side_effect=lambda obs, **kw: obs
        ):
            for value in (1.0, 4.0):
                history = torch.full((1, 2, 2), value)
                actual = module.Pi0.sample_actions(
                    core,
                    observation,
                    noise=torch.zeros_like(history),
                    num_steps=5,
                    memory_inputs=history,
                    _cache_memory=True,
                )
                torch.testing.assert_close(actual, -(3 * history + 1))
            for attention in attentions:
                self.assertEqual(attention.prepare_memory.call_count, 2)
            self.assertEqual(core.run_suffix.call_count, 10)
            core.training = True
            with self.assertRaisesRegex(RuntimeError, "eval"):
                module.Pi0.sample_actions(
                    core, observation, memory_inputs=history, _cache_memory=True
                )

    def test_prepare_train_export_and_serve_dense_history(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            small_backbone(expert_width=1024) as source,
            mock.patch.object(siglip, "_decode_variant", return_value=VARIANT),
            mock.patch.object(_Tokenizer, "_get_processor", return_value=FakeTokenizer()),
        ):
            root = Path(directory)
            config = train_config(root, "memory", source)
            config.data.memory_features = str(root / "features")
            config.data.memory_dataset_id = "frames-v1"
            vision = root / "vision.npz"
            np.savez(vision, **vision_archive())
            config.model.memory.vision_checkpoint = str(vision)
            config.model.memory.budget = 32
            fields = config.data.field_mapping.images
            frames = FrameSource(
                {
                    "episode_index": 0,
                    "frame_index": i,
                    "task": task,
                    "observation.state": np.zeros(2, np.float32),
                    "action": np.ones((2, 2), np.float32),
                    **{k: np.full((32, 64, 3), i * 100, np.uint8) for k in fields.values()},
                }
                for i, task in enumerate(("pick", "pick", "place"))
            )
            with mock.patch("dobot_nexus.data.lerobot.source.build_source", return_value=frames):
                prepared = prepare_features(config, device="cpu", batch_size=2)
            self.assertEqual((prepared["episodes"], prepared["frames"]), (1, 3))
            _, spec = resolve_config(config, mode="train")
            columns = dict(
                image_field=fields.base, language_field=config.data.field_mapping.language
            )
            store = FrameFeatureStore(
                config.data.memory_features, spec, dataset_id="frames-v1", **columns
            )
            for name, value in columns.items():
                self.assertEqual(store.manifest["feature_spec"][name], value)
            for column in ("image_field", "language_field"):
                remapped = clone_config(config)
                if column == "image_field":
                    remapped.data.field_mapping.images.base = fields.left_wrist
                else:
                    remapped.data.field_mapping.language = "alternate_prompt"
                with (
                    self.subTest(column=column),
                    self.assertRaisesRegex(ValueError, "dataset fields"),
                ):
                    MemoryPipeline(remapped, Pi0Config.from_config(remapped, mode="train"), spec)
            with self.assertRaisesRegex(ValueError, "different dataset"):
                FrameFeatureStore(config.data.memory_features, spec, dataset_id="wrong", **columns)
            self.assertEqual([store.lookup(0, i).mask.sum() for i in range(3)], [16, 32, 16])
            snapshots = [store.lookup(0, i) for i in range(3)]
            with self.assertRaisesRegex(ValueError, "immutable"):
                prepare_features(config, device="cpu")
            model = Pi0MemoryModel(config, mode="train")
            with mock.patch(
                "dobot_nexus.models.pi0.data.pipeline.build_source", return_value=frames
            ):
                loader = model.build_train_dataloader(config)
            batch = model.adapter.encode_train_batch(next(iter(loader)), torch.device("cpu"))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loss = model.training_step(batch).loss
            self.assertTrue(loss.isfinite())
            loss.backward()
            modulation = model.core_model.llm.layers[0].memory_modulation.modulation.weight
            self.assertGreater(modulation.grad.abs().max().item(), 0)
            optimizer.step()
            Path(config.model.checkpoint).unlink()
            artifact = ArtifactManager(root / "run").save(
                model=model,
                runtime=LocalRuntime(config),
                optimizer=optimizer,
                scheduler=None,
                config=config,
                step=1,
                metrics={"loss": loss.item()},
            )
            manifest = load_manifest(artifact)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["model"]["memory"]["vision_sha256"], content_sha256(vision))
            bundled = artifact / manifest["files"]["memory_vision"]
            self.assertEqual(content_sha256(bundled), content_sha256(vision))
            shutil.rmtree(config.data.memory_features)
            for path in (vision, config.data.tokenizer_path, config.data.norm_stats_path):
                Path(path).unlink()

            pointer = artifact.parent / "latest_deployment.txt"
            infer = infer_config(pointer, "memory")

            def rotate_before_build(resolved, *, mode):
                pointer.write_text("another_step\n")
                return build_model(resolved, mode=mode)

            with mock.patch(
                "dobot_nexus.inference.policy.build_model", side_effect=rotate_before_build
            ):
                session = open_memory_session(infer)
            policy = session.policy
            self.assertEqual(Path(infer.model.artifact), pointer)
            self.assertEqual(Path(policy.config.model.artifact), artifact.resolve())
            self.assertEqual(session.options.execute_steps, 2)
            pointer.write_text(artifact.name + "\n")
            self.assertEqual(policy.model.pi0_config.max_token_len, 12)
            spec = policy.model.memory_config
            self.assertEqual(Path(spec.vision_checkpoint), bundled.resolve())
            restored = policy.model.core_model.llm.layers[0].memory_modulation
            torch.testing.assert_close(modulation, restored.modulation.weight, atol=0, rtol=0)
            encoder = session.encoder
            self.assertEqual(encoder.model.stem.weight.dtype, torch.float32)
            self.assertEqual(encoder.model.head.weight.dtype, torch.bfloat16)
            self.assertFalse(any(p.requires_grad for p in encoder.model.parameters()))
            for i, frame in enumerate(frames):
                obs = ObservationBatch(
                    images={camera: frame[key][None] for camera, key in fields.items()},
                    state=frame["observation.state"][None],
                    prompt=[frame["task"]],
                )
                prepared = session.prepare(obs, step_indices=[i])
                expected = snapshots[i]
                for name in ("features", "positions", "mask"):
                    np.testing.assert_array_equal(
                        getattr(prepared.memory, name)[0], getattr(expected, name)
                    )
                with mock.patch.object(
                    restored.attention,
                    "prepare_memory",
                    wraps=restored.attention.prepare_memory,
                ) as project:
                    actions = session.predict_chunk(
                        obs,
                        step_indices=[i],
                        options=SamplingOptions(num_steps=3, seed=42),
                    ).actions
                self.assertEqual(actions.shape, (1, 2, 2))
                project.assert_called_once()
                if i == 0:
                    observation_file, output_file = root / "observation.npz", root / "actions.npz"
                    np.savez(
                        observation_file,
                        state=obs.state,
                        prompt=obs.prompt,
                        **{f"image.{k}": v for k, v in obs.images.items()},
                    )
                    infer_memory(
                        [
                            "--config",
                            str(ROOT / "examples/configs/infer/pi0_memory.yaml"),
                            "--input",
                            str(observation_file),
                            "--output",
                            str(output_file),
                            f"model.artifact={pointer}",
                            "model.precision=fp32",
                            "runtime.device=cpu",
                            "inference.execute_steps=2",
                            "inference.seed=42",
                            "inference.num_steps=3",
                        ]
                    )
                    with np.load(output_file, allow_pickle=False) as output:
                        np.testing.assert_array_equal(output["actions"], actions)
            for key, value in (("max_token_len", 13), ("memory", {"budget": 16})):
                bad = clone_config(infer)
                bad.model[key] = value
                with self.assertRaisesRegex(ValueError, "override"):
                    InferencePolicy(bad)
            bundled.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "vision_sha256"):
                InferencePolicy(infer)
