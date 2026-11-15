"""Memory integration contracts for the model-owned Torch core."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.config import clone_config
from dobot_nexus.models.pi05.configuration_pi05 import Pi05Config
from dobot_nexus.models.pi05.memory.checkpoint import load_memory_checkpoint
from dobot_nexus.models.pi05.memory.data import MemoryDataAdapter
from dobot_nexus.models.pi05.memory.configuration import MemoryConfig
from dobot_nexus.models.pi05.memory.feature_store import content_sha256
from dobot_nexus.models.pi05.memory.modeling_memory import Pi05MemoryModel
from dobot_nexus.models.pi05.torch_core import gemma
from dobot_nexus.models.pi05.torch_core.numerics import OpenPiLinear
from dobot_nexus.models.pi05.torch_core.pi0 import Pi0
from dobot_nexus.models.pi05.torch_core.source import load_core_source
from dobot_nexus.models.pi05.training.compile import (
    _compiled_helpers,
    configure_training_compile,
)


def _meta_core():
    """Only the structural pieces used by model assembly; no full backbone allocation."""
    with torch.device("meta"):
        core = torch.nn.Module()
        core.llm = torch.nn.Module()
        core.llm.configs = [
            gemma.Config(width=w, depth=1, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4)
            for w in (8, 1024)
        ]
        core.llm.layers = torch.nn.ModuleList([
            gemma.Block(core.llm.configs, adarms=[False, True])
        ])
        core.img = torch.nn.Module()
        core.img.encoder = torch.nn.Module()
        vision = torch.nn.Module()
        vision.mlp = torch.nn.Linear(1, 1)
        core.img.encoder.layers = torch.nn.ModuleList([vision])
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            setattr(core, name, torch.nn.Linear(2, 2))
    return core


class MemoryIntegrationTest(unittest.TestCase):
    def test_deployment_bundles_vision_without_policy_conversion_or_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "vision.npz"
            np.savez(source, kernel=np.arange(6, dtype=np.float32))
            digest = content_sha256(source)
            model = Pi05MemoryModel.__new__(Pi05MemoryModel)
            torch.nn.Module.__init__(model)
            model.memory_config = MemoryConfig(
                vision_checkpoint=str(source), vision_sha256=digest
            )
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
        from dobot_nexus.models.pi05.memory.torch import MemoryAttention

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

    def test_sampler_builds_history_cache_once_per_layer_and_request(self):
        observation = SimpleNamespace(state=torch.zeros(1, 2))
        attention = SimpleNamespace(prepare_memory=mock.Mock(side_effect=lambda x: (x + 1, x + 2)))
        core = SimpleNamespace(
            training=False, action_horizon=2, action_dim=2,
            llm=SimpleNamespace(layers=[SimpleNamespace(
                memory_modulation=SimpleNamespace(attention=attention))]),
            _encode_memory=lambda inputs: {"memory_tokens": inputs, "memory_mask": torch.ones(1)},
            build_prefix_cache=lambda obs: (None, None, None),
            run_suffix=mock.Mock(side_effect=lambda *args, **kwargs: kwargs["memory_kv"][0][0]),
            velocity_from_suffix=lambda value: value,
        )
        with mock.patch("dobot_nexus.models.pi05.torch_core.pi0.model.preprocess_observation",
                        side_effect=lambda obs, **kw: obs):
            for value in (1.0, 4.0):
                history = torch.full((1, 2, 2), value)
                actual = Pi0.sample_actions(core, observation, noise=torch.zeros_like(history),
                                            num_steps=5, memory_inputs=history, _cache_memory=True)
                torch.testing.assert_close(actual, -(history + 1))
            self.assertEqual(attention.prepare_memory.call_count, 2)
            self.assertEqual(core.run_suffix.call_count, 10)
            core.training = True
            with self.assertRaisesRegex(RuntimeError, "eval"):
                Pi0.sample_actions(core, observation, memory_inputs=history, _cache_memory=True)

    def test_assembly_reuses_openpi_numerics_and_instance_compilation(self):
        source = load_core_source()
        _compiled_helpers.cache_clear()
        self.addCleanup(_compiled_helpers.cache_clear)
        for enabled in (False, True):
            with self.subTest(compile=enabled):
                model = Pi05MemoryModel.__new__(Pi05MemoryModel)
                torch.nn.Module.__init__(model)
                model.core_model = _meta_core()
                model.core_source = source
                model.pi05_config = SimpleNamespace(operator_numerics="openpi",
                                                    torch_dtype=torch.bfloat16)
                parameters = dict(model.core_model.named_parameters())
                with mock.patch.object(
                    torch, "compile", side_effect=lambda fn: mock.Mock(wraps=fn)
                ):
                    configure_training_compile(model.core_model, enabled=enabled)
                with torch.device("meta"):
                    model._configure_core()
                core = model.core_model
                self.assertIsInstance(core.action_in_proj, OpenPiLinear)
                self.assertTrue(core.preserve_projection_inputs)
                self.assertEqual(core.sampling_noise_dtype, torch.float32)
                for name, parameter in parameters.items():
                    self.assertIs(dict(core.named_parameters())[name], parameter)
                for block in core.llm.layers:
                    self.assertIs(block.memory_modulation.attention.rope, block.attn.rope)
                    self.assertTrue(block.pre_attention_norms[-1].openpi_math)
                    self.assertEqual(block.mlps[-1].gelu_approximate, "tanh")
                self.assertTrue(all(p.is_meta for p in core.parameters()))

    def test_core_forward_passes_history_to_loss(self):
        core = SimpleNamespace(compute_loss=mock.Mock(return_value="loss"))
        inputs = (torch.zeros(1, 2, 2048), torch.zeros(1, 2, 768), torch.ones(1, 2))
        self.assertEqual(Pi0.forward(core, "obs", "actions", memory_inputs=inputs), "loss")
        self.assertIs(core.compute_loss.call_args.kwargs["memory_inputs"], inputs)

    def test_memory_uses_shared_device_transfer(self):
        from dobot_nexus.models.pi05.data_adapter_pi05 import Pi05DataAdapter

        self.assertIs(MemoryDataAdapter.move_encoded, Pi05DataAdapter.move_encoded)

    def test_partial_history_checkpoint_cannot_be_silently_initialized(self):
        core = torch.nn.Module()
        core.base = torch.nn.Linear(1, 1)
        core.memory_encoder = torch.nn.Linear(1, 1)
        state = core.state_dict()
        del state["memory_encoder.bias"]
        with mock.patch(
            "dobot_nexus.models.pi05.memory.checkpoint.load_core_state_dict",
            return_value=(Path("partial.pt"), state),
        ):
            with self.assertRaisesRegex(ValueError, "Incomplete memory checkpoint"):
                load_memory_checkpoint(core, "partial.pt", allow_initialization=True)

    def test_openpi_math_rejects_legacy_inference_compilation_before_loading(self):
        config = clone_config({
            "model": {"checkpoint": "unused", "state_dim": 14, "action_dim": 14,
                      "action_horizon": 50, "operator_numerics": "openpi"},
            "runtime": {"compile": True, "compile_mode": "legacy", "cuda_graph": True},
        })
        with self.assertRaisesRegex(ValueError, "runtime.compile=false"):
            Pi05Config.from_config(config, mode="infer")
        # Training uses the three training helpers, not the inference compile profile.
        self.assertTrue(Pi05Config.from_config(config, mode="train").torch_compile)
        config.runtime.compile = False
        parsed = Pi05Config.from_config(config, mode="infer")
        self.assertTrue(parsed.cuda_graph)
        self.assertEqual(parsed.operator_numerics, "openpi")
        config.runtime.compile = True
        config.runtime.compile_mode = "openpi"
        self.assertEqual(Pi05Config.from_config(config, mode="infer").torch_compile_mode, "openpi")
        config.model.operator_numerics = "legacy"
        with self.assertRaisesRegex(ValueError, "requires.*openpi"):
            Pi05Config.from_config(config, mode="infer")

    def test_openpi_inference_boundaries_preserve_memory_modulation(self):
        from dobot_nexus.models.pi05.memory.torch import MemoryModulation
        from dobot_nexus.models.pi05.openpi_torch import openpi_ops

        torch.manual_seed(42)
        module = MemoryModulation(gemma._apply_rope).eval().to(torch.bfloat16)
        with torch.no_grad():
            module.modulation.weight.normal_(std=0.02)
            module.modulation.bias.normal_(std=0.02)
        x = torch.randn(1, 3, 1024).to(torch.bfloat16)
        memory = torch.randn(1, 5, 1024).to(torch.bfloat16)
        mask = torch.tensor([[True, True, False, True, False]])
        with torch.inference_mode():
            kv = module.attention.prepare_memory(memory)
            expected = module(x, memory, mask)
            actual = module(x, memory, mask, cached_kv=kv, inference_ops=openpi_ops)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
