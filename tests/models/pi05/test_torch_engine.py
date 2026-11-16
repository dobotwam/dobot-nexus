from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from unittest import mock

import torch

from dobot_nexus.models.pi05.openpi_torch.engine import TorchPi05Engine, _signature


@dataclass
class Observation:
    state: torch.Tensor
    images: dict
    tokenized_prompt: torch.Tensor | None = None


class TinyCore(torch.nn.Module):
    action_horizon = 2
    action_dim = 3

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def build_prefix_cache(self, observation):
        return observation.state + observation.images["base"].mean()

    def run_suffix(self, prefix, noise, memory):
        return noise * self.weight + prefix[:, None] + memory

    def sample_actions(self, observation, *, noise=None, rng=None, num_steps=10,
                       memory=None, _prefix_fn=None, _suffix_fn=None):
        if noise is None:
            noise = torch.randn(observation.state.shape[0], 2, 3,
                                generator=rng, device=observation.state.device)
        prefix = (_prefix_fn or self.build_prefix_cache)(observation)
        for _ in range(num_steps):
            noise = (_suffix_fn or self.run_suffix)(prefix, noise, memory)
        return noise


class TorchEngineTest(unittest.TestCase):
    def setUp(self):
        self.core = TinyCore().eval()
        self.observation = Observation(torch.ones(1, 3), {"base": torch.zeros(1, 4)})

    def test_signature_tracks_structure_not_values(self):
        changed = replace(self.observation, state=torch.zeros(1, 3))
        self.assertEqual(_signature(self.observation), _signature(changed))
        self.assertNotEqual(_signature(self.observation), _signature(
            replace(changed, state=torch.zeros(2, 3))))
        self.assertNotEqual(_signature(self.observation), _signature(
            replace(changed, state=torch.zeros(1, 3, dtype=torch.bfloat16))))
        self.assertNotEqual(_signature({"memory": None}), _signature({"memory": (torch.ones(1),)}))
        with self.assertRaises(TypeError):
            _signature(object())

    def test_eager_preserves_rng_and_model_kwargs(self):
        engine = TorchPi05Engine(self.core)
        actual = engine.sample_actions(self.observation, num_steps=2, memory=torch.ones(1),
            rng=torch.Generator().manual_seed(42))
        expected = self.core.sample_actions(self.observation, num_steps=2, memory=torch.ones(1),
            rng=torch.Generator().manual_seed(42))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)

    def test_compile_captures_regions_and_refreshes_inputs(self):
        # Exercise Dynamo/fullgraph on CPU without compiling GPU kernels in CI.
        real_compile = torch.compile
        with mock.patch("torch.compile", side_effect=lambda fn, **kw: real_compile(
            fn, backend="eager", fullgraph=kw["fullgraph"], dynamic=kw["dynamic"]
        )) as compiler:
            engine = TorchPi05Engine(self.core, compile=True)
            for value, steps in ((1.0, 2), (3.0, 5), (-2.0, 2)):
                obs = replace(self.observation, state=torch.full((1, 3), value))
                memory = torch.tensor(value)
                actual = engine.sample_actions(obs, num_steps=steps, memory=memory,
                    rng=torch.Generator().manual_seed(42))
                expected = self.core.sample_actions(obs, num_steps=steps, memory=memory,
                    rng=torch.Generator().manual_seed(42))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(compiler.call_count, 2)
            engine.clear()
            self.assertIsNone(engine._compiled)

    def test_acceleration_rejects_training_and_cpu_graphs(self):
        self.core.train()
        engine = TorchPi05Engine(self.core, compile=True)
        with self.assertRaisesRegex(RuntimeError, "eval"):
            engine.sample_actions(self.observation)
        self.core.eval()
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            TorchPi05Engine(self.core, cuda_graph=True).sample_actions(self.observation)
        with self.assertRaisesRegex(ValueError, "positive"):
            engine.sample_actions(self.observation, num_steps=0)
        with self.assertRaises(ValueError):
            TorchPi05Engine(self.core, max_graphs=0)
        with self.assertRaises(ValueError):
            TorchPi05Engine(self.core, compile_mode="reduce-overhead")

    def test_legacy_profile_rejects_incompatible_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "preserve_precision"):
            TorchPi05Engine(self.core, compile=True, compile_mode="legacy",
                            preserve_precision=False)
        engine = TorchPi05Engine(self.core, compile=True, compile_mode="legacy")
        with self.assertRaisesRegex(ValueError, "baseline legacy"):
            engine.sample_actions(self.observation)

    def test_modulations_are_used_without_compilation(self):
        engine = TorchPi05Engine(self.core, cuda_graph=True)
        native = self.core.run_suffix
        seen = []

        def suffix(*args, cached_modulations):
            seen.append(cached_modulations)
            return native(*args)

        modulations = (torch.tensor(1), torch.tensor(2))
        self.core.run_suffix = suffix
        inputs = (self.observation, torch.zeros(1, 2, 3), {"memory": torch.ones(1)})
        engine._sample(inputs, 2, modulations)
        self.assertEqual(seen, list(modulations))
        self.assertIsNone(engine._compiled)


class LegacyOperatorsTest(unittest.TestCase):
    def test_explicit_projection_preserves_master_gradients_and_compiled_values(self):
        from dobot_nexus.models.pi05.torch_core.numerics import (
            OpenPiLinear, configure_input_dtype_linears,
        )
        from dobot_nexus.models.pi05.openpi_torch import legacy_ops, openpi_ops

        core = torch.nn.ModuleDict({'ordinary': torch.nn.Linear(8, 4),
                                   'promoted': OpenPiLinear(8, 4)})
        parameter = core['ordinary'].weight
        promoted = core['promoted']
        keys = list(core.state_dict())
        configure_input_dtype_linears(core)
        self.assertIs(core['ordinary'].weight, parameter)
        self.assertIs(core['promoted'], promoted)
        self.assertEqual(list(core.state_dict()), keys)
        value = torch.randn(2, 8, dtype=torch.bfloat16)
        with torch.autocast('cpu', enabled=False):
            expected = core['ordinary'](value)
            self.assertEqual(expected.dtype, torch.bfloat16)
            expected.float().square().mean().backward()
            self.assertEqual(parameter.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            for ops in (legacy_ops, openpi_ops):
                compiled = torch.compile(ops.project, backend='aot_eager', fullgraph=True)
                with torch.inference_mode():
                    torch.testing.assert_close(compiled(core['ordinary'], value),
                                               expected, rtol=0, atol=0)

    def test_compiled_gemms_preserve_autocast_with_fp32_parameters(self):
        from dobot_nexus.models.pi05.openpi_torch import legacy_ops as ops

        # AOT removes ambient autocast from opaque kernels; backend="eager"
        # alone does not expose this mixed-parameter inference regression.
        x, weight, bias = torch.randn(2, 8), torch.randn(5, 8), torch.randn(5)
        cases = [
            (ops.linear, torch.nn.functional.linear, (x.bfloat16(), weight, bias)),
            (ops.linear, torch.nn.functional.linear, (x, weight, None)),
            (ops.matmul, torch.matmul, (x.bfloat16(), weight.T)),
            (ops.einsum, torch.einsum, ("ij,kj->ik", x.bfloat16(), weight)),
        ]
        for operator, native, args in cases:
            with self.subTest(operator=operator.__name__, bias=args[-1] is not None):
                compiled = torch.compile(operator, backend="aot_eager", fullgraph=True)
                with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                    expected = native(*args)
                    self.assertEqual(expected.dtype, torch.bfloat16)
                    torch.testing.assert_close(operator(*args), expected, rtol=0, atol=0)
                    torch.testing.assert_close(compiled(*args), expected, rtol=0, atol=0)

    def test_compiled_openpi_projection_respects_disabled_autocast(self):
        from dobot_nexus.models.pi05.openpi_torch import openpi_ops as ops
        from dobot_nexus.models.pi05.torch_core.numerics import OpenPiLinear

        module = OpenPiLinear(8, 16).to(torch.bfloat16)
        value = torch.randn(2, 4, 8)
        compiled = torch.compile(ops.project, backend="aot_eager", fullgraph=True)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = module(value)
            self.assertEqual(expected.dtype, torch.float32)
            torch.testing.assert_close(compiled(module, value), expected, rtol=0, atol=0)

    def test_compiled_time_embedding_preserves_native_autocast(self):
        from dobot_nexus.models.pi05.openpi_torch import legacy_ops as ops
        from dobot_nexus.models.pi05.torch_core.pi0 import posemb_sincos

        compiled = torch.compile(ops.time_embedding, backend="aot_eager", fullgraph=True)
        for dtype in (torch.float32, torch.bfloat16):
            pos = torch.tensor([1.0, 2 / 3, 1 / 3], dtype=dtype)
            for enabled in (False, True):
                with self.subTest(dtype=dtype, autocast=enabled):
                    with torch.inference_mode(), torch.autocast(
                        "cpu", dtype=torch.bfloat16, enabled=enabled,
                    ):
                        expected = posemb_sincos(pos, 32)
                        torch.testing.assert_close(compiled(pos, 32), expected, rtol=0, atol=0)

    def test_openpi_boundaries_preserve_promoted_linear_and_reciprocal_sqrt(self):
        from dobot_nexus.models.pi05.openpi_torch import openpi_ops as ops
        from dobot_nexus.models.pi05.torch_core.numerics import OpenPiLinear

        module = OpenPiLinear(8, 16).to(torch.bfloat16)
        value = torch.randn(2, 4, 8)
        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16):
                x = value.to(dtype)
                actual = ops.project(module, x)
                torch.testing.assert_close(actual, module(x), rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                variance = x.float().square().mean(-1, keepdim=True)
                expected = torch.reciprocal(torch.sqrt(variance + 1e-6))
                torch.testing.assert_close(ops.rms_inverse(x), expected, rtol=0, atol=0)
        torch.library.opcheck(
            ops.rms_inverse, (value,), test_utils=("test_schema", "test_faketensor")
        )

    def test_fake_kernels_match_native_layouts_and_schemas(self):
        from dobot_nexus.models.pi05.openpi_torch import legacy_ops as ops

        x = torch.randn(2, 4, 8).transpose(0, 1)
        gate, value = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        positions = torch.arange(4).repeat(2, 1)
        cases = [
            (ops.add, (x, torch.randn(8))),
            (ops.sub, (x, torch.randn(8))),
            (torch.ops.nexus_pi05_legacy.linear.default, (x, torch.randn(5, 8), torch.randn(5))),
            (torch.ops.nexus_pi05_legacy.matmul.default, (x, torch.randn(8, 5))),
            (torch.ops.nexus_pi05_legacy.einsum.default, ("btd,bsd->bts", x, torch.randn(4, 7, 8))),
            (ops.softmax, (x, -1)),
            (ops.gelu_glu, (gate.transpose(0, 1), value.transpose(0, 1))),
            (ops.silu, (x,)),
            (ops.rms_inverse, (x,)),
            (ops.rope_angles, (positions, 8)),
            (torch.ops.nexus_pi05_legacy.time_embedding.default, (torch.tensor([1.0, 0.7]), 8)),
            (torch.ops.nexus_pi05_legacy.time_embedding.default,
             (torch.tensor([1.0, 0.7]), 8, 0.004, 4.0, torch.bfloat16)),
        ]
        for op, args in cases:
            with self.subTest(op=op):
                torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))

    def test_modulation_cache_uses_each_native_time_and_batch(self):
        from types import SimpleNamespace

        from dobot_nexus.models.pi05.openpi_torch import conditioning as ops

        projection = torch.nn.Linear(3, 9).to(torch.bfloat16)
        norm = SimpleNamespace(ada_modulation=projection)
        seen = []

        class Core:
            llm = SimpleNamespace(adarms=[False, True], final_norms=[None, norm],
                layers=[SimpleNamespace(pre_attention_norms=[None, norm],
                                        pre_ffw_norms=[None, norm])])

            def embed_suffix(self, observation, noise, timestep):
                seen.append(timestep.clone())
                return None, None, None, timestep[:, None].expand(-1, 3).to(torch.bfloat16)

        noise = torch.zeros(2, 4, 3)
        with torch.inference_mode():
            cache = ops.prepare_modulations(Core(), None, noise, 7)
            t, dt = 1.0, -1.0 / 7
            for times, row in zip(seen, cache, strict=True):
                torch.testing.assert_close(times, torch.full((2,), t), rtol=0, atol=0)
                expected = projection(times[:, None].expand(-1, 3).to(torch.bfloat16))
                self.assertEqual(len(row), 3)
                for actual in row:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                t += dt
        self.assertEqual(len(cache), 7)


if __name__ == "__main__":
    unittest.main()
