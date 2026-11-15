"""Contracts for the shared Torch core and instance-local training compilation."""

from __future__ import annotations

import subprocess
import sys
import unittest
from unittest import mock

import torch

from dobot_nexus.models.pi05.torch_core import gemma, lora
from dobot_nexus.models.pi05.torch_core.source import load_core_source
from dobot_nexus.models.pi05.torch_core.utils import gelu_glu
from dobot_nexus.models.pi05.training.compile import (
    _compiled_helpers,
    configure_training_compile,
)


class TorchCoreTest(unittest.TestCase):
    def test_fresh_import_is_lazy_and_does_not_compile_or_load_compatibility(self):
        subprocess.run(
            [sys.executable, "-c", """
import sys
import torch
from unittest.mock import patch
error = AssertionError('import-time compilation')
with patch.object(torch, 'compile', side_effect=error) as compile:
    from dobot_nexus.models.pi05.torch_core.source import load_core_source
    assert 'dobot_nexus.models.pi05.torch_core.pi0' not in sys.modules
    first = load_core_source()
    second = load_core_source()
    assert torch.compile is compile
    assert first.Pi0Config is second.Pi0Config
    assert first.Observation is second.Observation
    assert first.Pi0Config.__module__ == 'dobot_nexus.models.pi05.torch_core.pi0_config'
    assert not any(name.startswith(('_dobot_nexus_pi_ops_', 'jax', 'rlinf', 'ray',
                                  'dobot_nexus.models.pi05.compat')) for name in sys.modules)
"""],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_training_compile_is_per_instance_and_preserves_parameters(self):
        config = gemma.Config(width=8, depth=1, mlp_dim=16,
                              num_heads=2, num_kv_heads=1, head_dim=4)
        train = gemma.Block([config, config], adarms=[False, True])
        infer = gemma.Block([config, config], adarms=[False, True])
        # Include the optional FFN variant in helper routing without a full model.
        train.extra_lora = lora.FeedForward(8, 16, lora.LoRAConfig(rank=2, alpha=2))
        before = dict(train.named_parameters())
        keys = list(train.state_dict())
        compiled = {}

        def compile_helper(fn):
            compiled[fn] = mock.Mock(wraps=fn)
            return compiled[fn]

        _compiled_helpers.cache_clear()
        self.addCleanup(_compiled_helpers.cache_clear)
        with mock.patch.object(torch, "compile", side_effect=compile_helper) as compile:
            configure_training_compile(train, enabled=True)
            compile.assert_has_calls([
                mock.call(gemma._apply_rope), mock.call(gemma._fused_gated_residual),
                mock.call(gelu_glu),
            ])
            self.assertEqual(compile.call_count, 3)

        self.assertIs(train.attn.rope, compiled[gemma._apply_rope])
        self.assertIs(train.mlps[0].activation, compiled[gelu_glu])
        self.assertIs(train.extra_lora.activation, compiled[gelu_glu])
        x = torch.ones(1, 2, 8)
        train.gated_residual(x, x, x)
        compiled[gemma._fused_gated_residual].assert_called_once_with(x, x, x)
        self.assertIs(infer.attn.rope, gemma._apply_rope)
        self.assertIs(infer.mlps[0].activation, gelu_glu)
        self.assertIs(infer.gated_residual, gemma._gated_residual)
        self.assertEqual(keys, list(train.state_dict()))
        for name, parameter in train.named_parameters():
            self.assertIs(parameter, before[name])

        configure_training_compile(train, enabled=False)
        self.assertIs(train.attn.rope, gemma._apply_rope)
        self.assertIs(train.extra_lora.activation, gelu_glu)

    def test_source_types_match_direct_imports(self):
        from dobot_nexus.models.pi05.torch_core.model import Observation
        from dobot_nexus.models.pi05.torch_core.pi0_config import Pi0Config

        source = load_core_source()
        self.assertIs(source.Observation, Observation)
        self.assertIs(source.Pi0Config, Pi0Config)


if __name__ == "__main__":
    unittest.main()
