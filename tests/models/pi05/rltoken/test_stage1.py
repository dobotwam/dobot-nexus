import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dobot_nexus.algorithms.rltoken.artifacts import load_provider, write_model_bundle
from dobot_nexus.core.config import _wrap
from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions
from dobot_nexus.models.pi05.rltoken.modeling_stage1 import Pi05RLTokenModel
from tests.models.pi05.test_model import _FakeCore


class TinyPrefixCore(_FakeCore):
    def compute_loss(self, observation, actions, *, train=True, rng=None, _return_prefix=False):
        self.forward_calls = getattr(self, "forward_calls", 0) + 1
        loss = super().compute_loss(observation, actions, train=train, rng=rng)
        prefix = self.weight * torch.ones(actions.shape[0], 4, 6, device=actions.device)
        return (
            (loss, prefix, torch.ones(actions.shape[0], 4, device=actions.device, dtype=torch.bool))
            if _return_prefix
            else loss
        )

    def build_prefix_cache(self, observation):
        self.prefix_calls = getattr(self, "prefix_calls", 0) + 1
        batch = observation.state.shape[0]
        return self.weight * torch.ones(batch, 4, 6), torch.ones(batch, 4, dtype=torch.bool), ()

    def sample_actions(self, observation, *, num_steps=10, noise=None, rng=None, _prefix_fn=None):
        _prefix_fn(observation)
        return super().sample_actions(observation, num_steps=num_steps, noise=noise, rng=rng)


class Stage1Test(unittest.TestCase):
    def test_camera_slot_assignment_changes_frozen_feature_identity(self):
        from dataclasses import replace

        model = Pi05RLTokenModel(self.config, mode="train")
        before = model.rl_feature_spec.feature_id
        mapping = dict(model.pi05_config.camera_mapping)
        mapping["base"], mapping["left_wrist"] = mapping["left_wrist"], mapping["base"]
        model.pi05_config = replace(model.pi05_config, camera_mapping=mapping)
        self.assertNotEqual(before, model.rl_feature_spec.feature_id)

    def test_batched_features_preserve_per_observation_codec_context(self):
        model = Pi05RLTokenModel(self.config, mode="train").eval().requires_grad_(False)
        obs = ObservationBatch(
            {key: np.zeros((3, 224, 224, 3), np.uint8) for key in model.policy_spec.camera_names},
            np.array([[0, 0], [0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            ["pick"] * 3,
        )
        packet = model.extract_rl_features(obs, SamplingOptions(seed=3))
        self.assertEqual(model.core_model.prefix_calls, 1)
        self.assertEqual(packet.latent.shape[0], 3)
        for index in range(3):
            row = packet.row(index)
            np.testing.assert_allclose(
                row.action_context["raw_state"], obs.state[index : index + 1]
            )
            encoded = model.adapter.encode_executed_actions(
                np.zeros((1, 2, 2), np.float32),
                aux={**row.action_context, "executed_steps": [2]},
            )
            self.assertTrue(encoded.executed_mask.all())

    def test_core_returns_same_forward_prefix_without_changing_default_loss(self):
        from dobot_nexus.models.pi05.torch_core.pi0 import Pi0

        class LLM(torch.nn.Module):
            def forward(self, inputs, **kwargs):
                return inputs, ()

        core = Pi0.__new__(Pi0)
        torch.nn.Module.__init__(core)
        core.embed_dtype = torch.float32
        core.action_horizon = 2
        core.memory_encoder = None
        core.llm = LLM()
        core.action_out_proj = torch.nn.Identity()
        prefix = torch.ones(1, 3, 2, requires_grad=True)
        core.embed_prefix = lambda obs: (
            prefix,
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
        )
        core.embed_suffix = lambda obs, x, t: (
            x,
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([True, False]),
            None,
        )
        with (
            patch(
                "dobot_nexus.models.pi05.torch_core.model.preprocess_observation",
                side_effect=lambda obs, **kw: obs,
            ) as prepare,
            patch(
                "dobot_nexus.models.pi05.torch_core.model._observation_to_dtype",
                side_effect=lambda obs, dtype: obs,
            ),
        ):
            args = dict(noise=torch.ones(1, 2, 2), time=torch.tensor([0.5]))
            plain = core.compute_loss(None, torch.zeros(1, 2, 2), **args)
            detailed, returned, valid = core.compute_loss(
                None, torch.zeros(1, 2, 2), _return_prefix=True, **args
            )
        torch.testing.assert_close(plain, detailed)
        self.assertIs(returned, prefix)
        self.assertTrue(valid.all())
        self.assertEqual(prepare.call_count, 2)

    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.save({"weight": torch.tensor(0.25)}, self.root / "base.pt")
        (self.root / "tokenizer.model").write_bytes(b"tokenizer-fixture")
        (self.root / "stats.json").write_text(
            json.dumps(
                {name: {"mean": [0.0, 0.0], "std": [1.0, 1.0]} for name in ("state", "actions")}
            )
        )
        self.config = _wrap(
            {
                "schema_version": 1,
                "task": "train",
                "algorithm": "bc",
                "model": {
                    "target": "dobot_nexus.models.pi05.rltoken.modeling_stage1:Pi05RLTokenModel",
                    "checkpoint": str(self.root / "base.pt"),
                    "precision": "fp32",
                    "state_dim": 2,
                    "action_dim": 2,
                    "action_horizon": 3,
                    "rltoken": {
                        "token": {
                            "input_dim": 6,
                            "width": 8,
                            "heads": 2,
                            "layers": 1,
                            "max_tokens": 8,
                        },
                        "alpha_vla": 1.0,
                        "control_dt": 0.02,
                        "action_codec": {
                            "name": "absolute_affine_v1",
                            "low": [-2.0, -2.0],
                            "high": [2.0, 2.0],
                            "units": "rad",
                        },
                    },
                },
                "data": {
                    "norm_stats_path": str(self.root / "stats.json"),
                    "tokenizer_path": str(self.root / "tokenizer.model"),
                    "field_mapping": {
                        "images": {"base": "base", "left_wrist": "left", "right_wrist": "right"}
                    },
                    "transforms": {"action": {"type": "absolute"}},
                },
                "runtime": {"device": "cpu", "precision": "fp32"},
                "train": {"gradient_checkpointing": False},
            }
        )
        self.core_patch = patch(
            "dobot_nexus.models.pi05.torch_core.pi0_config.Pi0Config.create",
            lambda config: TinyPrefixCore(config),
        )
        self.core_patch.start()
        self.addCleanup(self.core_patch.stop)
        tokenizer_patch = patch(
            "dobot_nexus.models.pi05.data.pipeline._Tokenizer._get_processor",
            return_value=SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]),
        )
        tokenizer_patch.start()
        self.addCleanup(tokenizer_patch.stop)

    def test_one_forward_and_separate_loss_gradients(self):
        model = Pi05RLTokenModel(self.config, mode="train")
        batch = {"observation": None, "actions": torch.ones(1, 3, 32)}
        output = model.training_step(batch)
        self.assertEqual(model.core_model.forward_calls, 1)
        output.aux["token_loss"].backward()
        self.assertIsNone(model.core_model.weight.grad)
        self.assertTrue(any(p.grad is not None for p in model.token_encoder.parameters()))
        model.zero_grad(set_to_none=True)
        output = model.training_step(batch)
        output.aux["flow_loss"].backward()
        self.assertIsNotNone(model.core_model.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.token_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.token_decoder.parameters()))

    def test_stage1_full_restore_and_feature_export_share_prefix_once(self):
        model = Pi05RLTokenModel(self.config, mode="train")
        artifact = write_model_bundle(self.root / "stage1", model, model.root_config)
        saved = torch.load(artifact / "deployment/pytorch/full_weights.pt", weights_only=True)
        self.assertTrue(any(key.startswith("token_decoder.") for key in saved))
        provider, cfg, spec = load_provider(artifact)
        image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        observation = ObservationBatch(
            {key: image for key in provider.policy_spec.camera_names},
            np.zeros((1, 2), np.float32),
            ["pick"],
        )
        packet = provider.extract_rl_features(observation, SamplingOptions(seed=3))
        self.assertEqual(provider.core_model.prefix_calls, 1)
        self.assertEqual(packet.state.shape[-1], 32)
        self.assertEqual(provider.policy_spec.state_dim, 2)
        packet.validate(spec)
        bundled = write_model_bundle(self.root / "features", provider, cfg)
        feature_weights = torch.load(
            bundled / "deployment/pytorch/full_weights.pt", weights_only=True
        )
        self.assertFalse(any(key.startswith("token_decoder.") for key in feature_weights))
        again, _, again_spec = load_provider(bundled)
        self.assertEqual(spec, again_spec)
        torch.testing.assert_close(
            again.extract_rl_features(observation, SamplingOptions(seed=3)).latent, packet.latent
        )
        # Complete Stage 1 restoration may not silently omit its Decoder.
        self.config.model.rltoken.initialization = "rltoken"
        self.config.model.checkpoint = str(bundled / "deployment/pytorch/full_weights.pt")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            Pi05RLTokenModel(self.config, mode="train")

    def test_command_codec_roundtrip_and_out_of_domain_detection(self):
        adapter = Pi05RLTokenModel(self.config, mode="train").adapter
        aux = {"observation_id": "obs", "raw_state": [[0.0, 0.0]], "executed_steps": [2]}
        commands = np.array([[[-2.0, 1.0], [2.0, -1.0]]], dtype=np.float32)
        encoded = adapter.encode_executed_actions(commands, aux=aux)
        np.testing.assert_allclose(adapter.decode_actions(encoded.actions).actions, commands)
        commands[0, 1, 0] = 3.0
        encoded = adapter.encode_executed_actions(commands, aux=aux)
        self.assertFalse(encoded.encodable_mask[0, 1])
        self.assertGreater(encoded.actions[0, 1, 0], 1.0)
        with self.assertRaisesRegex(ValueError, "observation_id"):
            adapter.encode_executed_actions(commands)
