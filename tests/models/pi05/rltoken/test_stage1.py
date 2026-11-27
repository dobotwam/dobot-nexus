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
    def test_bf16_frozen_provider_survives_artifact_roundtrip(self):
        self.config.model.precision = self.config.runtime.precision = "bf16"
        self.config.model.rltoken.training_mode = "token_only"
        self.config.model.rltoken.frozen_parameter_precision = "bf16"
        model = Pi05RLTokenModel(self.config, mode="train")
        self.assertEqual(next(model.core_model.parameters()).dtype, torch.bfloat16)
        self.assertFalse(any(p.requires_grad for p in model.core_model.parameters()))
        self.assertEqual(next(model.token_encoder.parameters()).dtype, torch.float32)
        artifact = write_model_bundle(self.root / "bf16", model, model.root_config)
        provider, _, spec = load_provider(artifact)
        self.assertEqual(next(provider.core_model.parameters()).dtype, torch.bfloat16)
        self.assertEqual(spec, model.rl_feature_spec)
        self.config.model.rltoken.frozen_parameter_precision = "fp32"
        fp32 = Pi05RLTokenModel(self.config, mode="train")
        self.assertNotEqual(spec.feature_id, fp32.rl_feature_spec.feature_id)

    def test_fp32_compute_rejects_bf16_frozen_provider(self):
        self.config.model.rltoken.training_mode = "token_only"
        self.config.model.rltoken.frozen_parameter_precision = "bf16"
        with self.assertRaisesRegex(ValueError, "FP32 compute"):
            Pi05RLTokenModel(self.config, mode="train")

    def test_export_uses_only_gathered_state(self):
        model = Pi05RLTokenModel(self.config, mode="train")
        gathered = model.state_dict()
        expected = model.rl_feature_spec.to_dict()
        with patch.object(model, "state_dict", side_effect=AssertionError("second FSDP gather")):
            with self.assertRaisesRegex(RuntimeError, "Export Stage 1 weights"):
                model.artifact_metadata()
            self.assertEqual(set(model.export_backend_state_dict(gathered)), set(gathered))
            self.assertEqual(model.artifact_metadata()["rl_features"], expected)
            with self.assertRaisesRegex(ValueError, "all core"):
                model.export_backend_state_dict({})

    def test_autocast_off_disables_core_and_token_amp_and_changes_identity(self):
        from dobot_nexus.algorithms.rltoken.identity import content_digest

        self.config.model.precision = self.config.runtime.precision = "bf16"
        self.config.model.rltoken.training_mode = "token_only"
        model = Pi05RLTokenModel(self.config, mode="train")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.25)
        with patch("dobot_nexus.models.pi05.rltoken.features.content_digest",
                   wraps=content_digest) as digest:
            historical = model.rl_feature_spec.feature_id
        inputs = digest.call_args.args[0]
        self.assertNotIn("autocast", inputs)
        # Default feature identity of these fixed weights on the rebased branch.
        self.assertEqual(historical, "94138ea1356539f89ed11a923a6d1044083839f54109c80ef4529216321f0a8f")
        self.config.runtime.autocast = "off"
        disabled = Pi05RLTokenModel(self.config, mode="train")
        disabled.load_state_dict(model.state_dict())
        model = disabled
        self.assertEqual(model.rl_feature_spec.feature_id, content_digest(dict(inputs, autocast="off")))

        observed = []
        hook = model.token_encoder.register_forward_pre_hook(
            lambda *_: observed.append(("token", torch.is_autocast_enabled("cpu")))
        )
        self.addCleanup(hook.remove)
        prefix = model.core_model.build_prefix_cache

        def capture(obs):
            observed.append(("core", torch.is_autocast_enabled("cpu")))
            return prefix(obs)

        image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        observation = ObservationBatch({k: image for k in model.policy_spec.camera_names},
                                       np.zeros((1, 2), np.float32), ["pick"])
        with (patch.object(model.core_model, "build_prefix_cache", capture),
              patch("dobot_nexus.models.pi05.torch_core.model.preprocess_observation",
                    side_effect=lambda obs, **kwargs: obs),
              torch.autocast("cpu", dtype=torch.bfloat16)):
            model.training_step({"observation": SimpleNamespace(state=torch.ones(1, 32)),
                                 "actions": torch.ones(1, 3, 32)})
            model.eval().extract_rl_features(observation, SamplingOptions(seed=0))
        self.assertEqual(observed, [("core", False), ("token", False)] * 2)

    def test_bf16_provider_rejects_joint_training(self):
        self.config.model.rltoken.frozen_parameter_precision = "bf16"
        with self.assertRaisesRegex(ValueError, "token_only"):
            Pi05RLTokenModel(self.config, mode="train")

    def test_normalized_joint_delta_codec_roundtrip_and_controller_identity(self):
        codec = {
            "name": "normalized_joint_delta_v1", "low": [-1, -1], "high": [1, 1],
            "units": "normalized_joint_delta_and_gripper", "reference_bounds": "clip",
            "controller": {"mode": "pd_joint_delta_pos", "delta_reference": "current_qpos",
                           "arm_delta_low": [-0.1], "arm_delta_high": [0.1],
                           "gripper_position_low": -0.01, "gripper_position_high": 0.04},
        }
        self.config.model.rltoken.action_codec = _wrap(codec)
        model = Pi05RLTokenModel(self.config, mode="train")
        adapter = model.adapter
        context = {"observation_id": "x", "raw_state": [[0, 0]], "executed_steps": [1]}
        commands = np.array([[[0.2, -0.8]]], np.float32)
        encoded = adapter.encode_executed_actions(commands, aux=context)
        np.testing.assert_array_equal(encoded.actions.numpy(), commands)
        np.testing.assert_array_equal(adapter.decode_actions(encoded.actions).actions, commands)
        tiny = np.array([[[1e-10, -1e-9], [1e-4, -1e-4]]], dtype=np.float32)
        tiny_encoded = adapter.encode_executed_actions(tiny, aux={**context, "executed_steps": [2]})
        np.testing.assert_array_equal(tiny_encoded.actions.numpy(), tiny)
        np.testing.assert_array_equal(adapter.decode_actions(tiny_encoded.actions).actions, tiny)
        self.assertTrue(encoded.encodable_mask.all())
        invalid = adapter.encode_executed_actions(commands * 10, aux=context)
        self.assertFalse(invalid.encodable_mask.all())
        reference = adapter.decode_vla_actions(np.array([[[2, -2]]]), aux=context)
        np.testing.assert_array_equal(reference.actions, [[[1, -1]]])
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter.decode_vla_actions(np.array([[[np.nan, 0]]]), aux=context)
        before = model.rl_feature_spec.feature_id
        self.config.model.rltoken.action_codec.controller.arm_delta_high = [0.2]
        changed = Pi05RLTokenModel(self.config, mode="train")
        self.assertNotEqual(before, changed.rl_feature_spec.feature_id)

    def test_camera_slot_assignment_changes_frozen_feature_identity(self):
        from dataclasses import replace

        model = Pi05RLTokenModel(self.config, mode="train")
        before = model.rl_feature_spec.feature_id
        mapping = dict(model.pi05_config.camera_mapping)
        mapping["base"], mapping["left_wrist"] = mapping["left_wrist"], mapping["base"]
        model.pi05_config = replace(model.pi05_config, camera_mapping=mapping)
        self.assertNotEqual(before, model.rl_feature_spec.feature_id)

    def test_joint_delta_codec_rejects_implicit_scaling_and_mismatched_units(self):
        self.config.model.rltoken.action_codec = _wrap({
            "name": "normalized_joint_delta_v1", "low": [-1, -1], "high": [1, 1],
            "units": "normalized_joint_delta_and_gripper",
        })
        with self.assertRaisesRegex(ValueError, "current_qpos"):
            Pi05RLTokenModel(self.config, mode="train")
        self.config.model.rltoken.action_codec.units = "rad"
        with self.assertRaisesRegex(ValueError, "units"):
            Pi05RLTokenModel(self.config, mode="train")

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

    def test_token_episode_ids_are_validated_before_disjointness(self):
        from dobot_nexus.algorithms.rltoken.token_pretraining import TokenPretrainTrainer
        for validation in (["0"], [False], [1.0], [-1], [1, 1], [0]):
            self.config.algorithm = _wrap(dict(token=dict(train_episodes=[0],
                                                          validation_episodes=validation)))
            with self.subTest(validation=validation), self.assertRaisesRegex(ValueError, "episode"):
                TokenPretrainTrainer(self.config, None)

    def test_token_trainer_publishes_validation_and_resumes(self):
        import logging
        from dobot_nexus.algorithms.rltoken.token_pretraining import TokenPretrainTrainer
        self.config.model.rltoken.training_mode = "token_only"
        self.config.model.rltoken.token.width = 6
        self.config.model.rltoken.token.mode = "bottleneck"
        self.config.algorithm = _wrap(dict(target="dobot_nexus.algorithms.rltoken.token_pretraining:TokenPretrainTrainer",
            token=dict(train_episodes=[0,1], validation_episodes=[2,3], min_zero_ratio=1.2, min_shuffle_ratio=1.2)))
        self.config.exp = _wrap(dict(seed=3))
        self.config.train.max_steps = 1
        self.config.train.save_every = 1
        self.config.data.batch_size = 2
        def collate(rows):
            return {"observation": SimpleNamespace(state=torch.ones(len(rows),32)),
                    "actions": torch.ones(len(rows),3,32)}
        loader = SimpleNamespace(dataset=[0,1,2,3], collate_fn=collate)
        experiment = SimpleNamespace(run_dir=self.root/"token-run", logger=logging.getLogger("token-test"))
        with patch.object(Pi05RLTokenModel, "build_train_dataloader", return_value=loader), patch(
            "dobot_nexus.models.pi05.rltoken.data_adapter.Pi05RLDataAdapter.encode_train_batch",
            side_effect=lambda raw, device: raw), patch(
            "dobot_nexus.models.pi05.torch_core.model.preprocess_observation", side_effect=lambda o, **kw: o):
            result = TokenPretrainTrainer(self.config, experiment).run()
            self.assertEqual(result.global_step, 1)
            report = json.loads((experiment.run_dir/"token/step-000000001/token_validation.json").read_text())
            self.assertIn("feature_id", report)
            self.assertFalse(report["accepted"])  # Constant fake prefixes cannot pass shuffle dependence.
            self.config.train.resume_from = result.checkpoint
            self.config.train.max_steps = 2
            with patch("dobot_nexus.algorithms.rltoken.token_pretraining.feature_metadata",
                       side_effect=OSError("bundle staged")), self.assertRaisesRegex(OSError, "bundle staged"):
                TokenPretrainTrainer(self.config, experiment).run()
            self.assertFalse((experiment.run_dir / "token/step-000000002").exists())
            self.assertEqual((experiment.run_dir / "token/last").read_text(), "step-000000001")
            # Interrupt the second publication after the complete bundle is in place,
            # before the last pointer and resume state have advanced.
            import os
            replace_path = os.replace
            def interrupt(path, target, **kwargs):
                if path == "last" and "src_dir_fd" in kwargs:
                    raise OSError("interrupted publication")
                return replace_path(path, target, **kwargs)
            with patch.object(os, "replace", interrupt), self.assertRaisesRegex(OSError, "interrupted"):
                TokenPretrainTrainer(self.config, experiment).run()
            self.assertEqual(torch.load(result.checkpoint, weights_only=True)["step"], 1)
            resumed = TokenPretrainTrainer(self.config, experiment).run()
            self.assertEqual(torch.load(resumed.checkpoint, weights_only=True)["step"], 2)
            self.assertEqual((experiment.run_dir / "token/last").read_text(), "step-000000002")
            self.assertEqual(resumed.global_step, 2)

            self.config.train.resume_from = None
            self.config.train.max_steps = 1
            self.config.algorithm.token.data_loader = "fixture:loader"
            experiment.run_dir = self.root / "custom-token-run"
            supplied = []
            def custom_loader(model, config):
                supplied.append(config.data.episode_ids)
                return loader
            with patch("dobot_nexus.algorithms.rltoken.token_pretraining.import_from_string",
                       side_effect=lambda target: custom_loader if target == "fixture:loader" else Pi05RLTokenModel):
                TokenPretrainTrainer(self.config, experiment).run()
            self.assertEqual(supplied, [[0,1], [2,3]])

    def test_token_only_keeps_vla_frozen_and_skips_flow_loss(self):
        from dobot_nexus.algorithms.rltoken.identity import content_digest
        self.config.model.rltoken.training_mode = "token_only"
        self.config.model.rltoken.alpha_vla = 0.
        self.config.model.rltoken.token.width = 6
        self.config.model.rltoken.token.mode = "bottleneck"
        model = Pi05RLTokenModel(self.config, mode="train").train()
        before = content_digest(model.core_model.state_dict())
        optimizer = torch.optim.Adam(model.get_param_groups({}), lr=.001)
        self.assertFalse(model.core_model.training)
        self.assertTrue(all(p.requires_grad for group in optimizer.param_groups for p in group["params"]))
        with patch("dobot_nexus.models.pi05.torch_core.model.preprocess_observation", side_effect=lambda o, **kw: o):
            output = model.training_step({"observation": SimpleNamespace(state=torch.ones(2,32)),
                                          "actions": torch.ones(2,3,32)})
        output.loss.backward()
        optimizer.step()
        self.assertFalse(hasattr(model.core_model, "forward_calls"))
        self.assertEqual(before, content_digest(model.core_model.state_dict()))
        self.assertEqual(model.core_model.prefix_calls, 1)

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
