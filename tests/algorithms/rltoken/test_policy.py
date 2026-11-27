from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from dobot_nexus.algorithms.rltoken.artifacts import retain_feature_bundle, write_model_bundle
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.policy import RLTokenPolicy
from dobot_nexus.core.config import _wrap
from dobot_nexus.core.contracts import SamplingOptions
from dobot_nexus.inference.policy import InferencePolicy

from .fakes import TinyFeatureModel, observation


class PolicyTest(unittest.TestCase):
    def test_deployment_restores_phase_routing_and_requires_live_signal(self):
        from .test_routing import SWITCH, flagged
        config = self.make_config()
        config.algorithm = _wrap(dict(target="dobot_nexus.algorithms.rltoken.training:RLTokenTrainer",
                                      policy_switch=SWITCH))
        policy = RLTokenPolicy(config, mode="infer")
        root = self.root / "phase-deployment"
        write_model_bundle(root, policy, config)
        config.model.artifact = str(root)
        config.algorithm.policy_switch.execution_scope = "all"
        loaded = RLTokenPolicy(config, mode="infer")
        self.assertEqual(loaded.policy_switch.execution_scope, "critical_phase")
        calls = []
        hook = loaded.actor.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
        self.addCleanup(hook.remove)
        with self.assertRaisesRegex(ValueError, "rlt_switch_flags"):
            loaded.predict_action_chunk(observation(), SamplingOptions())
        before = loaded.predict_action_chunk(flagged(False), SamplingOptions(deterministic=True))
        self.assertTrue((before.actions == 0).all())
        self.assertFalse(before.aux["actor_switch"][0])
        after = loaded.predict_action_chunk(flagged(True), SamplingOptions(deterministic=True))
        self.assertTrue(after.aux["actor_switch"][0])
        self.assertEqual(calls, [1])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_config(self, action=2, latent=4, proprio=3):
        source = self.root / f"source-{action}"
        asset = self.root / f"asset-{action}.bin"
        asset.write_bytes(b"provider-owned-asset")
        cfg = _wrap(
            {
                "schema_version": 1,
                "task": "train",
                "model": {
                    "target": "tests.algorithms.rltoken.fakes:TinyFeatureModel",
                    "asset": str(asset),
                    "state_dim": 2,
                    "action_dim": action,
                    "action_horizon": 3,
                    "latent_dim": latent,
                    "proprio_dim": proprio,
                },
                "data": {},
                "train": {},
                "runtime": {"device": "cpu"},
            }
        )
        model = TinyFeatureModel(cfg, mode="infer")
        write_model_bundle(source, model, cfg)
        return _wrap(
            {
                "schema_version": 1,
                "task": "train",
                "model": {
                    "target": "dobot_nexus.algorithms.rltoken.policy:RLTokenPolicy",
                    "state_dim": 2,
                    "action_dim": action,
                    "action_horizon": 2,
                    "rltoken": {"chunk_size": 2, "hidden_dims": [8], "fixed_std": 0.1},
                    "features": {"artifact": str(source)},
                },
                "data": {},
                "train": {},
                "runtime": {"device": "cpu"},
            }
        )

    def test_unseeded_actor_sampling_respects_torch_manual_seed(self):
        config = self.make_config()
        for residual in (None, {"hidden_dims": [8], "exploration_std": 0.1}):
            config.model.rltoken.residual = residual
            policy = RLTokenPolicy(config, mode="infer")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                first = policy.predict_action_chunk(observation(), SamplingOptions()).actions
                torch.manual_seed(42)
                second = policy.predict_action_chunk(observation(), SamplingOptions()).actions
            self.assertTrue((first == second).all())

    def test_residual_deployment_preserves_reference_and_sampler_config(self):
        for latent_dim in (None, 4):
            config = self.make_config()
            config.algorithm = _wrap(
                dict(
                    target="dobot_nexus.algorithms.rltoken.training:RLTokenTrainer",
                    reference_num_steps=17,
                )
            )
            config.model.rltoken.residual = _wrap(
                dict(hidden_dims=[8], latent_dim=latent_dim, exploration_std=0)
            )
            policy = RLTokenPolicy(config, mode="infer")
            with torch.no_grad():
                policy.actor.network[-1].weight.fill_(0.1)
            before = policy.predict_action_chunk(
                observation(), SamplingOptions(deterministic=True, seed=4)
            )
            self.assertEqual(before.actions.shape, (1, 2, 2))
            artifact = self.root / "residual"
            write_model_bundle(artifact, policy, config)
            moved = self.root / f"moved-residual-{latent_dim}"
            artifact.rename(moved)
            shutil.rmtree(config.model.features.artifact)
            cfg = _wrap(
                dict(
                    schema_version=1,
                    task="infer",
                    model=dict(target=config.model.target, artifact=str(moved)),
                    runtime=dict(device="cpu"),
                )
            )
            loaded = InferencePolicy(cfg)
            after = loaded.predict_chunk(observation(), SamplingOptions(deterministic=True, seed=4))
            torch.testing.assert_close(
                torch.from_numpy(before.actions), torch.from_numpy(after.actions)
            )
            self.assertEqual(loaded.model.config.algorithm.reference_num_steps, 17)
            self.assertEqual(after.actions.dtype.name, "float32")
            if latent_dim:
                other = loaded.predict_chunk(
                    observation(), SamplingOptions(deterministic=True, seed=9)
                )
                self.assertFalse((after.actions == other.actions).all())

    def test_two_providers_frozen_and_deploy_without_source_artifact(self):
        for action, latent, proprio in ((2, 4, 3), (3, 7, 2)):
            config = self.make_config(action, latent, proprio)
            policy = RLTokenPolicy(config, mode="infer")
            original = content_digest(policy.provider.state_dict())
            policy.train()
            self.assertFalse(policy.provider.training)
            self.assertTrue(all(not p.requires_grad for p in policy.provider.parameters()))
            expected = policy.predict_action_chunk(
                observation(), SamplingOptions(seed=4, deterministic=True)
            )
            self.assertEqual(expected.actions.shape, (1, 2, action))
            self.assertEqual(original, content_digest(policy.provider.state_dict()))
            artifact = self.root / f"deploy-{action}"
            write_model_bundle(artifact, policy, config)
            moved = self.root / f"moved-{action}"
            artifact.rename(moved)
            shutil.rmtree(config.model.features.artifact)
            inference = _wrap(
                {
                    "schema_version": 1,
                    "task": "infer",
                    "model": {"target": config.model.target, "artifact": str(moved)},
                    "runtime": {"device": "cpu"},
                }
            )
            loaded = InferencePolicy(inference)
            actual = loaded.predict_chunk(
                observation(), SamplingOptions(seed=4, deterministic=True)
            )
            torch.testing.assert_close(
                torch.from_numpy(actual.actions), torch.from_numpy(expected.actions)
            )
            self.assertFalse((moved / "deployment/nexus").exists())
            shutil.rmtree(moved / "deployment/features")
            with self.assertRaisesRegex(FileNotFoundError, "feature manifest"):
                InferencePolicy(inference)

    def test_learner_does_not_construct_provider_or_offline_loader(self):
        policy = RLTokenPolicy(self.make_config(), mode="train")
        self.assertFalse(hasattr(policy, "provider"))
        with self.assertRaisesRegex(RuntimeError, "RLTokenTrainer"):
            policy.build_train_dataloader(None)

    def test_retained_provider_can_export_after_original_assets_are_removed(self):
        config = self.make_config()
        original = Path(config.model.features.artifact)
        retained = retain_feature_bundle(original, self.root / "retained")
        config.model.features.artifact = str(retained)
        policy = RLTokenPolicy(config, mode="infer")
        shutil.rmtree(original)
        (self.root / "asset-2.bin").unlink()
        deployed = write_model_bundle(self.root / "deployed", policy, config)
        shutil.rmtree(retained)
        infer = _wrap(
            {
                "schema_version": 1,
                "task": "infer",
                "model": {"target": config.model.target, "artifact": str(deployed)},
                "runtime": {"device": "cpu"},
            }
        )
        loaded = InferencePolicy(infer)
        self.assertEqual(
            Path(loaded.model.provider.config.model.asset).read_bytes(), b"provider-owned-asset"
        )
