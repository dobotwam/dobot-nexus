from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dobot_nexus.algorithms.rltoken.artifacts import load_provider
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.learner import create_learner
from dobot_nexus.algorithms.rltoken.offline_data import Episode, build_cache, load_cache
from dobot_nexus.algorithms.rltoken.pretraining import (
    RLTokenPrewarmTrainer,
    initialize_from_prewarm,
)
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.transitions import Finalization
from dobot_nexus.core.config import _wrap

from . import test_policy
from .fakes import feature_spec, observation, transition


class OfflineTest(unittest.TestCase):
    def test_cache_reuses_recorded_reference_seeds(self):
        from unittest.mock import patch
        factory = test_policy.PolicyTest()
        factory.root = self.root
        cfg = factory.make_config()
        provider, _, _ = load_provider(cfg.model.features.artifact)
        episodes = [Episode(str(i), [observation()] * 5, torch.zeros(4, 2),
                            torch.zeros(4), torch.zeros(4, dtype=torch.bool), 'vla', True,
                            reference_seeds={0:42,2:43}) for i in range(2)]
        with patch.object(provider, 'extract_rl_features', wraps=provider.extract_rl_features) as extract:
            build_cache(provider, episodes, self.root/'recorded-noise', chunk_size=2,
                        dataset_version='v1', reward_definition='success', gamma_step=.99)
        self.assertEqual([call.args[1].seed for call in extract.call_args_list], [42,43,42,43])

    def test_cache_reads_only_required_chunk_observations_once(self):
        factory = test_policy.PolicyTest()
        factory.root = self.root
        provider, _, _ = load_provider(factory.make_config().model.features.artifact)
        accessed = []
        class Observations:
            def __len__(self):
                return 5
            def __getitem__(self, index):
                if not 0 <= index < len(self):
                    raise IndexError(index)
                accessed.append(index)
                return observation()
        episodes = [Episode(str(i), Observations(), torch.zeros(4, 2), torch.zeros(4),
                            torch.zeros(4, dtype=torch.bool), 'vla', False) for i in range(2)]
        build_cache(provider, episodes, self.root/'lazy-cache', chunk_size=2,
                    dataset_version='v1', reward_definition='success', gamma_step=.99)
        # Retain the actual final observation for a truncated chunk; no interior image reads.
        self.assertEqual(accessed, [0, 2, 4, 0, 2, 4])

    def test_cache_filters_phase_and_rejects_different_training_scope(self):
        from .test_routing import SWITCH, flagged
        factory = test_policy.PolicyTest()
        factory.root = self.root
        cfg = factory.make_config()
        provider, _, spec = load_provider(cfg.model.features.artifact)
        episodes = [Episode(str(i), [flagged(False), flagged(False), flagged(True), flagged(True), flagged(True)],
                            torch.full((4, 2), .1), torch.tensor([0., 0., 0., 1.]),
                            torch.zeros(4, dtype=torch.bool), "vla", True) for i in range(4)]
        kwargs = dict(chunk_size=2, dataset_version="v1", reward_definition="success", gamma_step=.99)
        path = build_cache(provider, episodes, self.root / "phases", policy_switch=SWITCH, **kwargs)
        cache = load_cache(path, spec, chunk_size=2, reward_definition="success", policy_switch=SWITCH)
        rows = cache["train"] + cache["validation"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r.observation_id.observation_id == "2" and r.critical_phase for r in rows))
        for row in rows:
            self.assertAlmostEqual(row.mc_return, .99)
        with self.assertRaisesRegex(ValueError, "policy_switch identity"):
            load_cache(path, spec, chunk_size=2, reward_definition="success")
        episodes[0].observations[0] = observation()
        with self.assertRaisesRegex(ValueError, "rlt_switch_flags"):
            build_cache(provider, episodes, self.root / "missing", policy_switch=SWITCH, **kwargs)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cache_held_out_episodes_and_execution_masks(self):
        factory = test_policy.PolicyTest()
        factory.root = self.root
        cfg = factory.make_config()
        provider, _, spec = load_provider(cfg.model.features.artifact)
        episodes = [Episode(str(i), [observation()] * 5, torch.full((4, 2), .1),
                            torch.tensor([0., 0., 0., 1.]), torch.zeros(4, dtype=torch.bool),
                            'demo' if i % 2 else 'vla', True) for i in range(8)]
        episodes[-1].terminated = False
        episodes[1].supervision_commands = torch.full((4, 2), .2)
        episodes[1].supervision_mask = torch.ones(4, dtype=torch.bool)
        episodes[1].timestamps = torch.arange(5, dtype=torch.float64) * spec.control_dt
        cache_path = build_cache(provider, episodes, self.root/'cache', chunk_size=2,
            dataset_version='v1', reward_definition='success', gamma_step=.99 ** .5, seed=4)
        cache = load_cache(cache_path, spec, chunk_size=2, reward_definition='success', gamma_step=.99 ** .5)
        train_ids = {r.observation_id.episode_id for r in cache['train']}
        val_ids = {r.observation_id.episode_id for r in cache['validation']}
        self.assertFalse(train_ids & val_ids)
        for row in cache['train'] + cache['validation']:
            self.assertIsNone(row.episode_success)
            if row.observation_id.episode_id == '1':
                self.assertTrue(row.executed.actions.eq(.1).all())
                self.assertTrue(row.supervision.eq(.2).all())
                self.assertTrue(row.features.reference_actions.eq(0).all())
            if row.observation_id.episode_id == '7':
                self.assertIsNone(row.mc_return)
        with self.assertRaisesRegex(ValueError, 'identity'):
            load_cache(cache_path, spec, chunk_size=2, reward_definition='wrong')

        cache["metadata"]["format"] = "rltoken_episode_cache_v1"
        cache["identity"] = content_digest(cache["metadata"])
        torch.save(cache, cache_path / "cache.pt")
        manifest = dict(
            cache["metadata"], identity=cache["identity"], payload_digest=content_digest(cache)
        )
        (cache_path / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "cache format"):
            load_cache(cache_path, spec, chunk_size=2, reward_definition="success")

    def test_joint_prewarm_handoff_and_resume_with_single_episode(self):
        factory = test_policy.PolicyTest()
        factory.root = self.root
        cfg = factory.make_config()
        provider, _, spec = load_provider(cfg.model.features.artifact)
        episode = Episode(
            "only",
            [observation()] * 5,
            torch.full((4, 2), 0.25),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.zeros(4, dtype=torch.bool),
            "demo",
            True,
            success=True,
        )
        kwargs = dict(
            chunk_size=2, dataset_version="v1", reward_definition="success", gamma_step=0.99
        )
        with self.assertRaisesRegex(ValueError, "two episodes"):
            build_cache(provider, [episode], self.root / "split", validation_fraction=0.2, **kwargs)
        path = build_cache(
            provider, [episode], self.root / "cache", validation_fraction=0.0, **kwargs
        )
        cache = load_cache(path, spec, chunk_size=2, reward_definition="success", gamma_step=0.99)
        self.assertEqual(len(cache["train"]), 2)
        self.assertEqual(cache["validation"], [])
        cfg.model.rltoken.residual = {"hidden_dims": [8]}
        cfg.algorithm = _wrap(
            dict(
                target="dobot_nexus.algorithms.rltoken.training:RLTokenTrainer",
                recipe="rlt_td3_residual_v1",
                task_identity="success",
                learning=dict(
                    batch_size=2,
                    warmup=1,
                    critic_actor_ratio=2,
                    td3={"bc_action_space": "postclip"},
                ),
            )
        )
        cfg.data = _wrap(dict(cache_path=str(path), replay=dict(capacity=10)))
        replay = Replay(self.root / "online", spec, 2, capacity=10)
        self.addCleanup(replay.close)
        heads = HeadConfig(**cfg.model.rltoken)
        learning = LearningConfig(**cfg.algorithm.learning)
        learner = create_learner(spec, heads, learning, replay)
        cfg.train = _wrap(dict(max_steps=1, save_every=1))
        experiment = SimpleNamespace(run_dir=self.root, logger=logging.getLogger("prewarm-test"))
        from dobot_nexus.algorithms.rltoken.checkpoint import load_resume, save_resume

        with (
            patch.object(Replay, "sample", side_effect=AssertionError("deep sample")),
            patch.object(type(learner), "state_dict", side_effect=AssertionError("deep snapshot")),
            patch("dobot_nexus.algorithms.rltoken.pretraining.save_resume",
                  wraps=save_resume) as checkpoint_writer,
        ):
            result = RLTokenPrewarmTrainer(cfg, experiment).run()
        prewarm = self.root / "rltoken/prewarm.pt"
        saved = torch.load(prewarm, weights_only=False)["payload"]["state"]
        self.assertEqual(result.global_step, 1)
        self.assertEqual(saved["actor_updates"], 0)
        with self.assertRaisesRegex(ValueError, "actor update"):
            initialize_from_prewarm(learner, prewarm, task_identity="success")
        first_checkpoint = Path(result.checkpoint)
        first_manifest = json.loads((first_checkpoint / "manifest.json").read_text())
        self.assertEqual(
            checkpoint_writer.call_args.kwargs["feature_manifest_sha256"],
            first_manifest["feature_manifest_sha256"],
        )
        fingerprint = first_manifest["task_fingerprint"]
        cfg.train.resume_from = result.checkpoint
        cfg.train.keep_last_checkpoints = 2
        cfg.train.max_steps = 3
        result = RLTokenPrewarmTrainer(cfg, experiment).run()
        self.assertEqual(result.global_step, 3)
        checkpoints = sorted(first_checkpoint.parent.glob("step-*"))
        self.assertEqual(len(checkpoints), 2)
        self.assertNotIn(first_checkpoint, checkpoints)
        for path in checkpoints:
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["task_fingerprint"], fingerprint)
            self.assertTrue((path / manifest["feature_manifest"]).is_file())
        state = torch.load(prewarm, weights_only=False)["payload"]["state"]
        initialize_from_prewarm(learner, prewarm, task_identity="success")
        self.assertEqual(learner.update_step, 3)
        self.assertEqual((learner.actor_updates, learner.critic_updates, replay.budget), (0, 0, 0))
        self.assertTrue(learner.pretrained_ready)
        self.assertEqual(learner.ready_revision, learner.actor_revision)
        for key in (
            "actor",
            "critic",
            "target",
            "target_actor",
            "actor_optimizer",
            "critic_optimizer",
            "actor_scheduler",
            "critic_scheduler",
        ):
            self.assertEqual(
                content_digest(getattr(learner, key).state_dict()), content_digest(state[key])
            )
        learner.update(incoming=[transition(7, spec=spec)])
        self.assertEqual((learner.update_step, learner.actor_updates), (4, 1))
        from dobot_nexus.algorithms.rltoken.artifacts import write_model_bundle
        from dobot_nexus.algorithms.rltoken.policy import RLTokenPolicy
        from dobot_nexus.core.contracts import SamplingOptions
        from dobot_nexus.inference.policy import InferencePolicy

        learner.update(0, incoming=[transition(8, spec=spec), transition(9, spec=spec)])
        checkpoint = save_resume(
            self.root / "checkpoints", learner, task_fingerprint="online", run_id="run"
        )
        expected = learner.update(2)
        digest = content_digest(learner.state_dict())
        restored = create_learner(spec, heads, learning, replay)
        load_resume(checkpoint, restored, task_fingerprint="online")
        self.assertEqual(restored.update(2), expected)
        self.assertEqual(content_digest(restored.state_dict()), digest)
        policy = RLTokenPolicy(cfg, mode="infer")
        policy.actor.load_state_dict(restored.actor.state_dict(), strict=True)
        artifact = write_model_bundle(self.root / "deployment", policy, cfg)
        infer = _wrap(
            dict(
                schema_version=1,
                task="infer",
                model=dict(target=cfg.model.target, artifact=str(artifact)),
                runtime=dict(device="cpu"),
            )
        )
        options = SamplingOptions(deterministic=True)
        output = InferencePolicy(infer).predict_chunk(observation(), options)
        self.assertEqual(output.actions.dtype.name, "float32")
        self.assertEqual(output.actions.shape, (1, 2, 2))
        self.assertTrue(
            (output.actions == policy.predict_action_chunk(observation(), options).actions).all()
        )
        fresh = create_learner(spec, heads, learning, replay)
        initialize_from_prewarm(fresh, prewarm, task_identity="success", inherit_optimizer=False)
        self.assertEqual(fresh.update_step, 0)
        self.assertFalse(fresh.critic_optimizer.state)
        from dataclasses import replace

        for field, value in (("gamma", 0.95), ("actor_lr", 0.001), ("critic_lr", 0.001)):
            other = create_learner(spec, heads, replace(learning, **{field: value}), replay)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                initialize_from_prewarm(other, prewarm, task_identity="success")

        for changed_heads, td3 in (
            ({}, {}),
            ({"critic_output": "sigmoid"}, {"critic_objective": "bce"}),
            ({"latent_dim": 4, "exploration_std": 0}, {"actor_update_action_noise": False}),
        ):
            changed = replace(heads, residual={**heads.residual, **changed_heads})
            changed_learning = replace(learning, td3=td3)
            wrong = create_learner(spec, changed, changed_learning, replay)
            with self.assertRaisesRegex(ValueError, "Prewarm"):
                initialize_from_prewarm(wrong, prewarm, task_identity="success")
        scoped = Replay(self.root / "scoped", spec, 2, training_scope="critical_phase")
        self.addCleanup(scoped.close)
        with self.assertRaisesRegex(ValueError, "training_scope"):
            initialize_from_prewarm(
                create_learner(spec, heads, learning, scoped), prewarm, task_identity="success"
            )

        for path in reversed(checkpoints):
            cfg.train.resume_from = str(path)
            cfg.train.max_steps = int(path.name.split("-")[1])
            self.assertEqual(RLTokenPrewarmTrainer(cfg, experiment).run().checkpoint, str(path))
        cfg.train.max_steps = 3
        cfg.train.keep_last_checkpoints = 1
        last = Path(RLTokenPrewarmTrainer(cfg, experiment).run().checkpoint)
        self.assertEqual(list(last.parent.glob("step-*")), [last])

    def test_finalization_is_durable_idempotent_and_rejects_unknown(self):
        replay = Replay(self.root/'journal', feature_spec(), 2)
        self.addCleanup(replay.close)
        row = transition(rewards=None, accepted=None, reward_finalized=False)
        self.assertEqual(replay.commit(row).status, 'pending')
        self.assertEqual(replay.budget, 0)
        label = Finalization(row.key, torch.tensor([0., 1.]), True)
        first = replay.commit(label)
        self.assertEqual(replay.commit(label), first)
        self.assertEqual(replay.budget, 1)
        self.assertFalse(replay.pending())
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            replay.commit(Finalization(row.key, torch.zeros(2), True))
        with self.assertRaisesRegex(ValueError, 'unknown'):
            replay.commit(Finalization(('unknown',), torch.zeros(2), True))

    def test_shared_episode_indices_and_image_conversion(self):
        import numpy as np

        from dobot_nexus.data.images import image_uint8
        from dobot_nexus.data.lerobot.dataset import episode_indices
        source = SimpleNamespace(dataset=SimpleNamespace(episode_bounds={0:(0,4), 1:(4,9)}))
        self.assertEqual(episode_indices(source, [1]), list(range(4,9)))
        with self.assertRaises(ValueError):
            episode_indices(source, [2])
        self.assertEqual(image_uint8(torch.ones(3,8,8)).shape, (8,8,3))
        class Inline:
            dataset_root = self.root
            def __len__(self): return 9
        (self.root/"meta").mkdir()
        (self.root/"meta/episodes.jsonl").write_text('{"episode_index":0,"length":4}\n{"episode_index":1,"length":5}\n')
        self.assertEqual(episode_indices(Inline(), [1]), list(range(4,9)))
        with self.assertRaises(ValueError):
            image_uint8(np.full((8,8,3), np.nan))
