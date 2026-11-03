import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dobot_nexus.algorithms.rltoken.artifacts import load_provider
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.learner import create_learner
from dobot_nexus.algorithms.rltoken.offline_data import Episode, build_cache, load_cache
from dobot_nexus.algorithms.rltoken.pretraining import RLTokenPretrainTrainer, initialize_from_stage
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.transitions import Finalization
from dobot_nexus.core.config import _wrap
from .fakes import observation, transition, feature_spec
from . import test_policy


class OfflineTest(unittest.TestCase):
    def test_diagnostic_validation_records_policy_and_rejects_nonfinite_loss(self):
        from unittest.mock import patch
        factory = test_policy.PolicyTest()
        factory.root = self.root
        cfg = factory.make_config()
        provider, _, spec = load_provider(cfg.model.features.artifact)
        episodes = [Episode(str(i), [observation()] * 3, torch.zeros(2, 2),
                            torch.tensor([0., 1.]), torch.zeros(2, dtype=torch.bool),
                            'demo', True) for i in range(2)]
        cache = build_cache(provider, episodes, self.root/'cache', chunk_size=2,
                            dataset_version='v1', reward_definition='success', gamma_step=.99 ** .5)
        cfg.model.rltoken.residual = _wrap(dict(rho=[.1, .1], actor_width=8, critic_width=8))
        cfg.algorithm = _wrap(dict(recipe='rlt_td3_dual_residual_v1', learning=dict(
            batch_size=2, warmup=1, critic_actor_ratio=2, td3={}), pretrain=dict(
            phase='actor_bc', task_identity='success', initialize={})))
        cfg.data = _wrap(dict(cache_path=str(cache), replay=dict(capacity=100, sampling='demo_joint_50')))
        cfg.train = _wrap(dict(max_steps=1, save_every=1))
        for policy, score in [('loss_threshold', 2.), ('diagnostic', 2.), ('diagnostic', float('nan'))]:
            with self.subTest(policy=policy, score=score):
                cfg.algorithm.pretrain.validation_policy = policy
                cfg.algorithm.pretrain.pop('max_validation_loss', None)
                if policy == 'loss_threshold':
                    cfg.algorithm.pretrain.max_validation_loss = 0.
                run = self.root/f'{policy}-{score}'
                experiment = SimpleNamespace(run_dir=run, logger=logging.getLogger('offline-test'))
                trainer = RLTokenPretrainTrainer(cfg, experiment)
                with patch('dobot_nexus.algorithms.rltoken.pretraining.validation_loss', return_value=score):
                    if score != score:
                        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite offline validation'):
                            trainer.run()
                        self.assertFalse((run/'rltoken/best.pt').exists())
                        continue
                    trainer.run()
                path = run/'rltoken/best.pt'
                payload = torch.load(path, weights_only=False)['payload']
                self.assertEqual(payload['validation']['loss'], score)
                self.assertEqual(payload['validation']['policy'], policy)
                self.assertEqual(payload['accepted'], policy == 'diagnostic')
                replay = Replay(run/'online', spec, 2, capacity=100)
                try:
                    learner = create_learner(spec, HeadConfig(**cfg.model.rltoken),
                                             LearningConfig(**cfg.algorithm.learning), replay)
                    if policy == 'diagnostic':
                        initialize_from_stage(learner, path, task_identity='success', component='actor')
                        self.assertTrue(learner.pretrained_ready)
                    else:
                        with self.assertRaisesRegex(ValueError, 'acceptance mismatch'):
                            initialize_from_stage(learner, path, task_identity='success', component='actor')
                finally:
                    replay.close()

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

    def test_cache_split_mc_mask_and_four_offline_phases(self):
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
            if row.observation_id.episode_id == '1':
                self.assertTrue(row.executed.actions.eq(.1).all())
                self.assertTrue(row.supervision.eq(.2).all())
                self.assertTrue(row.features.reference_actions.eq(0).all())
            if row.observation_id.episode_id == '7':
                self.assertIsNone(row.mc_return)
        with self.assertRaisesRegex(ValueError, 'identity'):
            load_cache(cache_path, spec, chunk_size=2, reward_definition='wrong')
        cfg.model.rltoken.residual = _wrap(dict(rho=[.1, .1], actor_width=8, critic_width=8))
        cfg.algorithm = _wrap(dict(recipe='rlt_td3_dual_residual_v1', learning=dict(
            batch_size=2, warmup=1, critic_actor_ratio=2, td3={}), pretrain=dict(
            phase='actor_bc', task_identity='success', max_validation_loss=100., initialize={})))
        cfg.data = _wrap(dict(cache_path=str(cache_path), replay=dict(capacity=100, sampling='demo_joint_50')))
        cfg.train = _wrap(dict(max_steps=2, save_every=1))
        paths = {}
        for phase in ('actor_bc', 'base_critic', 'residual_critic', 'offline_joint'):
            cfg.algorithm.pretrain.phase = phase
            if phase == 'residual_critic':
                cfg.algorithm.pretrain.initialize = _wrap(dict(actor=paths['actor_bc'], base=paths['base_critic']))
            elif phase == 'offline_joint':
                cfg.algorithm.pretrain.initialize = _wrap(dict(all=paths['residual_critic']))
            else:
                cfg.algorithm.pretrain.initialize = {}
            run = self.root/phase
            experiment = SimpleNamespace(run_dir=run, logger=logging.getLogger('offline-test'))
            result = RLTokenPretrainTrainer(cfg, experiment).run()
            self.assertEqual(result.global_step, 2)
            paths[phase] = str(run/'rltoken/best.pt')
            self.assertTrue(Path(paths[phase]).is_file())
            if phase == "offline_joint":
                cfg.algorithm.pretrain.initialize = {}
                cfg.train.resume_from = result.checkpoint
                cfg.train.max_steps = 3
                resumed = RLTokenPretrainTrainer(cfg, experiment).run()
                self.assertEqual(resumed.global_step, 3)
        replay = Replay(self.root/'online', spec, 2, capacity=100)
        self.addCleanup(replay.close)
        learner = create_learner(spec, HeadConfig(**cfg.model.rltoken),
                                 LearningConfig(**cfg.algorithm.learning), replay)
        initialize_from_stage(learner, paths['offline_joint'], task_identity='success')
        self.assertTrue(learner.pretrained_ready)
        self.assertEqual(learner.actor_updates, 0)
        self.assertGreater(learner.ready_revision, 0)
        self.assertEqual(replay.budget, 0)
        scoped_replay = Replay(self.root / 'critical-online', spec, 2, capacity=100,
                               training_scope='critical_phase')
        self.addCleanup(scoped_replay.close)
        scoped_learner = create_learner(spec, HeadConfig(**cfg.model.rltoken),
                                        LearningConfig(**cfg.algorithm.learning), scoped_replay)
        with self.assertRaisesRegex(ValueError, 'Stage artifact identity'):
            initialize_from_stage(scoped_learner, paths['offline_joint'], task_identity='success')

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
