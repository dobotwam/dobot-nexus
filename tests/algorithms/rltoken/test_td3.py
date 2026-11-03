import copy
import tempfile
import unittest
from pathlib import Path

import torch

from dobot_nexus.algorithms.rltoken.actor_critic import build_actor
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig, validate_recipe
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.learner import create_learner
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.transitions import residual_batch
from .fakes import feature_spec, transition


class TD3Tests(unittest.TestCase):
    def test_td_target_uses_reference_without_smoothing_outside_actor_phase(self):
        rows = [transition(i, next_actor_enabled=enabled) for i, enabled in enumerate((False, True))]
        batch = residual_batch(rows, 2, "cpu")
        seen, actor_batches = [], []
        target_hook = self.learner.target.register_forward_pre_hook(
            lambda module, args: seen.append(args[2].clone()))
        actor_hook = self.learner.target_actor.register_forward_pre_hook(
            lambda module, args: actor_batches.append(len(args[0])))
        self.addCleanup(target_hook.remove)
        self.addCleanup(actor_hook.remove)
        self.learner.target_values(batch)
        self.assertEqual(actor_batches, [1])
        torch.testing.assert_close(seen[0][0], batch["next_reference"][0, :2])
        self.assertGreater(float(seen[0][1].abs().max()), 0.)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spec = feature_spec()
        self.heads = HeadConfig(2, residual=dict(rho=[.1, .2], actor_width=8, critic_width=8))
        self.config = LearningConfig(batch_size=2, warmup=1, critic_actor_ratio=2,
                                     td3=dict(target_noise=0.02), seed=7)
        self.replay = Replay(Path(self.tmp.name)/'journal', self.spec, 2, capacity=10,
                             sampling='demo_joint_50', demo_capacity=10, allow_terminal_chunks=True)
        self.addCleanup(self.replay.close)
        self.learner = create_learner(self.spec, self.heads, self.config, self.replay)
        self.rows = [transition(i, origin='demo', terminated=True, rewards=torch.tensor([0., float(i)]))
                     for i in range(2)]

    def test_recipe_rejects_mixed_legacy_and_td3(self):
        with self.assertRaises(ValueError):
            validate_recipe('rlt_ac_fixed_v1', self.heads, self.config)

    def test_reference_immutable_and_residual_bounded(self):
        before = content_digest(self.rows)
        batch = residual_batch(self.rows, 2, 'cpu')
        self.assertTrue(batch['reference_condition'].eq(0).all())
        actor = self.learner.actor
        output = actor(batch['latent'], batch['state'], batch['reference_condition'], deterministic=True)
        self.assertTrue(output.eq(0).all())
        target = actor.project(batch['supervision'], batch['reference_condition'])
        torch.testing.assert_close(target[0,0], torch.tensor([.1, .2]))
        self.learner.train_rows(self.rows)
        self.assertEqual(content_digest(self.rows), before)

    def test_strict_base_freeze_and_delayed_targets(self):
        learner = self.learner
        base = content_digest(learner.critic.base.state_dict())
        target_base = content_digest(learner.target.base.state_dict())
        actor_target = content_digest(learner.target_actor.state_dict())
        learner.train_rows(self.rows)
        self.assertEqual(learner.actor_updates, 0)
        self.assertEqual(content_digest(learner.target_actor.state_dict()), actor_target)
        learner.train_rows(self.rows)
        self.assertEqual(learner.actor_updates, 1)
        self.assertEqual(content_digest(learner.critic.base.state_dict()), base)
        self.assertEqual(content_digest(learner.target.base.state_dict()), target_base)
        self.assertTrue(all(p.grad is None for p in learner.critic.parameters()))
        self.assertTrue(all(not p.requires_grad for p in learner.target.parameters()))

    def test_bc_critic_only_and_next_update_resume(self):
        learner = self.learner
        learner.set_phase('actor_bc')
        critic = content_digest(learner.critic.state_dict())
        learner.train_rows(self.rows)
        self.assertEqual(critic, content_digest(learner.critic.state_dict()))
        self.assertGreater(learner.actor_revision, 0)
        learner.set_phase('residual_critic')
        actor = content_digest(learner.actor.state_dict())
        target_actor = content_digest(learner.target_actor.state_dict())
        learner.train_rows(self.rows)
        self.assertEqual(actor, content_digest(learner.actor.state_dict()))
        self.assertEqual(target_actor, content_digest(learner.target_actor.state_dict()))
        learner.set_phase('offline_joint')
        learner.train_rows(self.rows)
        state = learner.state_dict()
        learner.train_rows(self.rows)
        expected = content_digest(learner.state_dict())
        learner.load_state_dict(state)
        learner.train_rows(self.rows)
        self.assertEqual(expected, content_digest(learner.state_dict()))

    def test_pool_retention_and_offline_no_online_budget(self):
        for row in self.rows:
            self.replay.commit(row)
            self.replay.commit(row)
        self.assertEqual(self.replay.budget, 0)
        self.replay.capacity = 1
        for i in range(2, 6):
            self.replay.commit(transition(i, origin='online'))
        self.assertEqual(self.replay.budget, 4)
        self.assertEqual(sum(r.origin == 'demo' for r, _ in self.replay.rows.values()), 2)
        samples = self.replay.sample(10)
        self.assertEqual(sum(r.origin == 'demo' for r in samples), 5)

    def test_chunk_discount_terminal_and_base_bootstrap(self):
        self.assertAlmostEqual(self.learner.gamma_step ** 2, self.config.gamma)
        batch = residual_batch(self.rows, 2, 'cpu')
        target = self.learner.target_values(batch)
        torch.testing.assert_close(target, torch.tensor([0., self.learner.gamma_step]))
        self.learner.set_phase('base_critic')
        initial = content_digest(self.learner.critic.base.state_dict())
        self.learner.train_rows(self.rows)
        self.assertNotEqual(initial, content_digest(self.learner.critic.base.state_dict()))

    def test_offset_zero_correction_at_reference(self):
        batch = residual_batch(self.rows, 2, 'cpu')
        base, residual = self.learner.critic.components(batch['latent'], batch['state'],
            batch['reference_condition'], batch['reference_condition'])
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        with torch.no_grad():
            self.learner.critic.offset[0][-1].bias.fill_(.1)
        _, residual = self.learner.critic.components(batch['latent'], batch['state'],
            batch['reference_condition'], batch['reference_condition'])
        self.assertTrue((residual[:,0] > 0).all())

    def test_auxiliary_losses_skip_failure_expert(self):
        from dataclasses import replace
        config = replace(self.config, td3=dict(direction_weight=.01, rank_weight=.1))
        learner = create_learner(self.spec, self.heads, config, self.replay)
        metrics = learner.train_rows(self.rows)
        self.assertGreaterEqual(metrics["direction_loss"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["critic_loss"])))
        from dobot_nexus.algorithms.rltoken.auxiliary import direction_loss
        batch = residual_batch([self.rows[0]], 2, "cpu")
        self.assertEqual(float(direction_loss(learner.critic, learner.actor, batch)), 0.)

    def test_bottleneck_does_not_read_teacher(self):
        from dobot_nexus.algorithms.rltoken.token import TokenConfig, build_token_modules
        config = TokenConfig(8, width=8, heads=2, max_tokens=4, mode='bottleneck', dropout=0)
        encoder, decoder = build_token_modules(config)
        encoder.eval(); decoder.eval()
        prefix = torch.randn(2, 4, 8)
        valid = torch.ones(2, 4, dtype=torch.bool)
        latent = encoder(prefix, valid)
        torch.testing.assert_close(decoder(latent, prefix, valid), decoder(latent, prefix + 100, valid))
        self.assertFalse(torch.equal(decoder(latent, prefix, valid), decoder(latent * 0, prefix, valid)))
