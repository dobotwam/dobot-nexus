from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from dobot_nexus.algorithms.rltoken.checkpoint import load_resume, save_resume
from dobot_nexus.algorithms.rltoken.configuration import (
    HeadConfig,
    LearningConfig,
    ResidualConfig,
    TD3Config,
    WeightScheduleConfig,
    validate_recipe,
)
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.learner import create_learner
from dobot_nexus.algorithms.rltoken.pretraining import validation_loss
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.td3 import legal_success_behavior_target
from dobot_nexus.algorithms.rltoken.transitions import residual_batch

from .fakes import feature_spec, transition


class ActionScore(nn.Module):
    def forward(self, latent, state, actions):
        score = actions.flatten(1).mean(-1)
        return torch.stack((2 - score, score), dim=-1)


class TD3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = feature_spec()
        self.replay = Replay(
            self.root / "journal", self.spec, 2, capacity=10, allow_terminal_chunks=True
        )
        self.addCleanup(self.replay.close)
        self.rows = [transition(0), transition(1, terminated=True, episode_success=True)]
        self.batch = residual_batch(self.rows, 2, "cpu")

    def learner(self, residual=None, td3=None, **learning):
        heads = HeadConfig(2, residual={"hidden_dims": [8], **(residual or {})})
        config = LearningConfig(
            batch_size=2, warmup=1, critic_actor_ratio=2, seed=7, td3=td3 or {}, **learning
        )
        return create_learner(self.spec, heads, config, self.replay)

    def test_config_rejects_incompatible_strategies(self):
        with self.assertRaisesRegex(ValueError, "reference_dropout"):
            self.learner(reference_dropout=0.1)
        for residual, td3 in (
            (dict(latent_dim=3), dict(actor_update_action_noise=False)),
            (dict(latent_dim=3, exploration_std=0), {}),
            (dict(critic_output="sigmoid"), {}),
            ({}, dict(critic_objective="bce")),
            (dict(critic_output="sigmoid"), dict(critic_objective="bce", target_q_clip=[0, 1])),
        ):
            with self.subTest(residual=residual, td3=td3), self.assertRaises(ValueError):
                self.learner(residual, td3)
        learner = self.learner()
        with self.assertRaises(ValueError):
            validate_recipe("rlt_ac_fixed_v1", learner.heads, learner.config)
        for values in (
            {"hidden_dims": []},
            {"scale": []},
            {"scale": float("nan")},
            {"scale": [1, 0]},
            {"eval_latent_mode": "mean"},
            {"latent_dim": False},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ResidualConfig(**values)
        for values in ({"target_q_clip": [2, 1]}, {"rank_temperature": 0}, {"grad_clip": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TD3Config(**values)

    def test_scale_broadcast_and_reference_detach(self):
        for scale, expected in [(1, [1] * 4), ([1, 2], [1, 2] * 2), ([1, 2, 3, 4], [1, 2, 3, 4])]:
            actor = self.learner(dict(scale=scale)).actor
            torch.testing.assert_close(actor.scale, torch.tensor(expected).reshape(1, 2, 2).float())
            ref = self.batch["reference_condition"].clone().requires_grad_()
            pi = actor(self.batch["latent"], self.batch["state"], ref, deterministic=True)
            torch.testing.assert_close(pi, ref)
            pi.sum().backward()
            self.assertIsNone(ref.grad)
        with self.assertRaisesRegex(ValueError, "scale"):
            self.learner(dict(scale=[1, 2, 3]))

    def test_preclip_bc_has_inward_gradient_while_q_sees_saturated_action(self):
        batch = self.batch
        batch["reference_condition"].fill_(2)
        for space in ("preclip", "postclip"):
            for q, bc in ((0.0, 1.0), (1.0, 0.0)):
                learner = self.learner(
                    dict(exploration_std=0),
                    dict(bc_action_space=space, weight_schedule={"enable": False}),
                    q_weight=q,
                    bc_weight=bc,
                )
                learner.critic = ActionScore()
                seen = []
                learner.critic.register_forward_pre_hook(
                    lambda module, args, seen=seen: seen.append(args[2])
                )
                loss, metrics = learner.actor_objective(batch)
                loss.backward()
                self.assertTrue(seen[0].eq(1).all())
                self.assertEqual(metrics["preclip_clip_fraction"], 1.0)
                grad = learner.actor.network[-1].bias.grad
                if space == "preclip" and bc:
                    self.assertTrue((grad > 0).all())
                else:
                    self.assertTrue((grad == 0).all())

    def test_actor_branches_share_one_draw(self):
        batch = self.batch
        for latent in (False, True):
            residual = dict(latent_dim=3, exploration_std=0) if latent else {}
            learner = self.learner(residual, dict(actor_update_action_noise=not latent))
            actor = learner.actor
            with torch.no_grad():
                actor.network[-1].weight.fill_(0.2)
            seed = 123
            kwargs = dict(deterministic=False, apply_action_noise=not latent)
            expected_rng = torch.Generator().manual_seed(seed)
            pi, u = actor.forward_with_preclip(
                batch["latent"],
                batch["state"],
                batch["reference_condition"],
                generator=expected_rng,
                **kwargs,
            )
            rng = torch.Generator().manual_seed(seed)
            seen = []
            learner.critic.register_forward_pre_hook(
                lambda module, args, seen=seen: seen.append(args[2])
            )
            with patch.object(
                actor, "forward_with_preclip", wraps=actor.forward_with_preclip
            ) as forward:
                _, metrics = learner.actor_objective(batch, generator=rng)
            forward.assert_called_once()
            self.assertTrue(torch.equal(seen[0], pi))
            target = torch.where(batch["episode_success"][:, None, None], batch["executed"], 0.0)
            self.assertEqual(metrics["actor_bc"], float((u.detach() - target).square().mean()))
            self.assertTrue(torch.equal(rng.get_state(), expected_rng.get_state()))

    def test_latent_evaluation_modes_and_explicit_sample(self):
        args = [self.batch[k] for k in ("latent", "state", "reference_condition")]
        for mode in ("sample", "zero"):
            actor = self.learner(
                dict(latent_dim=3, exploration_std=0, eval_latent_mode=mode),
                dict(actor_update_action_noise=False),
            ).actor
            inputs = []
            actor.network.register_forward_pre_hook(
                lambda module, args, inputs=inputs: inputs.append(args[0])
            )
            rng = torch.Generator().manual_seed(10)
            state = rng.get_state()
            actor(*args, deterministic=True, generator=rng)
            self.assertEqual(torch.equal(state, rng.get_state()), mode == "zero")
            self.assertEqual(bool(inputs[-1][:, -3:].eq(0).all()), mode == "zero")
            actor(*args, latent_z=torch.ones(2, 3))
            self.assertTrue(inputs[-1][:, -3:].eq(1).all())
            for z in (torch.zeros(2, 2), torch.full((2, 3), float("nan"))):
                with self.assertRaisesRegex(ValueError, "latent_z"):
                    actor(*args, latent_z=z)
            with self.assertRaisesRegex(ValueError, "output noise"):
                actor(*args, apply_action_noise=True)
        with self.assertRaisesRegex(ValueError, "latent_z"):
            self.learner().actor(*args, latent_z=torch.zeros(2, 3))

    def test_success_anchor_uses_only_legal_known_executions(self):
        ref = torch.tensor([[[2.0], [-3.0]]]).expand(3, -1, -1)
        executed = torch.full_like(ref, float("nan"))
        executed[0, 0] = 0.4
        success = torch.tensor([True, True, False])
        known = torch.tensor([True, False, True])
        valid = torch.tensor([[True, False], [True, True], [True, True]])
        target, mask = legal_success_behavior_target(executed, ref, success, known, valid)
        expected = ref.clamp(-1, 1)
        expected[0, 0] = 0.4
        torch.testing.assert_close(target, expected)
        self.assertEqual(mask.sum(), 1)
        executed[0, 0] = 1.000001
        with self.assertRaisesRegex(ValueError, "Executed"):
            legal_success_behavior_target(executed, ref, success, known, valid)
        with self.assertRaisesRegex(ValueError, "Reference"):
            legal_success_behavior_target(executed, ref * float("nan"), success, known, valid)

    def test_q_gate_strictly_compares_minimum_values_and_preserves_rows(self):
        rows = [
            transition(i, episode_success=True, takeover=torch.ones(2, dtype=torch.bool))
            for i in range(3)
        ]
        for row, value in zip(rows, (0.5, 0.0, -0.5), strict=True):
            row.executed.actions.fill_(value)
        batch = residual_batch(rows, 2, "cpu")
        self.assertTrue(batch["reference_condition"].eq(0).all())
        before = content_digest((rows, batch))
        learner = self.learner(dict(exploration_std=0), dict(success_anchor="q_gated_success"))
        learner.critic = ActionScore()
        _, metrics = learner.actor_objective(batch)
        self.assertAlmostEqual(metrics["actor_bc"], 0.25 / 3)
        self.assertAlmostEqual(metrics["anchor_q_gate_selected_fraction"], 1 / 3)
        self.assertEqual(content_digest((rows, batch)), before)
        batch["executed"][1, 0, 0] = 1.5
        with self.assertRaisesRegex(ValueError, "Executed"):
            learner.actor_objective(batch)
        reference = self.learner(dict(exploration_std=0), dict(success_anchor="reference_only"))
        self.assertEqual(reference.actor_objective(batch)[1]["actor_bc"], 0.0)

    def test_q_gate_ignores_unexecuted_terminal_candidate_suffix(self):
        row = transition(terminated=True, episode_success=True)
        row.executed.executed_mask[:, 1] = False
        learner = self.learner(td3={"success_anchor": "q_gated_success"})
        learner.critic = ActionScore()
        for suffix in (-0.75, 0.75):
            row.terminal_candidate_actions = torch.full_like(row.executed.actions, suffix)
            _, metrics = learner.actor_objective(residual_batch([row], 2, "cpu"))
            self.assertEqual(metrics["anchor_q_gate_selected_fraction"], 1.0)
            self.assertAlmostEqual(metrics["anchor_q_gate_executed_minus_reference"], 0.125)

    def test_critic_objectives_and_bootstrap_clipping(self):
        batch = self.batch
        for bce in (False, True):
            learner = self.learner(
                dict(critic_output="sigmoid" if bce else "linear"),
                dict(critic_objective="bce" if bce else "mse"),
            )
            target = torch.tensor([0.0, 0.7])
            args = [batch[name] for name in ("latent", "state", "critic_actions")]
            q = learner.critic(*args)
            loss, _ = learner.critic_objective(batch, target)
            expected = target[:, None].expand_as(q)
            if bce:
                logits = learner.critic(*args, return_logits=True)
                self.assertTrue(((q > 0) & (q < 1)).all())
                torch.testing.assert_close(q, logits.sigmoid())
                torch.testing.assert_close(
                    loss, F.binary_cross_entropy_with_logits(logits, expected)
                )
                with self.assertRaisesRegex(ValueError, "unmodified TD targets"):
                    learner.target_values(batch)
            else:
                torch.testing.assert_close(loss, F.mse_loss(q, expected))
        learner = self.learner(dict(critic_layer_norm=True), dict(target_q_clip=[0.0, 1.0]))
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in learner.critic.modules()), 2)
        with torch.no_grad():
            for head in (learner.target.q1, learner.target.q2):
                head[-1].weight.zero_()
                head[-1].bias.fill_(5)
        gamma = learner.config.gamma
        torch.testing.assert_close(
            learner.target_values(batch), torch.tensor([1 + gamma + gamma**2, 1 + gamma])
        )
        batch["critic_actions"][0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "Critic actions"):
            learner.critic_objective(batch, torch.ones(2))

    def test_target_routes_noise_only_to_enabled_rows(self):
        learner = self.learner()
        batch = residual_batch([transition(0, next_actor_enabled=False), transition(1)], 2, "cpu")
        seen = []
        learner.target.register_forward_pre_hook(
            lambda module, args, seen=seen: seen.append(args[2])
        )
        learner.target_values(batch)
        self.assertTrue(torch.equal(seen[0][0], batch["next_reference"][0, :2]))
        self.assertTrue(seen[0][1].ne(0).any())

    def test_rankq_validation_preserves_both_training_generators(self):
        learner = self.learner(td3=dict(rank_weight=0.1))
        rows = [
            *self.rows,
            transition(2, episode_success=False),
            transition(3, episode_success=True),
        ]
        metrics = learner.train_rows(rows)
        self.assertGreater(metrics["rank_loss"], 0)
        self.assertGreater(metrics["rank_support_loss"], 0)
        before = content_digest(learner.state_dict())
        self.assertEqual(validation_loss(learner, rows), validation_loss(learner, rows))
        self.assertEqual(before, content_digest(learner.state_dict()))

    def test_weight_schedule_and_independent_target_clock(self):
        schedule = WeightScheduleConfig(warmup_updates=2, ramp_updates=4)
        for step, progress in ((0, 0.0), (1, 0.0), (2, 0.25), (3, 0.5), (5, 1.0), (8, 1.0)):
            bc, q, metrics = schedule.resolve(step, 1, 2)
            self.assertAlmostEqual(bc, 7 + progress * (2.8 - 7))
            self.assertAlmostEqual(q, 0.05 + progress * 0.45)
            self.assertEqual(metrics["actor_weight_in_warmup"], float(step < 2))
            self.assertEqual(metrics["actor_weight_ramp_progress"], progress)
        self.assertEqual(replace(schedule, enable=False).resolve(0, 1, 2)[:2], (2, 1))
        self.assertEqual(replace(schedule, ramp_updates=0).resolve(2, 1, 2)[:2], (2.8, 0.5))
        for frequency in (1, 3):
            learner = self.learner(target_update_frequency=frequency)
            for step in range(1, 4):
                before = content_digest(learner.target.state_dict())
                learner.train_rows(self.rows)
                self.assertEqual(learner.update_step, step)
                self.assertEqual(learner.actor_updates, step // 2)
                self.assertEqual(
                    before != content_digest(learner.target.state_dict()), step % frequency == 0
                )

    def test_episode_outcome_propagation_and_bitwise_resume(self):
        learner = self.learner()
        first, last = self.rows
        prior = self.replay.commit(first)
        saved = self.replay.state_dict()
        self.replay.commit(last)
        self.replay.load_state_dict(saved)
        sample = self.replay.sample(20)
        self.assertTrue(any(r.key == first.key for r in sample))
        self.assertTrue(all(r.episode_success for r in sample))
        self.assertIsNone(self.replay.rows[first.key][0].episode_success)
        self.assertEqual(self.replay.commit(first), prior)
        with self.assertRaisesRegex(ValueError, "Conflicting episode outcome"):
            self.replay.commit(transition(5, episode_success=False))
        self.assertEqual(self.replay.budget, 2)
        learner.update(1)
        learner.rollout_rng = torch.Generator().manual_seed(100).get_state()
        checkpoint = save_resume(
            self.root / "checks", learner, task_fingerprint="task", run_id="run"
        )
        expected = learner.update()
        digest = content_digest(learner.state_dict())
        restored = self.learner()
        load_resume(checkpoint, restored, task_fingerprint="task")
        self.assertEqual(expected, restored.update())
        self.assertEqual(content_digest(restored.state_dict()), digest)

    def test_outside_scope_outcome_survives_replay_restore(self):
        path = self.root / "critical"
        replay = Replay(path, self.spec, 2, training_scope="critical_phase")
        first = transition(0, critical_phase=True)
        terminal = transition(1, critical_phase=False, terminated=True, episode_success=True)
        try:
            replay.commit(first)
            before_terminal = replay.state_dict()
            self.assertEqual(replay.commit(terminal).reason, "outside_training_scope")
            self.assertIs(replay.sample(1)[0].episode_success, True)
            legacy = replay.state_dict()
            legacy.pop("episode_outcomes_version", None)
            legacy["episode_outcomes"] = {}  # Older snapshots indexed only admitted rows.
        finally:
            replay.close()
        restored = Replay(path, self.spec, 2, training_scope="critical_phase")
        self.addCleanup(restored.close)
        for checkpoint in (before_terminal, legacy):
            restored.load_state_dict(checkpoint)
            self.assertIs(restored.sample(1)[0].episode_success, True)
            self.assertEqual(list(restored.rows), [first.key])
            self.assertEqual(restored.budget, 1)

    def test_pool_retention_and_offline_no_online_budget(self):
        replay = Replay(
            self.root / "pools",
            self.spec,
            2,
            capacity=1,
            sampling="demo_joint_50",
            demo_capacity=10,
        )
        self.addCleanup(replay.close)
        for i in range(2):
            row = transition(i, origin="demo", terminated=True)
            replay.commit(row)
            replay.commit(row)
        self.assertEqual(replay.budget, 0)
        for i in range(2, 6):
            replay.commit(transition(i, origin="online"))
        self.assertEqual(replay.budget, 4)
        self.assertEqual(sum(r.origin == "demo" for r, _ in replay.rows.values()), 2)
        self.assertEqual(sum(r.origin == "demo" for r in replay.sample(10)), 5)

    def test_full_bc_denominator_and_rank_candidates_respect_terminal_padding(self):
        from dobot_nexus.algorithms.rltoken.auxiliary import terminal_rank_loss

        row = transition(terminated=True, episode_success=True)
        row.executed.executed_mask[0, 1] = False
        row.executed.actions[0, 1].fill_(float("nan"))
        row.features.reference_actions[:, 1] = 0.7
        row.terminal_candidate_actions = torch.full_like(row.executed.actions, -0.8)
        batch = residual_batch([row], 2, "cpu")
        learner = self.learner(dict(exploration_std=0))
        _, metrics = learner.actor_objective(batch)
        self.assertEqual(metrics["actor_bc"], 0.25**2 / 2)
        self.assertEqual(metrics["success_anchor_fraction"], 0.5)
        candidates = []
        learner.critic.register_forward_pre_hook(lambda module, args: candidates.append(args[2]))
        rank, _ = terminal_rank_loss(learner.critic, batch, learner.settings, learner.generator)
        self.assertTrue(torch.isfinite(rank))
        self.assertGreater(len(candidates), 1)
        for candidate in candidates:
            self.assertTrue(candidate.abs().le(1).all())
            self.assertTrue(candidate[:, 1].eq(0.7).all())
        for field, value in (("truncated", True), ("episode_success_known", False)):
            skipped = dict(batch, **{field: torch.tensor([value])})
            rank, _ = terminal_rank_loss(
                learner.critic, skipped, learner.settings, learner.generator
            )
            self.assertEqual(float(rank), 0.0)
