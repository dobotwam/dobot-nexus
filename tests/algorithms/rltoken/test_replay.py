import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.transitions import training_batch

from .fakes import feature_spec, transition


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "journal.sqlite"
        self.spec = feature_spec()

    def replay(self, **kwargs):
        replay = Replay(self.path, self.spec, 2, **kwargs)
        self.addCleanup(replay.close)
        return replay

    def test_durable_dedup_after_eviction_and_conflict(self):
        replay = self.replay(capacity=1, updates_per_new_transition=2)
        row = transition()
        ack = replay.commit(row)
        replay.commit(transition(1))
        self.assertEqual(replay.commit(row), ack)
        self.assertEqual((len(replay.rows), replay.budget, replay.admitted), (1, 4, 2))
        row.rewards[0] = 9
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            replay.commit(row)

    def test_partial_unknown_reward_and_reset_are_recorded_but_not_learned(self):
        replay = self.replay()
        short = transition()
        short.executed.executed_mask[:, -1] = False
        reset = transition(3)
        reset.next_observation_id = replace(reset.next_observation_id, episode_id="after-reset")
        for row, reason in (
            (short, "short_chunk"),
            (transition(1, rewards=None), "missing_or_invalid_reward"),
            (transition(2, execution_confirmed=False), "unconfirmed_execution"),
            (reset, "reset_observation"),
            (transition(4, truncated=True, next_features=None), "undeclared_truncation_semantics"),
        ):
            ack = replay.commit(row)
            self.assertEqual((ack.status, ack.reason), ("rejected", reason))
        self.assertEqual((replay.budget, len(replay.rows)), (0, 0))
        self.assertEqual(replay.watermark, 5)

    def test_pending_finalize_and_tail_recovery_count_once(self):
        replay = self.replay()
        pending = transition(rewards=None, reward_finalized=False)
        self.assertEqual(replay.commit(pending).status, "pending")
        saved = replay.state_dict()
        final = pending.finalize(rewards=torch.tensor([1.0, 2.0]))
        ack = replay.commit(final, finalize=True)
        self.assertEqual(replay.budget, 1)
        replay.load_state_dict(saved)
        self.assertEqual((replay.budget, replay.admitted), (1, 1))
        self.assertEqual(replay.commit(final), ack)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            replay.commit(pending)

    def test_intervention_does_not_mutate_raw_or_next_reference(self):
        row = transition(terminated=True, next_features=None)
        row.takeover[0] = True
        self.assertEqual(row.admission(self.spec, 2)[0], "admitted")
        batch = training_batch([row], 2, "cpu")
        torch.testing.assert_close(batch["bc_target"][0, 0], torch.full((2,), 0.25))
        self.assertTrue(row.features.reference_actions.eq(0).all())
        self.assertIsNone(batch["next_latent"])
        self.assertFalse(batch["bootstrap"].any())

    @staticmethod
    def terminal_prefix(index=0):
        row = transition(index, terminated=True, next_features=None,
                         next_observation_id=None, rewards=torch.tensor([1.0, 0.0]),
                         terminal_candidate_actions=torch.full((1, 2, 2), 0.7))
        row.executed.executed_mask[:, 1] = False
        row.executed.encodable_mask[:, 1] = False
        row.executed.actions[:, 1] = 0
        return row

    def test_terminal_prefix_preserves_reward_and_actual_execution(self):
        from dobot_nexus.algorithms.rltoken.losses import td_target

        replay = self.replay(allow_terminal_chunks=True)
        row = self.terminal_prefix()
        ack = replay.commit(row)
        self.assertEqual((ack.status, ack.reason), ("admitted", "terminal_prefix"))
        self.assertEqual(replay.commit(row), ack)
        self.assertEqual(replay.budget, 1)
        saved = replay.sample(1)[0]
        batch = training_batch([saved], 2, "cpu")
        torch.testing.assert_close(batch["critic_actions"], torch.tensor([[[0.25, 0.25], [0.7, 0.7]]]))
        self.assertTrue(batch["executed"][:, 1].eq(0).all())
        self.assertEqual(int(saved.executed.executed_mask.sum()), 1)
        self.assertIsNone(batch["next_latent"])
        torch.testing.assert_close(
            td_target(batch["rewards"], batch["bootstrap"], torch.empty(0), 0.99),
            torch.ones(1))

    def test_terminal_recipe_rejects_ambiguous_or_fabricated_tails(self):
        cases = []
        row = self.terminal_prefix()
        row.terminated, row.truncated, row.bootstrap_truncation = False, True, True
        cases.append((row, "short_chunk"))
        row = self.terminal_prefix()
        row.executed.executed_mask[:] = False
        cases.append((row, "invalid_terminal_prefix"))
        row = self.terminal_prefix()
        row.executed.executed_mask[:] = torch.tensor([[False, True]])
        cases.append((row, "invalid_terminal_prefix"))
        row = self.terminal_prefix()
        row.terminal_candidate_actions = None
        cases.append((row, "missing_or_invalid_terminal_candidate"))
        row = self.terminal_prefix()
        row.terminal_candidate_actions[0, 1, 0] = float('nan')
        cases.append((row, "missing_or_invalid_terminal_candidate"))
        row = self.terminal_prefix()
        row.rewards[1] = 1
        cases.append((row, "terminal_padding_has_feedback"))
        row = self.terminal_prefix()
        row.takeover[1] = True
        cases.append((row, "terminal_padding_has_feedback"))
        for row, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(row.admission(self.spec, 2, allow_terminal_chunks=True),
                                 ("rejected", reason))

    def test_fixed_recipe_and_replay_partitions_stay_separate(self):
        row = self.terminal_prefix()
        self.assertEqual(row.admission(self.spec, 2), ("rejected", "short_chunk"))
        replay = self.replay()
        replay.commit(transition())
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            Replay(self.path, self.spec, 2, allow_terminal_chunks=True)

    def test_terminal_candidate_is_copied_and_survives_recovery(self):
        replay = self.replay(allow_terminal_chunks=True)
        row = self.terminal_prefix()
        ack = replay.commit(row)
        row.terminal_candidate_actions.fill_(-0.9)
        restored = self.replay(allow_terminal_chunks=True)
        self.assertEqual(restored.commit(self.terminal_prefix()), ack)
        self.assertEqual(restored.budget, 1)
        self.assertTrue(restored.sample(1)[0].terminal_candidate_actions.eq(0.7).all())
