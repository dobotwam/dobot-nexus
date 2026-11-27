import tempfile
import unittest
import weakref
from dataclasses import replace
from pathlib import Path

import torch

from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.replay import Replay, _dump
from dobot_nexus.algorithms.rltoken.transitions import residual_batch, training_batch

from .fakes import feature_spec, transition


def demo(row):
    return row.origin == "demo" or bool(row.takeover.any()) or row.source == "human"


class SamplerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def open(self, mode, **kwargs):
        replay = Replay(self.root / mode, feature_spec(), 2, seed=45, sampling=mode, **kwargs)
        self.addCleanup(replay.close)
        return replay

    def test_fifo_and_rng_match_list_sampler_after_every_event(self):
        for mode in ("uniform", "demo_joint_50"):
            with self.subTest(mode=mode):
                self.compare_sampler(mode)

    def compare_sampler(self, mode):
        # Enough for several ordinary rows; the longer command record evicts many at once.
        limit = 12 * len(_dump(transition().cpu_copy()))
        args = dict(
            capacity=11,
            demo_capacity=10,
            max_bytes=limit,
            demo_max_bytes=limit,
            updates_per_new_transition=3,
        )
        replay = self.open(mode, **args)
        reference = []
        rng = torch.Generator().manual_seed(45)
        budget = online = max_evicted = 0
        for i in range(80):
            row = transition(i, origin="demo" if i < 22 else ("vla" if i % 7 == 0 else "online"))
            if i >= 22 and i % 3 == 0:
                row.takeover[0] = True
            if i >= 22 and i % 5 == 0:
                row.source = "human"
            if i in (18, 76):
                row.candidate_commands = "x" * (limit * 3 // 4)
            if i == 32:
                pending = replace(row, reward_finalized=False, rewards=None)
                self.assertEqual(replay.commit(pending).status, "pending")
                self.assertNotIn(row.key, replay.rows)
            ack = replay.commit(row, finalize=(i == 32))
            self.assertEqual(replay.commit(row), ack)
            self.assertEqual(ack.status, "admitted")
            if row.origin not in ("demo", "vla"):
                budget += 3
                online += 1
            reference.append((row, len(_dump(row.cpu_copy()))))
            previous_len = len(reference)
            pool = [item for item in reference if mode == "uniform" or demo(item[0]) == demo(row)]
            cap = 10 if mode == "demo_joint_50" and demo(row) else 11
            while len(pool) > cap or sum(size for _, size in pool) > limit:
                evicted = pool.pop(0)[0].key
                reference = [item for item in reference if item[0].key != evicted]
            max_evicted = max(max_evicted, previous_len - len(reference))
            self.assertEqual(list(replay.rows), [r.key for r, _ in reference])
            self.assertEqual(replay.bytes, sum(n for _, n in reference))
            self.assertEqual((replay.budget, replay.online_admitted), (budget, online))

            def sample(pool, count):
                return [pool[n] for n in torch.randint(len(pool), (count,), generator=rng)]

            if mode == "uniform":
                expected = sample([r for r, _ in reference], 7)
            else:
                demos = [r for r, _ in reference if demo(r)]
                joint = [r for r, _ in reference if not demo(r)]
                expected = sample(demos or joint, 3) + sample(joint or demos, 4)
                self.assertEqual(replay._demo_bytes, sum(n for r, n in reference if demo(r)))
                self.assertEqual(replay._online_bytes, replay.bytes - replay._demo_bytes)
            actual = replay._sample_views(7)
            self.assertEqual([r.key for r in actual], [r.key for r in expected])
            self.assertTrue(torch.equal(replay.generator.get_state(), rng.get_state()))
            self.assertEqual(replay.last_sample_demo_fraction, sum(map(demo, expected)) / 7)
        self.assertGreaterEqual(max_evicted, 2)

    def test_legacy_restore_rebuilds_before_tail_and_retains_evicted_episode_labels(self):
        for mode in ("uniform", "demo_joint_50"):
            with self.subTest(mode=mode):
                replay = self.open(mode, capacity=1, demo_capacity=1)
                labeled = transition(0, terminated=True, episode_success=False)
                replay.commit(labeled)
                replay.commit(transition(1, source="human"))
                snapshot = replay.state_dict()
                snapshot.pop("episode_outcomes_version")
                snapshot.pop("episode_outcomes")
                for i in (2, 3):
                    row = transition(i, terminated=True, source="human" if i == 3 else "reference")
                    row.observation_id = replace(row.observation_id, episode_id="second")
                    replay.commit(row)
                self.assertFalse(
                    any(
                        r.observation_id.session == labeled.observation_id.session
                        for r, _ in replay.rows.values()
                    )
                )
                expected = replay._checkpoint_state_dict()
                keys = [r.key for r in replay._sample_views(5)]
                for restore in (replay.load_state_dict, replay._load_checkpoint_state_dict):
                    restore(snapshot)
                    self.assertEqual(list(replay.rows), list(expected["rows"]))
                    self.assertEqual(content_digest(replay.state_dict()), content_digest(expected))
                    self.assertEqual([r.key for r in replay._sample_views(5)], keys)
                    with self.assertRaisesRegex(ValueError, "Conflicting episode outcome"):
                        replay.commit(transition(4, episode_success=True))

    def test_empty_demo_pool_fallback_and_duplicate_public_hit_isolation(self):
        replay = self.open("demo_joint_50", capacity=1)
        row = transition(0)
        replay.commit(row)
        owned = replay.rows[row.key][0]
        first, second = replay.sample(2)
        first.executed.actions.zero_()
        self.assertTrue(second.executed.actions.eq(0.25).all())
        self.assertTrue(owned.executed.actions.eq(0.25).all())
        self.assertEqual(replay.last_sample_demo_fraction, 0.0)
        replay.commit(transition(1, record_transition=False, episode_success=False))
        view = replay._sample_views(1)[0]
        self.assertIs(view.episode_success, False)
        self.assertIsNone(owned.episode_success)
        self.assertIs(view.executed, owned.executed)
        ref = weakref.ref(owned)
        del view, owned
        replay.commit(transition(2))
        # The evicted slot must drop its strong reference immediately.
        self.assertIsNone(ref())

    def test_public_state_isolation_and_private_ownership(self):
        replay = self.open("uniform")
        row = transition(0, episode_success=True)
        replay.commit(row)
        public = replay.state_dict()
        borrowed = replay._checkpoint_state_dict()
        self.assertEqual(content_digest(public), content_digest(borrowed))
        self.assertIs(borrowed["rows"][row.key][0], replay.rows[row.key][0])
        for name in ("rows", "episode_outcomes", "status_counts"):
            borrowed[name].clear()
            self.assertTrue(getattr(replay, name))
        original = content_digest(replay.state_dict())
        public["rows"][row.key][0].takeover.fill_(True)
        public["episode_outcomes"].clear()
        public["status_counts"].clear()
        public["rng"].zero_()
        self.assertEqual(content_digest(replay.state_dict()), original)
        public = replay.state_dict()
        replay.load_state_dict(public)
        replay.rows[row.key][0].rewards.zero_()
        self.assertTrue(public["rows"][row.key][0].rewards.eq(1).all())
        public["rows"][row.key][0].takeover.fill_(True)
        self.assertFalse(replay.rows[row.key][0].takeover.any())

    def test_residual_batch_matches_legacy_for_every_key(self):
        terminal = transition(
            4, terminated=True, terminal_candidate_actions=torch.full((1, 2, 2), 0.7)
        )
        terminal.executed.executed_mask[:, 1] = False
        terminal.executed.actions[:, 1] = torch.nan
        terminal.takeover[0] = True
        truncated = transition(
            1,
            truncated=True,
            bootstrap_truncation=True,
            episode_success=True,
            next_actor_enabled=False,
        )
        following = transition(2, episode_success=False)
        following.next_features.latent.fill_(9)
        for rows in ([truncated, terminal, following], [terminal], [following]):
            expected = training_batch(rows, 2, "cpu")
            expected.update(
                reference_condition=torch.cat([r.features.reference_actions[:, :2] for r in rows]),
                valid_mask=torch.cat([r.executed.executed_mask for r in rows]),
                episode_success=torch.tensor([bool(r.episode_success) for r in rows]),
                episode_success_known=torch.tensor([r.episode_success is not None for r in rows]),
                terminated=~expected["bootstrap"],
                truncated=torch.tensor([bool(r.truncated) for r in rows]),
                intervene_flags=torch.stack([r.takeover for r in rows]),
            )
            before = content_digest(rows)
            actual = residual_batch(iter(rows), 2, torch.device("cpu"))
            self.assertEqual(actual.keys(), expected.keys())
            for key, value in expected.items():
                with self.subTest(key=key, size=len(rows)):
                    if value is None:
                        self.assertIsNone(actual[key])
                    else:
                        torch.testing.assert_close(
                            actual[key], value, rtol=0, atol=0, equal_nan=True
                        )
                        actual[key].zero_()
            self.assertEqual(content_digest(rows), before)
