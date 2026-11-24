from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.algorithms.rltoken.actor_critic import Actor
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.env import EnvSession, ExecutionFeedback
from dobot_nexus.algorithms.rltoken.learner import Learner
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.rollout import RolloutEngine
from dobot_nexus.algorithms.rltoken.transitions import ObservationID
from dobot_nexus.algorithms.rltoken.weight_sync import ActorSnapshot, ActorVersion
from dobot_nexus.core.config import _wrap

from .fakes import TinyFeatureModel, observation


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = _wrap({"model": {"state_dim": 2, "action_dim": 2, "action_horizon": 3}})
        self.provider = TinyFeatureModel(config, mode="infer")
        self.feature_spec = self.provider.spec_rl()
        self.head_config = HeadConfig(2, (8,))
        self.actor = Actor(self.feature_spec, self.head_config)
        self.adapter = self.provider.adapter
        self.mode = "infer"


class ConfirmingAgent:
    def __init__(self, *, short=False, timeout=False):
        self.short, self.timeout = short, timeout
        self.executions = self.queries = 0

    def reset(self, *, timeout):
        return observation()

    def execute(self, command_id, commands, *, timeout):
        self.executions += 1
        self.feedback = ExecutionFeedback(
            command_id,
            np.full_like(commands, 0.4),
            1 if self.short else 2,
            np.array([1.0, 2.0]),
            np.array([True, False]),
            observation(),
        )
        if self.timeout:
            raise TimeoutError("reply lost after execution")
        return self.feedback

    def query(self, command_id, *, timeout):
        self.queries += 1
        return self.feedback

    def close(self):
        pass


class EnvRolloutTest(unittest.TestCase):
    def test_per_step_final_feedback_labels_earlier_chunks_at_sampling(self):
        agent = ConfirmingAgent()
        execute = agent.execute

        def finish_second_chunk(*args, **kwargs):
            feedback = execute(*args, **kwargs)
            if agent.executions == 2:
                return replace(feedback, terminated=True, episode_success=False)
            return feedback

        agent.execute = finish_second_chunk
        session = self.session(agent)
        session.collect_chunk(self.engine.process, self.replay.commit)
        self.assertIsNone(self.replay.sample(1)[0].episode_success)
        session.collect_chunk(self.engine.process, self.replay.commit)
        first, last = (row for row, _ in self.replay.rows.values())
        self.assertIsNone(first.episode_success)
        self.assertIs(last.episode_success, False)
        sample = self.replay.sample(16)
        self.assertEqual({r.key for r in sample}, {first.key, last.key})
        self.assertTrue(all(r.episode_success is False for r in sample))
        self.assertEqual(self.replay.budget, 2)

    def test_episode_labels_are_finalized_after_durable_pending_ack(self):
        from dobot_nexus.algorithms.rltoken.transitions import Finalization
        agent = ConfirmingAgent()
        execute = agent.execute
        def terminal(*args, **kwargs):
            return replace(execute(*args, **kwargs), terminated=True)
        agent.execute = terminal
        seen = []
        def finalize(**kwargs):
            self.assertEqual(self.replay.budget, 0)
            self.assertEqual(len(self.replay.pending()), 1)
            seen.extend(kwargs["pending"])
            return [Finalization(row.key, torch.tensor([0., 0.]), True) for row in kwargs["pending"]]
        agent.finalize_episode = finalize
        session = EnvSession(agent, run_id="run", env_id="env", session_epoch="epoch",
                             chunk_size=2, action_dim=2, reward_mode="episode", deterministic=True)
        status = session.collect_chunk(self.engine.process, self.replay.commit)
        self.assertTrue(status["episode_ended"])
        self.assertEqual(len(seen), 1)
        self.assertEqual(self.replay.budget, 1)
        self.assertFalse(session.pending_records)
        self.assertTrue(next(iter(self.replay.rows.values()))[0].rewards.eq(0).all())

    def test_scalar_decision_forwards_deterministic_option(self):
        request = dict(operation="decide", identity=ObservationID("run", "env", "epoch", "ep", "0"),
                       observation=observation(), actor_ready=True, deterministic=True)
        before = self.engine.generator.get_state().clone()
        first = self.engine.process(request)
        second = self.engine.process(request)
        np.testing.assert_array_equal(first.commands, second.commands)
        torch.testing.assert_close(before, self.engine.generator.get_state())

    def test_unresolved_execution_is_journaled_and_stops_the_session(self):
        agent = ConfirmingAgent(timeout=True)

        def unavailable(*args, **kwargs):
            raise TimeoutError("controller status unavailable")

        agent.query = unavailable
        session = self.session(agent)
        result = session.collect_chunk(self.engine.process, self.replay.commit)
        self.assertEqual(result["reason"], "unconfirmed_execution")
        self.assertTrue(result["faulted"])
        self.assertEqual((self.replay.watermark, self.replay.budget, agent.executions), (1, 0, 1))
        from dobot_nexus.algorithms.rltoken.replay import _load

        payload = self.replay.connection.execute("SELECT payload FROM events").fetchone()[0]
        row = _load(payload)
        self.assertIsNone(row.executed_commands)
        self.assertIsNotNone(row.command_id)
        with self.assertRaisesRegex(RuntimeError, "reconciliation"):
            session.collect_chunk(self.engine.process, self.replay.commit)

    def setUp(self):
        torch.set_num_threads(1)
        self.policy = TinyPolicy()
        self.engine = RolloutEngine(self.policy, run_id="run", learner_epoch="epoch")
        self.sync(0)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.replay = Replay(Path(self.temp.name) / "journal.sqlite", self.policy.feature_spec, 2)
        self.addCleanup(self.replay.close)

    def sync(self, version):
        self.engine.weights.apply(
            ActorSnapshot.capture(
                self.policy.actor,
                ActorVersion("run", "epoch", version),
                self.policy.feature_spec.feature_id,
            )
        )

    def session(self, agent):
        return EnvSession(
            agent, run_id="run", env_id="env", session_epoch="epoch", chunk_size=2, action_dim=2
        )

    def test_observation_packet_survives_actor_version_change(self):
        identity = ObservationID("run", "env", "epoch", "episode", "0")
        before = self.engine.decide(identity, observation(), actor_ready=True)
        self.sync(1)
        after = self.engine.decide(identity, observation(), actor_ready=True)
        self.assertEqual(self.policy.provider.calls, 1)
        torch.testing.assert_close(
            before.features.reference_actions, after.features.reference_actions
        )
        self.assertNotEqual(before.actor_version, after.actor_version)
        altered = observation()
        altered.state[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Conflicting observation"):
            self.engine.packet(identity, altered)

    def test_confirmed_takeover_commands_feed_learner_without_reexecuting_on_timeout(self):
        agent = ConfirmingAgent(timeout=True)
        session = self.session(agent)
        result = session.collect_chunk(self.engine.process, self.replay.commit)
        self.assertEqual((agent.executions, agent.queries, result["admission"]), (1, 1, "admitted"))
        row = next(iter(self.replay.rows.values()))[0]
        self.assertEqual(row.source, "reference")
        self.assertIsNone(row.actor_version)
        torch.testing.assert_close(row.executed.actions, torch.full((1, 2, 2), 0.4))
        self.assertTrue(row.features.reference_actions.eq(0).all())
        np.testing.assert_allclose(row.executed_commands, 0.4)
        learner = Learner(
            self.policy.feature_spec,
            self.policy.head_config,
            LearningConfig(batch_size=1, warmup=1),
            self.replay,
        )
        self.assertEqual(learner.update()["updates"], 1)
        self.engine.weights.apply(
            ActorSnapshot.capture(
                learner.actor,
                ActorVersion("run", "epoch", learner.actor_updates),
                self.policy.feature_spec.feature_id,
            )
        )
        session.collect_chunk(self.engine.process, self.replay.commit, actor_ready=True)
        # Current s_next was reused for the next decision, not recomputed after synchronization.
        self.assertEqual(self.policy.provider.calls, 3)
        second = list(self.replay.rows.values())[1][0]
        self.assertEqual(second.source, "actor")
        self.assertEqual(second.actor_version[-1], 1)

    def test_short_feedback_is_durable_but_never_adds_budget(self):
        result = self.session(ConfirmingAgent(short=True)).collect_chunk(
            self.engine.process, self.replay.commit
        )
        self.assertEqual((result["admission"], result["reason"]), ("rejected", "short_chunk"))
        self.assertEqual((self.replay.watermark, self.replay.budget), (1, 0))

    def test_terminal_prefix_reaches_learner_with_candidate_and_reward(self):
        agent = ConfirmingAgent(short=True)
        original = agent.execute

        def execute(*args, **kwargs):
            return replace(original(*args, **kwargs), terminated=True,
                           rewards=np.array([1.0, 0.0]))

        agent.execute = execute
        replay = Replay(Path(self.temp.name) / "terminal.sqlite", self.policy.feature_spec,
                        2, allow_terminal_chunks=True)
        self.addCleanup(replay.close)
        session = self.session(agent)
        result = session.collect_chunk(self.engine.process, replay.commit)
        self.assertEqual(result["admission"], "admitted")
        row = replay.sample(1)[0]
        torch.testing.assert_close(row.terminal_candidate_actions, torch.zeros(1, 2, 2))
        self.assertEqual(int(row.executed.executed_mask.sum()), 1)
        self.assertIsNone(session.identity)
        learner = Learner(self.policy.feature_spec, self.policy.head_config,
                          LearningConfig(batch_size=1, warmup=1), replay)
        self.assertEqual(learner.update()["updates"], 1)
        self.assertEqual((learner.critic_updates, learner.actor_updates, replay.budget), (1, 1, 0))

    def test_reset_observation_is_not_used_as_terminal_next_state(self):
        agent = ConfirmingAgent()
        original = agent.execute

        def execute(*args, **kwargs):
            return replace(original(*args, **kwargs), terminated=True)

        agent.execute = execute
        session = self.session(agent)
        session.collect_chunk(self.engine.process, self.replay.commit)
        row = next(iter(self.replay.rows.values()))[0]
        self.assertIsNone(row.next_features)
        self.assertIsNone(session.identity)
        self.assertEqual(self.policy.provider.calls, 1)
