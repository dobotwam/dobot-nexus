import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.algorithms.rltoken.batching import observation_row, stack_observations
from dobot_nexus.algorithms.rltoken.env import ExecutionFeedback
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.rollout import RolloutEngine
from dobot_nexus.algorithms.rltoken.simulation import SimulationSession
from dobot_nexus.algorithms.rltoken.transitions import ObservationID
from dobot_nexus.algorithms.rltoken.weight_sync import ActorSnapshot, ActorVersion

from .fakes import observation
from .test_env_rollout import TinyPolicy


def observations(indices, step=0):
    rows = []
    for index in indices:
        row = observation()
        row.state[:] = index + step * 10
        row.env_ids = [f"sim-{index}"]
        rows.append(row)
    return stack_observations(rows, [f"sim-{index}" for index in indices])


class VectorAgent:
    def __init__(self, *, time_out=False, query_fails=False):
        self.resets = []
        self.executions = self.queries = self.closes = 0
        self.time_out, self.query_fails = time_out, query_fails

    def reset(self, env_indices, *, timeout):
        self.resets.append(list(env_indices))
        return observations(env_indices)

    def execute_batch(self, command_ids, commands, *, timeout):
        self.executions += 1
        self.command_shape = commands.shape
        self.feedbacks = []
        for index, command_id in enumerate(command_ids):
            self.feedbacks.append(
                ExecutionFeedback(
                    command_id,
                    commands[index : index + 1].copy(),
                    2,
                    np.ones(2, dtype=np.float32),
                    np.zeros(2, dtype=np.bool_),
                    observations([index], self.executions),
                )
            )
        if self.time_out:
            raise TimeoutError("Vector step completed; response lost")
        return self.feedbacks

    def query_batch(self, command_ids, *, timeout):
        self.queries += 1
        if self.query_fails:
            raise TimeoutError("Unknown vector execution")
        return self.feedbacks

    def close(self):
        self.closes += 1


class SimulationTest(unittest.TestCase):
    def test_reference_batch_seed_does_not_depend_on_random_run_or_episode_ids(self):
        seeds = []
        extract = self.policy.provider.extract_rl_features

        def record(obs, options):
            seeds.append(options.seed)
            return extract(obs, options)

        self.policy.provider.extract_rl_features = record
        self.engine.packets(self.identities(), observations(range(3)))
        fresh = RolloutEngine(
            self.policy,
            run_id="another-run",
            learner_epoch="another-epoch",
            seed=3,
            max_batch_size=3,
        )
        fresh.packets(
            [
                replace(
                    identity,
                    run_id="another-run",
                    session_epoch="another-epoch",
                    episode_id="random-uuid",
                )
                for identity in self.identities()
            ],
            observations(range(3)),
        )
        self.assertEqual(seeds[0], seeds[1])

    def setUp(self):
        torch.set_num_threads(1)
        self.policy = TinyPolicy()
        self.engine = RolloutEngine(
            self.policy, run_id="run", learner_epoch="epoch", seed=3, max_batch_size=3
        )
        self.sync(0)
        self.feature_batches = []
        extract = self.policy.provider.extract_rl_features

        def tracked(obs, options):
            self.feature_batches.append(obs.batch_size)
            packet = extract(obs, options)
            packet.latent[:, 0] = torch.from_numpy(obs.state[:, 0].copy())
            return packet

        self.policy.provider.extract_rl_features = tracked
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.replay = Replay(Path(self.temp.name) / "replay.sqlite", self.policy.feature_spec, 2)
        self.addCleanup(self.replay.close)

    def sync(self, update):
        self.engine.weights.apply(
            ActorSnapshot.capture(
                self.policy.actor,
                ActorVersion("run", "epoch", update),
                self.policy.feature_spec.feature_id,
            )
        )

    def session(self, agent):
        return SimulationSession(
            agent,
            batch_size=3,
            run_id="run",
            env_id="sim",
            session_epoch="epoch",
            chunk_size=2,
            action_dim=2,
            bootstrap_truncation=True,
        )

    def commit(self, rows):
        return [self.replay.commit(row) for row in rows]

    def identities(self):
        return [ObservationID("run", f"sim-{i}", "epoch", f"episode-{i}", "0") for i in range(3)]

    def test_feature_and_actor_forwards_are_batched_and_match_scalar_deterministic_output(self):
        actor_batches = []
        with self.engine.weights.request() as (actor, _):
            hook = actor.network.register_forward_pre_hook(
                lambda model, args: actor_batches.append(args[0].shape[0])
            )
        decisions = self.engine.decide_batch(
            self.identities(), observations(range(3)), actor_ready=True, deterministic=True
        )
        hook.remove()
        self.assertEqual(self.feature_batches, [3])
        self.assertEqual(actor_batches, [3])
        self.assertEqual({row.actor_version for row in decisions}, {("run", "epoch", 0)})
        for index, row in enumerate(decisions):
            packet = row.features
            self.assertEqual(packet.action_context["raw_state"], [[index, index]])
            with self.engine.weights.request() as (actor, _):
                expected = actor(
                    packet.latent, packet.state, packet.reference_actions, deterministic=True
                )
            np.testing.assert_allclose(row.commands, expected.detach().numpy(), atol=1e-6)
        self.sync(1)
        repeated = self.engine.decide_batch(
            self.identities(), observations(range(3)), actor_ready=True
        )
        self.assertEqual(self.feature_batches, [3])
        self.assertTrue(all(row.actor_version[-1] == 1 for row in repeated))

    def test_independent_reset_and_truncated_final_state_survive_next_batch(self):
        agent = VectorAgent()
        execute = agent.execute_batch

        def finish_differently(*args, **kwargs):
            rows = execute(*args, **kwargs)
            if agent.executions == 1:
                rows[0] = replace(rows[0], terminated=True)
                rows[1] = replace(rows[1], truncated=True)
            return rows

        agent.execute_batch = finish_differently
        session = self.session(agent)
        status = session.collect_chunk(self.engine.process, self.commit)
        self.assertEqual((status["chunks"], status["env_steps"], self.replay.budget), (3, 6, 3))
        self.assertEqual(agent.command_shape, (3, 2, 2))
        rows = [item[0] for item in self.replay.rows.values()]
        self.assertIsNone(rows[0].next_features)
        self.assertTrue(rows[1].truncated)
        self.assertEqual(rows[1].next_features.action_context["raw_state"], [[11, 11]])
        old_ids = [row.observation_id for row in rows]
        session.collect_chunk(self.engine.process, self.commit, actor_ready=True)
        self.assertEqual(agent.resets, [[0, 1, 2], [0, 1]])
        self.assertEqual(self.feature_batches, [3, 2, 2, 3])
        following = [item[0] for item in self.replay.rows.values()][3:]
        self.assertNotEqual(following[0].observation_id.episode_id, old_ids[0].episode_id)
        self.assertEqual(following[2].observation_id.episode_id, old_ids[2].episode_id)
        self.assertEqual(following[2].features.action_context["raw_state"], [[12, 12]])
        self.assertEqual((self.replay.watermark, self.replay.budget), (6, 6))
        with self.assertRaisesRegex(ValueError, "closed episode"):
            self.engine.packet(old_ids[0], observations([0]))
        session.close()
        session.close()
        self.assertEqual(agent.closes, 1)

    def test_short_chunk_rejects_only_affected_environment(self):
        agent = VectorAgent()
        execute = agent.execute_batch

        def short(*args, **kwargs):
            rows = execute(*args, **kwargs)
            rows[1] = replace(rows[1], executed_steps=1, terminated=True)
            return rows

        agent.execute_batch = short
        status = self.session(agent).collect_chunk(self.engine.process, self.commit)
        self.assertEqual(status["admission"], ["admitted", "rejected", "admitted"])
        self.assertEqual(status["env_steps"], 5)
        self.assertEqual((self.replay.watermark, self.replay.budget), (3, 2))

    def test_terminal_recipe_keeps_success_in_a_mixed_vector_batch(self):
        agent = VectorAgent()
        execute = agent.execute_batch

        def short(*args, **kwargs):
            rows = execute(*args, **kwargs)
            rows[1] = replace(rows[1], executed_steps=1, terminated=True,
                              rewards=np.array([1.0, 0.0], dtype=np.float32))
            return rows

        agent.execute_batch = short
        replay = Replay(Path(self.temp.name) / "terminal.sqlite", self.policy.feature_spec,
                        2, allow_terminal_chunks=True)
        self.addCleanup(replay.close)
        status = self.session(agent).collect_chunk(
            self.engine.process, lambda rows: [replay.commit(row) for row in rows])
        self.assertEqual(status["admission"], ["admitted"] * 3)
        self.assertEqual(status["reason"][1], "terminal_prefix")
        self.assertEqual((status["env_steps"], replay.budget), (5, 3))
        terminal = list(replay.rows.values())[1][0]
        self.assertIsNotNone(terminal.terminal_candidate_actions)
        self.assertIsNone(terminal.next_features)
        self.assertEqual(int(terminal.executed.executed_mask.sum()), 1)

    def test_timeout_queries_once_and_unknown_execution_is_durable(self):
        agent = VectorAgent(time_out=True)
        session = self.session(agent)
        session.collect_chunk(self.engine.process, self.commit)
        self.assertEqual((agent.executions, agent.queries, self.replay.budget), (1, 1, 3))
        agent.query_fails = True
        status = session.collect_chunk(self.engine.process, self.commit)
        self.assertTrue(status["faulted"])
        self.assertEqual(status["reason"], ["unconfirmed_execution"] * 3)
        self.assertEqual(
            (agent.executions, agent.queries, self.replay.watermark, self.replay.budget),
            (2, 2, 6, 3),
        )
        with self.assertRaisesRegex(RuntimeError, "reconciliation"):
            session.collect_chunk(self.engine.process, self.commit)
        self.assertEqual(agent.executions, 2)

    def test_misordered_feedback_or_ack_stops_collection_without_reexecution(self):
        for wrong in ("feedback", "ack"):
            with self.subTest(wrong=wrong):
                # Use a fresh session/engine for each failure, without resetting a faulted session.
                engine = RolloutEngine(
                    self.policy, run_id="run", learner_epoch="epoch", max_batch_size=3
                )
                engine.weights.apply(
                    ActorSnapshot.capture(
                        self.policy.actor,
                        ActorVersion("run", "epoch", 0),
                        self.policy.feature_spec.feature_id,
                    )
                )
                agent = VectorAgent()
                execute = agent.execute_batch
                if wrong == "feedback":
                    agent.execute_batch = lambda *a, execute=execute, **kw: list(
                        reversed(execute(*a, **kw))
                    )
                session = self.session(agent)
                commit = (
                    self.commit
                    if wrong == "feedback"
                    else lambda rows: list(reversed(self.commit(rows)))
                )
                with self.assertRaises(ValueError):
                    session.collect_chunk(engine.process, commit)
                with self.assertRaises(RuntimeError):
                    session.collect_chunk(engine.process, commit)
                self.assertEqual(agent.executions, 1)

    def test_batch_shape_ids_and_feature_context_fail_before_cache_mutation(self):
        ids = self.identities()
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.engine.packets([ids[0]] * 3, observations(range(3)))
        with self.assertRaisesRegex(ValueError, "env_ids"):
            self.engine.packets(ids[::-1], observations(range(3)))
        scalar = RolloutEngine(self.policy, run_id="run", learner_epoch="epoch")
        with self.assertRaisesRegex(ValueError, "batch size"):
            scalar.packets(ids, observations(range(3)))
        self.assertEqual(self.engine.environments, {})
        self.engine.packets(ids, observations(range(3)))
        conflicting = observations(range(3))
        conflicting.state[1, 0] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.engine.packets(ids, conflicting)
        for index in range(1, 4):
            self.engine.packet(replace(ids[0], observation_id=str(index)), observations([0], index))
        self.assertEqual(len(self.engine.environments["sim-0"]["cache"]), 2)
        with self.assertRaisesRegex(ValueError, "Expired"):
            self.engine.packet(ids[0], observations([0]))
        # Eviction in sim-0 must not evict sim-1's current packet.
        before = len(self.feature_batches)
        self.engine.packet(ids[1], observation_row(observations(range(3)), 1))
        self.assertEqual(len(self.feature_batches), before)
