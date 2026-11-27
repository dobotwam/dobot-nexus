import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dobot_nexus.algorithms.rltoken.batching import observation_row, stack_observations
from dobot_nexus.algorithms.rltoken.env import EnvSession
from dobot_nexus.algorithms.rltoken.replay import Replay, _load
from dobot_nexus.algorithms.rltoken.rollout import RolloutEngine
from dobot_nexus.algorithms.rltoken.routing import PolicySwitch
from dobot_nexus.algorithms.rltoken.simulation import SimulationSession
from dobot_nexus.algorithms.rltoken.transitions import Finalization, ObservationID
from dobot_nexus.algorithms.rltoken.weight_sync import ActorSnapshot, ActorVersion
from .fakes import observation, transition
from .test_env_rollout import ConfirmingAgent, TinyPolicy
from .test_simulation import VectorAgent


SWITCH = dict(execution_scope="critical_phase", training_scope="critical_phase", task_mode="full_task")


def flagged(flag, env_id=None):
    row = observation()
    row.control = {"rlt_switch_flags": np.array([flag])}
    row.env_ids = [env_id] if env_id else None
    return row


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def engine(self, size=1, config=None):
        policy = TinyPolicy()
        engine = RolloutEngine(policy, run_id="run", learner_epoch="epoch", max_batch_size=size,
                               policy_switch=SWITCH if config is None else config)
        engine.weights.apply(ActorSnapshot.capture(policy.actor, ActorVersion("run", "epoch", 0),
                                                   policy.feature_spec.feature_id))
        return engine

    def replay(self, engine, name="journal"):
        replay = Replay(self.root / name, engine.policy.feature_spec, 2, training_scope="critical_phase")
        self.addCleanup(replay.close)
        return replay

    def test_readiness_and_phase_are_independent_and_inactive_rows_skip_actor(self):
        engine = self.engine(2)
        identities = [ObservationID("run", f"env-{i}", "epoch", "episode", "0") for i in range(2)]
        obs = stack_observations([flagged(False), flagged(True)], ["env-0", "env-1"])
        calls = []
        with engine.weights.request() as (actor, _):
            hook = actor.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
        self.addCleanup(hook.remove)
        rng = engine.generator.get_state().clone()
        warmup = engine.decide_batch(identities, obs, actor_ready=False)
        self.assertEqual([r.source for r in warmup], ["reference", "reference"])
        self.assertEqual([r.record_transition for r in warmup], [False, True])
        torch.testing.assert_close(engine.generator.get_state(), rng)
        active = engine.decide_batch(identities, obs, actor_ready=True)
        self.assertEqual([r.source for r in active], ["reference", "actor"])
        self.assertIsNone(active[0].actor_version)
        self.assertIsNotNone(active[1].actor_version)
        self.assertEqual(calls, [1])
        np.testing.assert_allclose(active[0].commands, 0.)
        obs.control["rlt_switch_flags"][0] = True
        with self.assertRaisesRegex(ValueError, "Conflicting observation"):
            engine.decide_batch(identities, obs, actor_ready=True)

    def test_scope_options_missing_flags_and_batch_metadata(self):
        for execution in ("all", "critical_phase"):
            for training in ("all", "critical_phase"):
                switch = PolicySwitch(execution, training)
                _, act, record = switch.select(flagged(False))
                self.assertEqual(bool(act[0]), execution == "all")
                self.assertEqual(bool(record[0]), training == "all")
        with self.assertRaisesRegex(ValueError, "rlt_switch_flags"):
            PolicySwitch(**SWITCH).select(observation())
        _, act, record = PolicySwitch("critical_phase", "critical_phase", "critical_phase").select(observation())
        self.assertTrue(act[0] and record[0])
        for kwargs in (dict(execution_scope="last"), dict(training_scope="human"), dict(task_mode="auto")):
            with self.assertRaises(ValueError):
                PolicySwitch(**kwargs)
        batch = stack_observations([flagged(False), flagged(True)], ["a", "b"])
        row = observation_row(batch, 1)
        row.control["rlt_switch_flags"][:] = False
        self.assertTrue(batch.control["rlt_switch_flags"][1])
        with self.assertRaisesRegex(ValueError, "control signal"):
            stack_observations([observation(), flagged(True)], ["a", "b"])

    def test_scalar_boundary_excludes_approach_and_records_vla_warmup(self):
        engine = self.engine()
        replay = self.replay(engine)
        class Agent(ConfirmingAgent):
            def reset(self, **kwargs):
                return flagged(False)
            def execute(self, *args, **kwargs):
                feedback = super().execute(*args, **kwargs)
                return replace(feedback, final_observation=flagged(True), terminated=self.executions == 2)
            def finalize_episode(self, *, pending, **kwargs):
                self.pending = pending
                return [Finalization(r.key, torch.ones(2), True) for r in pending]
        agent = Agent()
        session = EnvSession(agent, run_id="run", env_id="env", session_epoch="epoch",
                             chunk_size=2, action_dim=2, reward_mode="episode")
        first = session.collect_chunk(engine.process, replay.commit, actor_ready=True)
        self.assertEqual(first["reason"], "outside_training_scope")
        self.assertEqual((replay.budget, len(session.pending_records)), (0, 0))
        # Entering the phase in the final observation does not admit the preceding approach chunk.
        session.collect_chunk(engine.process, replay.commit, actor_ready=False)
        self.assertEqual(len(agent.pending), 1)
        row = next(iter(replay.rows.values()))[0]
        self.assertEqual(row.source, "reference")
        self.assertTrue(row.critical_phase)
        self.assertEqual(replay.budget, 1)
        reset = session.collect_chunk(engine.process, replay.commit, actor_ready=True)
        self.assertFalse(reset["critical_phase"])
        self.assertFalse(reset["actor_switch"])

    def test_vector_phase_routing_and_independent_episode_reset(self):
        engine = self.engine(2)
        replay = self.replay(engine)
        class Agent(VectorAgent):
            def reset(self, env_indices, **kwargs):
                self.resets.append(list(env_indices))
                return stack_observations([flagged(i == 1) for i in env_indices],
                                           [f"sim-{i}" for i in env_indices])
            def execute_batch(self, *args, **kwargs):
                rows = super().execute_batch(*args, **kwargs)
                return [replace(row, final_observation=flagged(True, f"sim-{i}"),
                                terminated=i == 1 and self.executions == 1)
                        for i, row in enumerate(rows)]
        agent = Agent()
        session = SimulationSession(agent, batch_size=2, env_id="sim", run_id="run",
                                     session_epoch="epoch", chunk_size=2, action_dim=2)
        commit = lambda rows: [replay.commit(row) for row in rows]
        first = session.collect_chunk(engine.process, commit, actor_ready=True)
        self.assertEqual(first["actor_switch"], [False, True])
        self.assertEqual(first["record_transition"], [False, True])
        self.assertEqual(replay.budget, 1)
        second = session.collect_chunk(engine.process, commit, actor_ready=False)
        self.assertEqual(second["actor_switch"], [False, False])
        self.assertEqual(second["record_transition"], [True, True])
        self.assertEqual(agent.resets, [[0, 1], [1]])
        self.assertEqual(replay.budget, 3)

    def test_replay_scope_survives_recovery_and_late_finalization(self):
        engine = self.engine()
        replay = self.replay(engine)
        spec = engine.policy.feature_spec
        excluded = transition(spec=spec, critical_phase=False, record_transition=False, reward_finalized=False, accepted=None)
        self.assertEqual(replay.commit(excluded).reason, "outside_training_scope")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            replay.commit(Finalization(excluded.key, torch.ones(2), True))
        self.assertEqual(replay.commit(transition(1, spec=spec)).reason, "missing_critical_phase")
        pending = transition(2, spec=spec, critical_phase=True, record_transition=True, reward_finalized=False, accepted=None)
        replay.commit(pending)
        with self.assertRaisesRegex(ValueError, "Finalization"):
            replay.commit(replace(pending, critical_phase=False, reward_finalized=True, accepted=True), finalize=True)
        checkpoint = replay.state_dict()
        replay.commit(Finalization(pending.key, torch.ones(2), True))
        restored = self.replay(engine)
        restored.load_state_dict(checkpoint)
        self.assertEqual((len(restored.rows), restored.budget, len(restored.pending())), (1, 1, 0))
        self.assertEqual(restored.commit(excluded).reason, "outside_training_scope")
        payload = restored.connection.execute("SELECT payload FROM events ORDER BY sequence LIMIT 1").fetchone()[0]
        self.assertFalse(_load(payload).critical_phase)
        with self.assertRaisesRegex(ValueError, "identity"):
            Replay(self.root / "journal", engine.policy.feature_spec, 2, training_scope="all")
