"""Thin worker business-method contracts, with only the external Worker base substituted."""

import asyncio
import importlib
import queue
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.learner import Learner
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.weight_sync import WeightSlots
from dobot_nexus.core.config import _wrap

from .fakes import feature_spec, transition


class BoundedChannel:
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)

    def get_nowait(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty as exc:
            raise asyncio.QueueEmpty from exc

    def put_nowait(self, value):
        try:
            self.queue.put_nowait(value)
        except queue.Full as exc:
            raise asyncio.QueueFull from exc


class WorkerMethodsTest(unittest.TestCase):
    def test_vector_collection_crosses_request_response_and_durable_ack_channels(self):
        from dobot_nexus.algorithms.rltoken.rollout import RolloutEngine
        from dobot_nexus.algorithms.rltoken.simulation import SimulationSession
        from dobot_nexus.algorithms.rltoken.weight_sync import ActorSnapshot, ActorVersion

        from .test_env_rollout import TinyPolicy
        from .test_simulation import VectorAgent

        policy = TinyPolicy()
        engine = RolloutEngine(policy, run_id="run", learner_epoch="epoch", max_batch_size=3)
        engine.weights.apply(
            ActorSnapshot.capture(
                policy.actor, ActorVersion("run", "epoch", 0), policy.feature_spec.feature_id
            )
        )
        replay = Replay(Path(self.temp.name) / "vector.sqlite", policy.feature_spec, 2)
        self.addCleanup(replay.close)
        self.worker.learner = Learner(
            policy.feature_spec, policy.head_config, LearningConfig(), replay
        )
        self.worker.batch_size = 3
        rollout = self.module.RolloutWorker.__new__(self.module.RolloutWorker)
        rollout.channels, rollout.timeout, rollout.max_bytes = self.channels, 2.0, 2**20
        rollout.engine = engine
        env = self.module.EnvProxyWorker.__new__(self.module.EnvProxyWorker)
        env.channels, env.timeout, env.max_bytes = self.channels, 2.0, 2**20
        env.session = SimulationSession(
            VectorAgent(),
            batch_size=3,
            run_id="run",
            env_id="sim",
            session_epoch="epoch",
            chunk_size=2,
            action_dim=2,
        )
        self.addCleanup(env.close)
        with ThreadPoolExecutor(max_workers=1) as pool:
            collecting = pool.submit(env.collect_chunk, actor_ready=True)
            deadline = time.monotonic() + 5
            while not collecting.done() and time.monotonic() < deadline:
                rollout.serve_slice()
                self.worker.update(0)
            result = collecting.result(timeout=1)
        self.assertFalse(result["faulted"])
        self.assertEqual((result["chunks"], result["env_steps"], replay.budget), (3, 6, 3))
        self.assertEqual(policy.provider.calls, 2)

    def test_vector_ack_fits_one_channel_message_and_budget_counts_transitions(self):
        self.worker.batch_size = 3
        first = transition()
        second = replace(
            first,
            observation_id=replace(first.observation_id, env_id="env-1"),
            next_observation_id=replace(first.next_observation_id, env_id="env-1"),
        )
        third = replace(
            first,
            observation_id=replace(first.observation_id, env_id="env-2"),
            next_observation_id=replace(first.next_observation_id, env_id="env-2"),
        )
        self.channels["transitions"].put_nowait([first, second, third])
        result = self.worker.update(0)
        acks = self.channels["durable_acks"].get_nowait()
        self.assertEqual([ack.key for ack in acks], [row.key for row in (first, second, third)])
        self.assertEqual((result["admitted"], result["budget"]), (3, 3))
        # Exact duplicate transport cannot create new UTD budget.
        self.channels["transitions"].put_nowait([first, second, third])
        duplicate = self.worker.update(0)
        self.channels["durable_acks"].get_nowait()
        self.assertEqual((duplicate["admitted"], duplicate["budget"]), (3, 3))
        self.worker.batch_size = 1
        self.channels["transitions"].put_nowait([first, second])
        with self.assertRaisesRegex(ValueError, "batch exceeds"):
            self.worker.update(0)

    def test_simulation_gpu_visibility_is_explicit_and_robot_stays_cpu(self):
        import os

        for kind, gpu, expected in (
            ("robot", None, ""),
            ("simulation", None, ""),
            ("simulation", 0, "7"),
        ):
            config = _wrap(
                {
                    "environment": {"kind": kind, "batch_size": 1},
                    "runtime": {"resources": {"env_gpu": gpu}},
                }
            )
            with (
                self.subTest(kind=kind, gpu=gpu),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}),
            ):
                self.module.EnvProxyWorker(config, self.channels, "run", "epoch")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], expected)

    def setUp(self):
        package = ModuleType("rlinf")
        scheduler = ModuleType("rlinf.scheduler")
        scheduler.Worker = object
        self.module_name = "dobot_nexus.algorithms.rltoken.workers_rlinf"
        self.old_module = sys.modules.pop(self.module_name, None)
        with patch.dict(sys.modules, {"rlinf": package, "rlinf.scheduler": scheduler}):
            self.module = importlib.import_module(self.module_name)
        self.addCleanup(self.restore_module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        replay = Replay(Path(self.temp.name) / "journal.sqlite", feature_spec(), 2)
        self.addCleanup(replay.close)
        self.learner = Learner(
            feature_spec(), HeadConfig(2, (8,)), LearningConfig(warmup=2, batch_size=2), replay
        )
        self.channels = {
            key: BoundedChannel()
            for key in (
                "transitions",
                "durable_acks",
                "weights",
                "weight_acks",
                "requests",
                "responses",
            )
        }
        self.worker = self.module.LearnerWorker.__new__(self.module.LearnerWorker)
        self.worker.learner = self.learner
        self.worker.channels = self.channels
        self.worker.max_bytes = 2**20
        self.worker.timeout = 1.0
        self.worker.run_id, self.worker.epoch = "run", "epoch"
        self.worker.batch_size = 1

    def restore_module(self):
        sys.modules.pop(self.module_name, None)
        if self.old_module is not None:
            sys.modules[self.module_name] = self.old_module

    def test_zero_update_budget_still_emits_durable_ack_and_bounds_data_lane(self):
        self.channels["transitions"].put_nowait(transition())
        result = self.worker.update(0)
        self.assertEqual(result["updates"], 0)
        ack = self.channels["durable_acks"].get_nowait()
        self.assertEqual((ack.status, result["admitted"]), ("admitted", 1))
        with self.assertRaisesRegex(ValueError, "message exceeds"):
            self.module._put(self.channels["transitions"], b"x" * 100, 10)
        self.channels["transitions"].put_nowait(transition(1))
        with self.assertRaises(asyncio.QueueFull):
            self.channels["transitions"].put_nowait(transition(2))

    def test_both_sync_calls_complete_after_applied_ack(self):
        rollout = self.module.RolloutWorker.__new__(self.module.RolloutWorker)
        rollout.channels, rollout.timeout, rollout.max_bytes = self.channels, 1.0, 2**20
        rollout.engine = SimpleNamespace(
            weights=WeightSlots(self.learner.actor, self.learner.spec.feature_id, "run", "epoch")
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            receiving = pool.submit(rollout.request_actor_sync_model)
            sending = pool.submit(self.worker.sync_model_to_rollout)
            self.assertEqual(receiving.result(timeout=2), 0)
            self.assertEqual(sending.result(timeout=2), 0)
        self.assertEqual(rollout.engine.weights.version.actor_update, 0)
        start = time.monotonic()
        self.assertEqual(rollout.serve_slice()["served"], 0)
        self.assertLess(time.monotonic() - start, 0.5)
