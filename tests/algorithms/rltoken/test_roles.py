"""Exercise the online roles through their RPC and bounded channel contracts."""

from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dobot_nexus.algorithms.rltoken.artifacts import feature_metadata, write_model_bundle
from dobot_nexus.algorithms.rltoken.topology import RLTOKEN_CHANNELS, build_rltoken_topology
from dobot_nexus.algorithms.rltoken.weight_sync import ActorVersion
from dobot_nexus.core.config import _wrap, load_config
from dobot_nexus.runtime.contracts import ChannelFull, ChannelSpec
from dobot_nexus.runtime.local import LocalRuntime
from dobot_nexus.runtime.local.channel import BoundedLocalChannel

from .fakes import TinyFeatureModel, transition
from .test_env_rollout import ConfirmingAgent
from .test_simulation import VectorAgent

ROOT = Path(__file__).resolve().parents[3]


def scalar_agent(config):
    return ConfirmingAgent()


def vector_agent(config):
    return VectorAgent()


def online_config(root, *, vector=False, td3=False):
    """Public online YAML with a self-contained CPU feature artifact."""
    template = "td3_online" if td3 else "stage2"
    config = load_config(ROOT / f"examples/configs/rltoken/{template}.yaml")
    source = _wrap(
        {
            "schema_version": 1,
            "task": "train",
            "train": {},
            "data": {},
            "runtime": {"device": "cpu"},
            "model": {
                "target": f"{TinyFeatureModel.__module__}:TinyFeatureModel",
                "state_dim": 2,
                "action_dim": 2,
                "action_horizon": 3,
            },
        }
    )
    bundle = write_model_bundle(root / "source", TinyFeatureModel(source, mode="infer"), source)
    config.exp.run_dir = str(root)
    config.model.features.artifact = str(bundle)
    config.model.update(state_dim=2, action_dim=2, action_horizon=2)
    config.model.rltoken.update(chunk_size=2, hidden_dims=[8])
    config.algorithm.learning.update(warmup=1, batch_size=2)
    if td3:
        config.model.rltoken.residual.hidden_dims = [8]
        config.algorithm.pop("initialize_from")
    config.environment.update(
        target=f"{__name__}:scalar_agent",
        min_actor_updates=1,
        max_chunks=20,
        timeout=1,
    )
    if vector:
        config.environment.update(
            target=f"{__name__}:vector_agent",
            kind="simulation",
            batch_size=3,
            env_id="sim",
        )
    config.runtime.rpc_timeout = 10
    config.train.update(max_steps=3, save_every=2)
    return config


class RoleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def start(self, vector=False):
        config = online_config(self.root, vector=vector)
        self.spec = feature_metadata(config.model.features.artifact)[1]
        topology = build_rltoken_topology(
            config,
            run_id="run",
            epoch="epoch",
            task_fingerprint="task",
            resume_path=None,
        )
        self.session = LocalRuntime().start(topology)
        self.addCleanup(self.session.close)
        self.learner = self.session.role("learner")
        self.channels = self.session.channels
        return self.session.role("rollout"), self.session.role("env")

    def test_vector_collection_crosses_request_response_and_durable_ack_channels(self):
        rollout, env = self.start(vector=True)
        recv = rollout.call("request_actor_sync_model")
        self.learner.call("sync_model_to_rollout").wait()
        recv.wait()
        with patch.object(
            TinyFeatureModel,
            "extract_rl_features",
            autospec=True,
            side_effect=TinyFeatureModel.extract_rl_features,
        ) as extract:
            collecting = env.call("collect_chunk", actor_ready=True)
            while not collecting.done() and time.monotonic() < collecting.deadline:
                rollout.call("serve_slice").wait()
                status = self.learner.call("update", 0).wait()
            result = collecting.wait()
        self.assertFalse(result["faulted"])
        self.assertEqual((result["chunks"], result["env_steps"], status["budget"]), (3, 6, 3))
        self.assertEqual(extract.call_count, 2)

    def test_vector_ack_is_one_message_and_duplicates_do_not_add_budget(self):
        self.start(vector=True)
        first = transition(spec=self.spec)
        rows = [
            replace(
                first,
                observation_id=replace(first.observation_id, env_id=f"env-{i}"),
                next_observation_id=replace(first.next_observation_id, env_id=f"env-{i}"),
            )
            for i in range(3)
        ]
        for _ in range(2):
            self.channels.put_nowait("transitions", rows)
            status = self.learner.call("update", 0).wait()
            self.assertEqual(
                [ack.key for ack in self.channels.try_get("durable_acks")],
                [row.key for row in rows],
            )
            self.assertEqual((status["admitted"], status["budget"]), (3, 3))
        self.channels.put_nowait("transitions", [*rows, first])
        with self.assertRaisesRegex(ValueError, "batch exceeds"):
            self.learner.call("update", 0).wait()

    def test_zero_update_durable_ack_full_and_oversized_message(self):
        self.start()
        self.channels.put_nowait("transitions", transition(spec=self.spec))
        status = self.learner.call("update", 0).wait()
        self.assertEqual(status["updates"], 0)
        self.assertEqual(
            (self.channels.try_get("durable_acks").status, status["admitted"]), ("admitted", 1)
        )
        self.channels.put_nowait("transitions", transition(1, self.spec))
        with self.assertRaises(ChannelFull):
            self.channels.put_nowait("transitions", transition(2, self.spec))
        lane = BoundedLocalChannel(ChannelSpec("transitions", max_message_bytes=10))
        with self.assertRaisesRegex(ValueError, "message exceeds"):
            lane.put_nowait(b"x" * 100)

    def test_both_sync_calls_complete_after_applied_ack(self):
        rollout, _ = self.start()
        recv = rollout.call("request_actor_sync_model")
        send = self.learner.call("sync_model_to_rollout")
        self.assertEqual(recv.wait(), ActorVersion("run", "epoch", 0))
        self.assertEqual(send.wait(), ActorVersion("run", "epoch", 0))
        start = time.monotonic()
        self.assertEqual(
            rollout.call("serve_slice").wait(), {"served": 0, "applied_actor_update": 0}
        )
        self.assertLess(time.monotonic() - start, 0.5)

    def test_robot_and_simulation_map_public_gpu_resources_and_channel_limits(self):
        config = online_config(self.root)
        config.runtime.update(ray_tmpdir="/tmp", max_message_bytes=777)
        for kind, gpu in (("robot", None), ("simulation", None), ("simulation", 0)):
            config.environment.kind = kind
            config.runtime.resources.env_gpu = gpu
            with self.subTest(kind=kind, gpu=gpu):
                topology = build_rltoken_topology(
                    config, run_id="r", epoch="e", task_fingerprint="fp", resume_path="saved"
                )
                rollout, learner, env = topology.roles
                self.assertEqual((rollout.placement.gpu_id, learner.placement.gpu_id), (0, 1))
                self.assertEqual(env.placement.device_type, "cpu" if gpu is None else "gpu")
                self.assertEqual(env.placement.gpu_id, gpu)
                self.assertEqual(learner.init_kwargs["resume_path"], "saved")
                self.assertEqual(learner.init_kwargs["task_fingerprint"], "fp")
                self.assertEqual(tuple(c.name for c in topology.channels), RLTOKEN_CHANNELS)
                self.assertEqual({c.max_message_bytes for c in topology.channels}, {777})
                self.assertEqual((topology.rpc_timeout, topology.ray_tmpdir), (10, "/tmp"))
