"""Optional RLinf adapters. No algorithm object inherits Worker."""

from __future__ import annotations

import asyncio
import os
import pickle
import shutil
import time
import uuid
from pathlib import Path

import torch
from rlinf.scheduler import Worker

from dobot_nexus.core.import_utils import import_from_string

from .artifacts import retain_feature_bundle, write_model_bundle
from .batching import environment_batch_size
from .checkpoint import load_resume, save_resume
from .configuration import LearningConfig
from .env import EnvSession
from .learner import create_learner
from .policy import RLTokenPolicy
from .replay import Replay
from .rollout import RolloutEngine
from .routing import PolicySwitch
from .simulation import SimulationSession
from .weight_sync import ActorSnapshot, ActorVersion


def _put(channel, payload, limit):
    if len(pickle.dumps(payload, protocol=5)) > limit:
        raise ValueError("RLToken message exceeds runtime.max_message_bytes.")
    channel.put_nowait(payload)


def _receive(channel, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return channel.get_nowait()
        except asyncio.QueueEmpty:
            time.sleep(0.005)
    raise TimeoutError("RLToken channel response deadline exceeded.")


def _device():
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each RLToken GPU worker must see exactly one assigned CUDA device.")
    torch.cuda.set_device(0)
    return "cuda:0"


class EnvProxyWorker(Worker):
    def __init__(self, config, channels, run_id, epoch):
        super().__init__()
        self.batch_size = environment_batch_size(config.environment)
        self.simulation = config.environment.get("kind", "robot") == "simulation"
        if not self.simulation or config.runtime.resources.get("env_gpu") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.config, self.channels, self.run_id, self.epoch = config, channels, run_id, epoch
        self.session = None
        self.timeout = float(config.runtime.get("rpc_timeout", 120))
        self.max_bytes = int(config.runtime.get("max_message_bytes", 64 * 2**20))

    def initialize(self):
        cfg = self.config.environment
        if self.simulation and self.config.runtime.resources.get("env_gpu") is not None:
            _device()
        agent = import_from_string(cfg.target)(cfg)
        session_type = SimulationSession if self.simulation else EnvSession
        self.session = session_type(
            agent,
            run_id=self.run_id,
            env_id=cfg.get("env_id", "robot-0"),
            session_epoch=self.epoch,
            chunk_size=self.config.model.rltoken.chunk_size,
            action_dim=self.config.model.action_dim,
            timeout=cfg.timeout,
            bootstrap_truncation=cfg.bootstrap_truncation,
            reward_mode=cfg.get("reward_mode", "per_step"),
            deterministic=cfg.get("deterministic", False),
            **({"batch_size": self.batch_size} if self.simulation else {}),
        )
        return True

    def restore_pending(self, rows):
        self.session.restore_pending(rows)
        return True

    def _rollout(self, request):
        request_id = uuid.uuid4().hex
        _put(self.channels["requests"], (request_id, request), self.max_bytes)
        response_id, response = _receive(self.channels["responses"], self.timeout)
        if response_id != request_id:
            raise ValueError("Inference reply request ID mismatch.")
        return response

    def _commit(self, row):
        _put(self.channels["transitions"], row, self.max_bytes)
        return _receive(self.channels["durable_acks"], self.timeout)

    def collect_chunk(self, actor_ready=False):
        return self.session.collect_chunk(self._rollout, self._commit, actor_ready=actor_ready)

    def close(self):
        if self.session is not None:
            self.session.close()


class RolloutWorker(Worker):
    def __init__(self, config, channels, run_id, epoch):
        super().__init__()
        self.config, self.channels, self.run_id, self.epoch = config, channels, run_id, epoch
        self.engine = None
        self.timeout = float(config.runtime.get("rpc_timeout", 120))
        self.max_bytes = int(config.runtime.get("max_message_bytes", 64 * 2**20))

    def initialize(self):
        device = _device()
        feature_root = retain_feature_bundle(
            self.config.model.features.artifact,
            Path(self.config.exp.run_dir) / "rltoken/features",
        )
        self.config.model.features.artifact = str(feature_root)
        # Assets and model-owned config now point into the retained run bundle.
        policy = RLTokenPolicy(self.config, mode="infer").to(device)
        self.engine = RolloutEngine(
            policy,
            run_id=self.run_id,
            learner_epoch=self.epoch,
            seed=int(self.config.exp.get("seed", 0)),
            max_batch_size=environment_batch_size(self.config.environment),
            reference_num_steps=self.config.algorithm.get("reference_num_steps",
                10 if self.config.algorithm.get("recipe") == "rlt_td3_residual_v1" else None),
        )
        return True

    def serve_slice(self):
        served = 0
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline and served < 4:
            try:
                request_id, request = self.channels["requests"].get_nowait()
            except asyncio.QueueEmpty:
                time.sleep(0.005)
                continue
            response = self.engine.process(request)
            _put(self.channels["responses"], (request_id, response), self.max_bytes)
            served += 1
        version = self.engine.weights.version
        return {"served": served, "applied_actor_update": version.actor_update if version else -1}

    def request_actor_sync_model(self):
        snapshot = _receive(self.channels["weights"], self.timeout)
        applied = self.engine.weights.apply(snapshot)
        _put(self.channels["weight_acks"], applied, self.max_bytes)
        return applied.actor_update

    def rng_state(self):
        return self.engine.generator.get_state().cpu()

    def restore_rng(self, state):
        if state is not None:
            self.engine.generator.set_state(state.cpu())
        return True

    def export_deployment(self, path):
        with self.engine.weights.request() as (actor, version):
            self.engine.policy.actor = actor
            path = Path(path)
            staging = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
            write_model_bundle(staging, self.engine.policy, self.config)
            # Every nested asset and manifest is complete before publishing the directory.
            from .checkpoint import _fsync, _sync_dir

            for file in staging.rglob("*"):
                if file.is_file():
                    _fsync(file)
            for directory in sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True):
                _sync_dir(directory)
            _sync_dir(staging)
            staging.rename(path)
            _sync_dir(path.parent)
        return {"path": str(path), "actor_update": version.actor_update}

    def close(self):
        self.engine = None


class LearnerWorker(Worker):
    def __init__(self, config, channels, run_id, epoch, task_fingerprint, resume_path):
        super().__init__()
        self.config, self.channels, self.run_id, self.epoch = config, channels, run_id, epoch
        self.task_fingerprint, self.resume_path = task_fingerprint, resume_path
        self.learner = None
        self.timeout = float(config.runtime.get("rpc_timeout", 120))
        self.max_bytes = int(config.runtime.get("max_message_bytes", 64 * 2**20))
        self.batch_size = environment_batch_size(config.environment)

    def initialize(self):
        device = _device()
        learning = LearningConfig(**self.config.algorithm.learning)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(learning.seed)
            policy = RLTokenPolicy(self.config, mode="train")
        journal = Path(self.config.exp.run_dir) / "rltoken/journal.sqlite"
        journal.parent.mkdir(parents=True, exist_ok=True)
        if self.resume_path and not journal.exists():
            shutil.copyfile(Path(self.resume_path) / "journal.sqlite", journal)
        elif journal.exists() and not self.resume_path:
            raise ValueError(
                "An existing RL journal requires train.resume_from or a new run directory."
            )
        replay = Replay(
            journal,
            policy.feature_spec,
            policy.head_config.chunk_size,
            **self.config.data.replay,
            updates_per_new_transition=learning.updates_per_new_transition,
            seed=learning.seed,
            task_fingerprint=self.task_fingerprint,
            training_scope=PolicySwitch.from_config(self.config).training_scope,
            allow_terminal_chunks=self.config.algorithm.get("recipe")
            in {"rlt_ac_terminal_v2", "rlt_td3_residual_v1"},
        )
        self.learner = create_learner(
            policy.feature_spec, policy.head_config, learning, replay, device=device, policy=policy
        )
        if self.resume_path:
            restored_run = load_resume(
                self.resume_path, self.learner, task_fingerprint=self.task_fingerprint
            )
            if restored_run != self.run_id:
                raise ValueError("Resume run identity mismatch.")
        elif self.config.algorithm.get("initialize_from"):
            from .pretraining import initialize_from_prewarm

            initialize_from_prewarm(
                self.learner,
                self.config.algorithm.initialize_from,
                task_identity=self.config.algorithm.task_identity,
                inherit_optimizer=self.config.algorithm.get("inherit_prewarm_optimizer", True),
            )
        if self.config.data.get("offline_cache") and not self.resume_path:
            from .offline_data import load_cache
            cache = load_cache(self.config.data.offline_cache, policy.feature_spec,
                               chunk_size=policy.head_config.chunk_size,
                               reward_definition=self.config.algorithm.task_identity,
                               gamma_step=self.learner.gamma_step,
                               reference_num_steps=self.config.algorithm.get("reference_num_steps", 10),
                               policy_switch=self.config.algorithm.get("policy_switch", {}))
            for row in cache["train"]:
                replay.commit(row)
        return self.learner.update(0)

    def update(self, max_updates=1):
        incoming = []
        # One scalar/vector collector has one unacknowledged message. Replay has one owner.
        batched = False
        try:
            payload = self.channels["transitions"].get_nowait()
            batched = isinstance(payload, list)
            incoming = payload if batched else [payload]
            if not 0 < len(incoming) <= self.batch_size:
                raise ValueError("Transition batch exceeds environment.batch_size.")
        except asyncio.QueueEmpty:
            pass
        admitted = self.learner.update(0, incoming=incoming)
        if admitted["acks"]:
            reply = admitted["acks"] if batched else admitted["acks"][0]
            _put(self.channels["durable_acks"], reply, self.max_bytes)
        result = self.learner.update(max_updates)
        result.pop("acks", None)
        return result

    def pending(self):
        return self.learner.replay.pending()

    def sync_model_to_rollout(self):
        self.learner.barrier()
        version = ActorVersion(self.run_id, self.epoch, getattr(self.learner, "actor_revision", self.learner.actor_updates))
        snapshot = ActorSnapshot.capture(self.learner.actor, version, self.learner.spec.feature_id)
        _put(self.channels["weights"], snapshot, self.max_bytes)
        applied = _receive(self.channels["weight_acks"], self.timeout)
        if applied != version:
            raise ValueError("Actor applied ACK version mismatch.")
        return applied.actor_update

    def rollout_rng(self):
        return getattr(self.learner, "rollout_rng", None)

    def snapshot(self, rollout_rng=None):
        if rollout_rng is not None:
            self.learner.rollout_rng = rollout_rng
        path = save_resume(
            Path(self.config.exp.run_dir) / "rltoken/checkpoints",
            self.learner,
            task_fingerprint=self.task_fingerprint,
            run_id=self.run_id,
            feature_manifest="../../features/manifest.yaml",
        )
        return str(path)

    def close(self):
        if self.learner is not None:
            self.learner.replay.close()
            self.learner = None
