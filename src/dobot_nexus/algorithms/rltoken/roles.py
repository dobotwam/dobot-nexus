"""Algorithm-owned online RLToken roles."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import torch

from dobot_nexus.core.import_utils import import_from_string
from dobot_nexus.runtime.contracts import ChannelEmpty

from .artifacts import retain_feature_bundle, write_model_bundle
from .batching import environment_batch_size
from .checkpoint import (
    _PENDING,
    _export_staging,
    _fsync,
    _publication_lock,
    _reservation_epoch,
    _sync_dir,
    load_resume,
    save_resume,
)
from .configuration import LearningConfig
from .env import EnvSession
from .learner import create_learner
from .policy import RLTokenPolicy
from .replay import Replay
from .rollout import RolloutEngine
from .routing import PolicySwitch
from .simulation import SimulationSession
from .weight_sync import ActorSnapshot, ActorVersion


class EnvRole:
    rpc_methods = ("restore_pending", "collect_chunk")

    def __init__(self, config, run_id, epoch):
        self.batch_size = environment_batch_size(config.environment)
        self.simulation = config.environment.get("kind", "robot") == "simulation"
        self.config, self.run_id, self.epoch = config, run_id, epoch
        self.session = None
        self.timeout = float(config.runtime.get("rpc_timeout", 120))

    def initialize(self, ctx):
        self.channels = ctx.channels
        cfg = self.config.environment
        agent = import_from_string(cfg.target)(cfg)
        session_type = SimulationSession if self.simulation else EnvSession
        try:
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
        except BaseException:
            agent.close()
            raise
        return True

    def restore_pending(self, rows):
        self.session.restore_pending(rows)
        return True

    def _rollout(self, request):
        request_id = uuid.uuid4().hex
        self.channels.put_nowait("requests", (request_id, request))
        response_id, response = self.channels.recv("responses", timeout=self.timeout)
        if response_id != request_id:
            raise ValueError("Inference reply request ID mismatch.")
        return response

    def _commit(self, row):
        self.channels.put_nowait("transitions", row)
        return self.channels.recv("durable_acks", timeout=self.timeout)

    def collect_chunk(self, actor_ready=False):
        return self.session.collect_chunk(self._rollout, self._commit, actor_ready=actor_ready)

    def close(self):
        if self.session is not None:
            self.session.close()


class RolloutRole:
    rpc_methods = (
        "serve_slice",
        "request_actor_sync_model",
        "rng_state",
        "restore_rng",
        "export_deployment",
        "feature_manifest_sha256",
    )

    def __init__(self, config, run_id, epoch, expected_manifest_sha256=None):
        self.config, self.run_id, self.epoch = config, run_id, epoch
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.engine = None
        self.timeout = float(config.runtime.get("rpc_timeout", 120))

    def initialize(self, ctx):
        self.channels = ctx.channels
        device = ctx.device
        feature_root = retain_feature_bundle(
            self.config.model.features.artifact,
            Path(self.config.exp.run_dir) / "rltoken/features",
            expected_manifest_sha256=self.expected_manifest_sha256,
        )
        self.config.model.features.artifact = str(feature_root)
        # Assets and model-owned config now point into the retained run bundle.
        policy = RLTokenPolicy(
            self.config, mode="infer", retained_feature_root=feature_root
        ).to(device)
        self._feature_manifest_sha256 = policy._retained_files[Path("manifest.yaml")].sha256
        self.engine = RolloutEngine(
            policy,
            run_id=self.run_id,
            learner_epoch=self.epoch,
            seed=int(self.config.exp.get("seed", 0)),
            max_batch_size=environment_batch_size(self.config.environment),
            reference_num_steps=self.config.algorithm.get(
                "reference_num_steps",
                10 if self.config.algorithm.get("recipe") == "rlt_td3_residual_v1" else None,
            ),
        )
        return True

    def serve_slice(self):
        served = 0
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline and served < 4:
            try:
                request_id, request = self.channels.try_get("requests")
            except ChannelEmpty:
                time.sleep(0.005)
                continue
            response = self.engine.process(request)
            self.channels.put_nowait("responses", (request_id, response))
            served += 1
        version = self.engine.weights.version
        return {"served": served, "applied_actor_update": version.actor_update if version else -1}

    def request_actor_sync_model(self):
        snapshot = self.channels.recv("weights", timeout=self.timeout)
        applied = self.engine.weights.apply(snapshot)
        self.channels.put_nowait("weight_acks", applied)
        return applied

    def rng_state(self):
        return self.engine.generator.get_state().cpu()

    def feature_manifest_sha256(self):
        return self._feature_manifest_sha256

    def restore_rng(self, state):
        if state is not None:
            self.engine.generator.set_state(state.cpu())
        return True

    def export_deployment(self, path):
        with self.engine.weights.request() as (actor, version):
            self.engine.policy.actor = actor
            path = Path(path)
            with _export_staging(path, self.epoch) as staging:
                write_model_bundle(staging, self.engine.policy, self.config)
                # Persist all nested assets before publishing the directory.
                for file in staging.rglob("*"):
                    if file.is_file():
                        _fsync(file)
                directories = (p for p in staging.rglob("*") if p.is_dir())
                for directory in sorted(directories, reverse=True):
                    _sync_dir(directory)
                _sync_dir(staging)
                with _publication_lock(path.parent):
                    if _PENDING.fullmatch(path.name) and _reservation_epoch(path) != self.epoch:
                        raise RuntimeError(
                            f"Deployment export reservation missing or cancelled: {path}"
                        )
                    staging.rename(path)
                    _sync_dir(path.parent)
        return {
            "path": str(path),
            "run_id": version.run_id,
            "learner_epoch": version.learner_epoch,
            "actor_update": version.actor_update,
        }

    def close(self):
        self.engine = None


class LearnerRole:
    rpc_methods = ("update", "pending", "sync_model_to_rollout", "rollout_rng", "snapshot")

    def __init__(self, config, run_id, epoch, task_fingerprint, resume_path):
        self.config, self.run_id, self.epoch = config, run_id, epoch
        self.task_fingerprint, self.resume_path = task_fingerprint, resume_path
        self.learner = None
        self.timeout = float(config.runtime.get("rpc_timeout", 120))
        self.batch_size = environment_batch_size(config.environment)

    def initialize(self, ctx):
        self.channels = ctx.channels
        device = ctx.device
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
        try:
            self.learner = create_learner(
                policy.feature_spec,
                policy.head_config,
                learning,
                replay,
                device=device,
                policy=policy,
            )
        except BaseException:
            replay.close()
            raise
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

            cache = load_cache(
                self.config.data.offline_cache,
                policy.feature_spec,
                chunk_size=policy.head_config.chunk_size,
                reward_definition=self.config.algorithm.task_identity,
                gamma_step=self.learner.gamma_step,
                reference_num_steps=self.config.algorithm.get("reference_num_steps", 10),
                policy_switch=self.config.algorithm.get("policy_switch", {}),
            )
            for row in cache["train"]:
                replay.commit(row)
        return self.learner.update(0)

    def update(self, max_updates=1):
        incoming = []
        # One scalar/vector collector has one unacknowledged message. Replay has one owner.
        batched = False
        try:
            payload = self.channels.try_get("transitions")
            batched = isinstance(payload, list)
            incoming = payload if batched else [payload]
            if not 0 < len(incoming) <= self.batch_size:
                raise ValueError("Transition batch exceeds environment.batch_size.")
        except ChannelEmpty:
            pass
        admitted = self.learner.update(0, incoming=incoming)
        if admitted["acks"]:
            reply = admitted["acks"] if batched else admitted["acks"][0]
            self.channels.put_nowait("durable_acks", reply)
        result = self.learner.update(max_updates)
        result.pop("acks", None)
        return result

    def pending(self):
        return self.learner.replay.pending()

    def sync_model_to_rollout(self):
        self.learner.barrier()
        version = ActorVersion(
            self.run_id,
            self.epoch,
            getattr(self.learner, "actor_revision", self.learner.actor_updates),
        )
        snapshot = ActorSnapshot.capture(self.learner.actor, version, self.learner.spec.feature_id)
        self.channels.put_nowait("weights", snapshot)
        applied = self.channels.recv("weight_acks", timeout=self.timeout)
        if applied != version:
            raise ValueError("Actor applied ACK version mismatch.")
        return applied

    def rollout_rng(self):
        return getattr(self.learner, "rollout_rng", None)

    def snapshot(self, rollout_rng=None, feature_manifest_sha256=None):
        if rollout_rng is not None:
            self.learner.rollout_rng = rollout_rng
        path = save_resume(
            Path(self.config.exp.run_dir) / "rltoken/checkpoints",
            self.learner,
            task_fingerprint=self.task_fingerprint,
            run_id=self.run_id,
            feature_manifest="../../features/manifest.yaml",
            feature_manifest_sha256=feature_manifest_sha256,
            keep_last=self.config.train.get("keep_last_checkpoints"),
        )
        return str(path)

    def close(self):
        if self.learner is not None:
            self.learner.replay.close()
            self.learner = None
