"""RLToken's single coordinator. CUDA models live exclusively in its workers."""

from __future__ import annotations

import json
import math
import time
import uuid
from functools import partial
from pathlib import Path

from dobot_nexus.core.config import to_plain_dict
from dobot_nexus.core.contracts import TrainResult
from dobot_nexus.runtime.factory import create_runtime
from dobot_nexus.runtime.sync import SingleInFlightSyncCoordinator

from .artifacts import feature_metadata
from .batching import environment_batch_size
from .configuration import (
    HeadConfig,
    LearningConfig,
    ResidualConfig,
    recipe_identity,
    validate_recipe,
)
from .identity import content_digest
from .routing import PolicySwitch
from .topology import build_rltoken_topology
from .weight_sync import ActorVersion


def validate_training(config):
    PolicySwitch.from_config(config)
    runtime = config.get("runtime", {})
    train = config.get("train", {})
    if runtime.get("launcher") != "rlinf":
        raise ValueError("RLTokenTrainer requires runtime.launcher=rlinf (one driver).")
    if runtime.get("fsdp", {}).get("enabled", False):
        raise ValueError("RLToken Stage 2 does not support FSDP.")
    if train.get("ema_decay") is not None or train.get("grad_accum_steps", 1) != 1:
        raise ValueError("RLToken Stage 2 requires EMA disabled and grad_accum_steps=1.")
    if config.get("optim") or config.get("scheduler"):
        raise ValueError(
            "Stage 2 uses algorithm.learning optimizers/constant schedules; "
            "remove BC optim/scheduler."
        )
    if config.model.target != "dobot_nexus.algorithms.rltoken.policy:RLTokenPolicy":
        raise ValueError("Stage 2 model.target must be RLTokenPolicy.")
    if config.model.get("checkpoint") or config.model.get("artifact"):
        raise ValueError(
            "Stage 2 starts from model.features.artifact; resume via train.resume_from."
        )
    heads = HeadConfig(**config.model.rltoken)
    learning = LearningConfig(**config.algorithm.learning)
    validate_recipe(config.algorithm.get("recipe", "rlt_ac_fixed_v1"), heads, learning)
    if config.algorithm.get("initialize_from") and train.get("resume_from"):
        raise ValueError("initialize_from starts a new run; resume_from restores an existing run.")
    if config.algorithm.get("initialize_from") and (learning.td3 is None or not config.algorithm.get("task_identity")):
        raise ValueError("Stage initialization requires the TD3 recipe and task_identity.")
    if heads.residual is not None:
        ResidualConfig(**heads.residual).validate_scale(heads.chunk_size, config.model.action_dim)
    if learning.td3 is not None and not config.algorithm.get("task_identity"):
        raise ValueError("TD3 requires a versioned task_identity for reward semantics.")
    if type(config.algorithm.get("reference_num_steps", 10)) is not int or config.algorithm.get("reference_num_steps", 10) <= 0:
        raise ValueError("reference_num_steps must be a positive integer.")
    if heads.chunk_size != config.model.action_horizon:
        raise ValueError("Public action_horizon must equal fixed chunk_size.")
    for name in ("max_steps", "save_every"):
        if type(train.get(name)) is not int or train[name] <= 0:
            raise ValueError(f"train.{name} must be a positive integer.")
    env = config.get("environment", {})
    environment_batch_size(env)
    if not isinstance(env.get("target"), str) or ":" not in env["target"]:
        raise ValueError(
            "environment.target must select a Robot Agent adapter using module:object."
        )
    if env.get("reward_mode") not in {"per_step", "episode"} or type(env.get("bootstrap_truncation")) is not bool:
        raise ValueError(
            "Declare environment.reward_mode=per_step or episode and boolean bootstrap_truncation."
        )
    for name in ("max_chunks", "min_actor_updates"):
        if type(env.get(name)) is not int or env[name] <= 0:
            raise ValueError(f"environment.{name} must be a positive integer.")
    if env.get("update_schedule", "continuous") not in {"continuous", "episode_end"}:
        raise ValueError("update_schedule must be continuous or episode_end.")
    timeout = env.get("timeout", 0)
    rpc_timeout = runtime.get("rpc_timeout", 120)
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or not math.isfinite(rpc_timeout)
        or rpc_timeout <= 2 * timeout
    ):
        raise ValueError("rpc_timeout must exceed twice the positive environment timeout.")
    resources = runtime.get("resources", {})
    env_gpu = resources.get("env_gpu")
    if env_gpu is not None and (
        env.get("kind", "robot") != "simulation" or type(env_gpu) is not int or env_gpu < 0
    ):
        raise ValueError("env_gpu requires simulation mode and a nonnegative GPU index.")
    indices = [resources.get(key) for key in ("rollout_gpu", "learner_gpu")]
    if any(type(index) is not int or index < 0 for index in indices) or indices[0] == indices[1]:
        raise ValueError("Assign distinct nonnegative rollout_gpu and learner_gpu indices.")
    replay = config.data.get("replay", {})
    if replay.get("capacity", 100000) < learning.warmup:
        raise ValueError("Replay capacity must cover learning.warmup.")
    return heads, learning


class RLTokenTrainer:
    def __init__(self, config, experiment, *, runtime_factory=None):
        self.runtime_factory = runtime_factory
        self.config, self.experiment = config, experiment
        self.heads, self.learning = validate_training(config)
        self.resume_path = self._resume_path()
        if self.resume_path:
            manifest = json.loads((self.resume_path / "manifest.json").read_text())
            feature_manifest = manifest.get("feature_manifest")
            if not feature_manifest or not (self.resume_path / feature_manifest).is_file():
                raise FileNotFoundError("Resume requires its retained frozen feature bundle.")
            config.model.features.artifact = str(
                (self.resume_path / feature_manifest).parent.resolve()
            )
        _, spec, _ = feature_metadata(config.model.features.artifact)
        if (
            spec.reference_horizon < self.heads.chunk_size
            or spec.action_dim != config.model.action_dim
        ):
            raise ValueError("Feature spec does not cover the configured fixed action chunk.")
        self.task_fingerprint = content_digest(
            {
                **PolicySwitch.from_config(config).identity(),
                **({"task_identity": config.algorithm.get("task_identity"),
                    "reference_num_steps": config.algorithm.get("reference_num_steps", 10)}
                   if self.learning.td3 is not None else {}),
                "feature_spec": spec,
                "heads": self.heads,
                "learning": self.learning,
                "rollout_seed": config.exp.get("seed", 0),
                "environment": to_plain_dict(config.environment),
                "replay": to_plain_dict(config.data.replay),
                "recipe": recipe_identity(config.algorithm.get("recipe", "rlt_ac_fixed_v1")),
            }
        )
        self.run_id = uuid.uuid4().hex
        if self.resume_path:
            manifest = json.loads((self.resume_path / "manifest.json").read_text())
            if manifest.get("task_fingerprint") != self.task_fingerprint:
                raise ValueError("Resume task/reward fingerprint mismatch.")
            self.run_id = manifest["run_id"]
        self.epoch = uuid.uuid4().hex

    def _resume_path(self):
        value = self.config.train.get("resume_from")
        if not value:
            return None
        if value == "auto":
            pointer = self.experiment.run_dir / "rltoken/checkpoints/latest_resume"
            if not pointer.is_file():
                raise FileNotFoundError("No complete RLToken checkpoint for resume_from=auto.")
            return pointer.parent / pointer.read_text().strip()
        path = Path(value).expanduser().resolve()
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(f"RLToken resume manifest missing: {path}")
        return path

    def run(self):
        topology = build_rltoken_topology(
            self.config,
            run_id=self.run_id,
            epoch=self.epoch,
            task_fingerprint=self.task_fingerprint,
            resume_path=self.resume_path,
        )
        runtime = (self.runtime_factory or create_runtime)(self.config, self.experiment.logger)
        session = runtime.start(topology)
        try:
            learner, rollout = session.role("learner"), session.role("rollout")
            status = session.init_results["learner"]
            if self.learning.td3 is not None and self.resume_path:
                rng = learner.call("rollout_rng").wait()
                rollout.call("restore_rng", rng).wait()
            if self.config.environment.reward_mode == "episode":
                pending = learner.call("pending").wait()
                session.role("env").call("restore_pending", pending).wait()
            sync = SingleInFlightSyncCoordinator()
            self._sync_weights(session, sync, status)
            return self._coordinate(session, sync, status)
        finally:
            session.close()

    def _sync_weights(self, session, sync, status, *, blocking=True):
        recv = partial(session.role("rollout").call, "request_actor_sync_model")
        send = partial(session.role("learner").call, "sync_model_to_rollout")
        expected = ActorVersion(
            self.run_id, self.epoch, status.get("actor_revision", status["actor_updates"])
        )

        def validate(received, sent):
            if received != expected or sent != expected:
                raise ValueError("Synchronized Actor version mismatch.")

        if blocking:
            sync.sync_blocking(recv, send, validator=validate)
            return expected
        sync.sync_async(recv, send, validator=validate)

    def _ready(self, status, applied):
        minimum = self.config.environment.min_actor_updates
        if status.get("pretrained_ready") and applied >= status["ready_revision"]:
            return True
        return (
            status["admitted"] >= self.learning.warmup
            and status["actor_updates"] >= minimum
            and applied >= minimum
        )

    def _coordinate(self, session, sync, status):
        learner, rollout, env = (session.role(name) for name in ("learner", "rollout", "env"))
        config = self.config
        handles = {}
        env_status = {"chunks": 0, "env_steps": 0}
        applied = sync.last_result[0].actor_update
        next_save = (
            status["critic_updates"] // config.train.save_every + 1
        ) * config.train.save_every
        checkpoint = ""
        deployed = False
        last_log = time.monotonic()
        pause = status["critic_updates"] >= config.train.max_steps
        episode_updates = config.environment.get("update_schedule", "continuous") == "episode_end"
        draining_episode = False
        episode_status_pending = False
        drain_on_stop = config.environment.get("drain_on_stop", False)

        while True:
            for name, result in session.wait_any(handles):
                del handles[name]
                if name == "learn":
                    status = result
                    episode_status_pending = False
                    if result["updates"]:
                        self._sync_weights(session, sync, status, blocking=False)
                elif name == "rollout":
                    applied = result["applied_actor_update"]
                else:
                    env_status = result
                    if episode_updates and result.get("episode_ended"):
                        draining_episode = True
                        episode_status_pending = True
                    if result.get("faulted"):
                        raise RuntimeError(
                            "Robot execution requires reconciliation; "
                            "the record was durably journaled."
                        )
            sync.poll()
            collection_done = env_status["chunks"] >= config.environment.max_chunks
            can_update = (
                status["budget"] > 0
                and status.get("replay_size", status["admitted"]) >= self.learning.warmup
            )
            if draining_episode and not can_update and not episode_status_pending:
                draining_episode = False
            stop = status["critic_updates"] >= config.train.max_steps or (
                collection_done
                and (not drain_on_stop or (not can_update and not episode_status_pending))
            )
            pause = pause or stop or status["critic_updates"] >= next_save
            ready = self._ready(status, applied)
            if pause and not handles:
                sync.drain()
                version = self._sync_weights(session, sync, status)
                applied = version.actor_update
                if self.learning.td3 is not None:
                    rollout_rng = rollout.call("rng_state").wait()
                    checkpoint = learner.call("snapshot", rollout_rng=rollout_rng).wait()
                else:
                    checkpoint = learner.call("snapshot").wait()
                ready = self._ready(status, applied)
                if ready:
                    root = self.experiment.run_dir / "rltoken"
                    destination = root / "deployments" / f"actor-{applied}-{uuid.uuid4().hex[:8]}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    exported = rollout.call("export_deployment", str(destination)).wait()
                    exported_version = ActorVersion(
                        exported["run_id"], exported["learner_epoch"], exported["actor_update"]
                    )
                    if exported_version != version or exported["path"] != str(destination):
                        raise ValueError("Deployment differs from the final Actor version or path.")
                    from .checkpoint import _fsync, _sync_dir

                    pointer = root / (".latest-deployment-" + uuid.uuid4().hex)
                    pointer.write_text(str(destination.relative_to(root)))
                    _fsync(pointer)
                    pointer.replace(root / "latest_deployment.txt")
                    _sync_dir(root)
                    deployed = True
                self.experiment.logger.info(
                    "RLToken checkpoint step=%d actor=%d path=%s",
                    status["critic_updates"],
                    status["actor_updates"],
                    checkpoint,
                )
                if stop:
                    break
                next_save = status["critic_updates"] + config.train.save_every
                pause = False
            # During a checkpoint barrier, keep servicing the last chunk until its durable ACK.
            if not pause and not collection_done and not draining_episode and "env" not in handles:
                handles["env"] = env.call("collect_chunk", actor_ready=ready)
            if (not pause or "env" in handles) and "rollout" not in handles:
                handles["rollout"] = rollout.call("serve_slice")
            if (not pause or "env" in handles) and "learn" not in handles:
                remaining = max(0, config.train.max_steps - status["critic_updates"])
                collect_only = pause or (
                    episode_updates and not draining_episode and not collection_done
                )
                handles["learn"] = learner.call(
                    "update", max_updates=0 if collect_only else min(1, remaining)
                )
            if time.monotonic() - last_log > 10:
                self.experiment.logger.info(
                    "RLToken step=%d actor=%d chunks=%d budget=%d",
                    status["critic_updates"],
                    status["actor_updates"],
                    env_status["chunks"],
                    status["budget"],
                )
                last_log = time.monotonic()
            time.sleep(0.01)
        metrics = {
            **status.get("metrics", {}),
            "actor_updates": float(status["actor_updates"]),
            "env_steps": float(env_status["env_steps"]),
            "chunks": float(env_status["chunks"]),
            "admitted": float(status["admitted"]),
            "budget": float(status["budget"]),
            "deployment_ready": float(deployed),
            "stop_update_limit": float(status["critic_updates"] >= config.train.max_steps),
            "stop_collection_limit": float(env_status["chunks"] >= config.environment.max_chunks),
        }
        return TrainResult(
            str(self.experiment.run_dir), checkpoint, status["critic_updates"], metrics
        )
