"""RLToken's single coordinator. CUDA models live exclusively in its workers."""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

from dobot_nexus.core.config import to_plain_dict
from dobot_nexus.core.contracts import TrainResult

from .artifacts import feature_metadata
from .batching import environment_batch_size
from .configuration import HeadConfig, LearningConfig, recipe_identity, validate_recipe
from .identity import content_digest
from .routing import PolicySwitch


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
    if heads.residual is not None and len(heads.residual["rho"]) != config.model.action_dim:
        raise ValueError("Residual rho must declare one scale per model action dimension.")
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
    def __init__(self, config, experiment):
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
        try:
            from .execution_rlinf import RLinfExecution
        except ImportError as exc:
            raise ImportError(
                "RLToken execution requires the pinned RLinf scheduler and its dependencies. "
                "Install algorithms/rltoken/requirements-rlinf.txt; "
                "BC and deployment do not require RLinf."
            ) from exc
        execution = RLinfExecution(
            self.config,
            self.experiment.logger,
            self.run_id,
            self.epoch,
            self.task_fingerprint,
            self.resume_path,
        )
        try:
            return self._coordinate(execution)
        finally:
            execution.close()

    def _coordinate(self, execution):
        status = execution.start()
        config = self.config
        handles = {}
        starts = {}
        env_status = {"chunks": 0, "env_steps": 0}
        applied = execution.applied_actor_update
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

        def launch(name, handle):
            if name in handles:
                raise RuntimeError(f"Duplicate in-flight {name} call.")
            handles[name], starts[name] = handle, time.monotonic()

        while True:
            for name, handle in list(handles.items()):
                if time.monotonic() - starts[name] > config.runtime.get("rpc_timeout", 120):
                    raise TimeoutError(f"RLToken {name} exceeded its RPC deadline.")
                if not handle.done():
                    continue
                result = execution.wait(handle)
                del handles[name]
                del starts[name]
                if name == "learn":
                    status = result
                    episode_status_pending = False
                    if result["updates"]:
                        execution.update_rollout_weights(no_wait=True)
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
            collection_done = env_status["chunks"] >= config.environment.max_chunks
            can_update = status["budget"] > 0 and status.get("replay_size", status["admitted"]) >= self.learning.warmup
            if draining_episode and not can_update and not episode_status_pending:
                draining_episode = False
            stop = (status["critic_updates"] >= config.train.max_steps or
                    (collection_done and (not drain_on_stop or (not can_update and not episode_status_pending))))
            pause = pause or stop or status["critic_updates"] >= next_save
            pretrained = status.get("pretrained_ready", False)
            revision = status.get("ready_revision", status["actor_updates"])
            ready = (pretrained and applied >= revision) or (
                status["admitted"] >= self.learning.warmup
                and status["actor_updates"] >= config.environment.min_actor_updates
                and applied >= config.environment.min_actor_updates
            )
            if pause and not handles:
                execution.finish_sync()
                execution.update_rollout_weights(no_wait=False)
                applied = execution.applied_actor_update
                if self.learning.td3 is not None:
                    rollout_rng = execution.wait(execution.rollout.rng_state())
                    checkpoint = execution.wait(execution.actor.snapshot(rollout_rng=rollout_rng))
                else:
                    checkpoint = execution.wait(execution.actor.snapshot())
                if (
                    (pretrained and applied >= revision) or
                    (status["admitted"] >= self.learning.warmup
                     and applied >= config.environment.min_actor_updates)
                ):
                    root = self.experiment.run_dir / "rltoken"
                    destination = root / "deployments" / f"actor-{applied}-{uuid.uuid4().hex[:8]}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    execution.wait(execution.rollout.export_deployment(str(destination)))
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
                launch("env", execution.env.collect_chunk(actor_ready=ready))
            if (not pause or "env" in handles) and "rollout" not in handles:
                launch("rollout", execution.rollout.serve_slice())
            if (not pause or "env" in handles) and "learn" not in handles:
                remaining = max(0, config.train.max_steps - status["critic_updates"])
                launch(
                    "learn", execution.actor.update(max_updates=0 if pause or
                        (episode_updates and not draining_episode and not collection_done)
                        else min(1, remaining))
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
