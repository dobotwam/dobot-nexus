"""Standalone offline jobs; no distributed scheduler or concrete model imports."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path

import torch

from dobot_nexus.core.contracts import TrainResult
from dobot_nexus.core.import_utils import import_from_string

from .artifacts import feature_metadata, file_digest, load_provider, retain_feature_bundle
from .checkpoint import (
    _fsync,
    _sync_dir,
    load_resume,
    resume_feature_root,
    save_resume,
    validate_retention_config,
)
from .configuration import HeadConfig, LearningConfig, validate_recipe
from .identity import content_digest
from .learner import create_learner
from .offline_data import build_cache, load_cache
from .replay import Replay
from .routing import PolicySwitch
from .transitions import residual_batch


def _prewarm_identity(learner, task_identity):
    return dict(
        format="rltoken_prewarm_v2",
        spec=learner.spec.to_dict(),
        heads=learner.heads,
        gamma=learner.config.gamma,
        training_scope=learner.replay.training_scope,
        task_identity=task_identity,
        bc_action_space=learner.settings.bc_action_space,
        success_anchor=learner.settings.success_anchor,
        critic_output=learner.critic.settings.critic_output,
        critic_objective=learner.settings.critic_objective,
        latent_dim=learner.actor.latent_dim,
    )


def save_prewarm(path, learner, *, feature_manifest, task_identity, cache_id, metrics=None):
    payload = dict(
        **_prewarm_identity(learner, task_identity),
        learning=learner.config,
        cache_id=cache_id,
        feature_manifest=feature_manifest,
        metrics=dict(metrics or {}),
        state=learner._checkpoint_state_dict(),
    )
    path = Path(path)
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        torch.save({"digest": content_digest(payload), "payload": payload}, temporary)
        _fsync(temporary)
        temporary.replace(path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_prewarm(path, learner, *, task_identity):
    wrapper = torch.load(path, map_location="cpu", weights_only=False)
    payload = wrapper["payload"]
    if content_digest(payload) != wrapper["digest"]:
        raise ValueError("Prewarm payload digest mismatch.")
    for key, expected in _prewarm_identity(learner, task_identity).items():
        if key not in payload or content_digest(payload[key]) != content_digest(expected):
            raise ValueError(f"Prewarm {key} mismatch.")
    return payload


def initialize_from_prewarm(learner, path, *, task_identity, inherit_optimizer=True):
    if type(inherit_optimizer) is not bool:
        raise ValueError("inherit_prewarm_optimizer must be boolean.")
    payload = read_prewarm(path, learner, task_identity=task_identity)
    state = payload["state"]
    if int(state.get("actor_updates", 0)) < 1 or int(state.get("actor_revision", 0)) < 1:
        raise ValueError("Prewarm requires an actor update and a positive actor revision.")
    if inherit_optimizer:
        for name in ("actor_lr", "critic_lr"):
            saved = getattr(payload["learning"], name)
            expected = getattr(learner.config, name)
            if saved != expected:
                raise ValueError(f"Prewarm {name} mismatch: saved {saved}, configured {expected}.")
    for name in ("actor", "critic", "target", "target_actor"):
        getattr(learner, name).load_state_dict(state[name], strict=True)
    if inherit_optimizer:
        for name in ("actor_optimizer", "critic_optimizer", "actor_scheduler", "critic_scheduler"):
            getattr(learner, name).load_state_dict(state[name])
        learner.update_step = state["update_step"]
    learner.actor_revision = int(state["actor_revision"])
    learner.pretrained_ready = True
    learner.ready_revision = learner.actor_revision
    return payload


@torch.no_grad()
def validation_loss(learner, rows):
    generator = torch.Generator(device=learner.device).manual_seed(0)
    total = 0.0
    for start in range(0, len(rows), learner.config.batch_size):
        subset = rows[start : start + learner.config.batch_size]
        batch = residual_batch(subset, learner.heads.chunk_size, learner.device)
        target = learner.target_values(batch, generator=generator)
        loss, _ = learner.critic_objective(batch, target, generator=generator)
        total += float(loss) * len(subset)
    if not math.isfinite(total):
        raise FloatingPointError("Nonfinite offline validation loss.")
    return total / len(rows)


class RLTokenCacheTrainer:
    """algorithm.target entry; reader(config.data) yields finalized Episode objects."""

    def __init__(self, config, experiment):
        self.config, self.experiment = config, experiment

    def run(self):
        cfg = self.config
        root, spec, _ = feature_metadata(cfg.model.features.artifact)
        if cfg.data.get("require_token_acceptance", True):
            report = root / "token_validation.json"
            if not report.is_file():
                raise ValueError(
                    "Token validation report missing; declare an explicit existing-provider "
                    "ablation to bypass."
                )
            validation = json.loads(report.read_text())
            if not validation.get("accepted") or validation.get("feature_id") != spec.feature_id:
                raise ValueError("Token dependence gate or feature identity failed.")
        provider, _, _ = load_provider(cfg.model.features.artifact)
        provider.to(cfg.runtime.get("device", "cpu"))
        episodes = import_from_string(cfg.data.episode_reader)(cfg.data)
        path = build_cache(
            provider,
            episodes,
            self.experiment.run_dir / "cache",
            **cfg.data.cache,
            policy_switch=cfg.algorithm.get("policy_switch", {}),
        )
        return TrainResult(str(self.experiment.run_dir), str(path), 0, {})


class RLTokenPrewarmTrainer:
    def __init__(self, config, experiment):
        self.config, self.experiment = config, experiment
        self.switch = PolicySwitch.from_config(config)
        self.heads = HeadConfig(**config.model.rltoken)
        self.learning = LearningConfig(**config.algorithm.learning)
        if config.algorithm.recipe != "rlt_td3_residual_v1":
            raise ValueError("Joint prewarm requires recipe rlt_td3_residual_v1.")
        validate_recipe(config.algorithm.recipe, self.heads, self.learning)
        self.task_identity = config.algorithm.get("prewarm", {}).get(
            "task_identity", config.algorithm.get("task_identity")
        )
        if not isinstance(self.task_identity, str) or not self.task_identity.strip():
            raise ValueError("Declare prewarm.task_identity or algorithm.task_identity.")
        validate_retention_config(config.train)
        for name in ("max_steps", "save_every"):
            value = config.train.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"train.{name} must be a positive integer.")
        if config.data.replay.get("sampling", "uniform") != "uniform":
            raise ValueError("Joint prewarm requires a single uniform Replay pool.")

    def run(self):
        cfg, root = self.config, Path(self.experiment.run_dir) / "rltoken"
        root.mkdir(parents=True, exist_ok=True)
        resume = cfg.train.get("resume_from")
        feature_source = cfg.model.features.artifact
        expected_manifest_sha256 = None
        if resume == "auto":
            pointer = root / "checkpoints/latest_resume"
            resume = pointer.parent / pointer.read_text().strip()
        if resume:
            resume_path = Path(resume)
            manifest = json.loads((resume_path / "manifest.json").read_text())
            feature_source = resume_feature_root(resume_path, manifest)
            expected_manifest_sha256 = manifest.get("feature_manifest_sha256")
        retained = retain_feature_bundle(
            feature_source, root / "features", expected_manifest_sha256=expected_manifest_sha256
        )
        feature_manifest_sha256 = file_digest(retained / "manifest.yaml")
        _, spec, _ = feature_metadata(retained)
        cache = load_cache(
            cfg.data.cache_path,
            spec,
            chunk_size=self.heads.chunk_size,
            reward_definition=self.task_identity,
            gamma_step=self.learning.gamma,
            reference_num_steps=cfg.algorithm.get("reference_num_steps", 10),
            policy_switch=cfg.algorithm.get("policy_switch", {}),
        )
        task = content_digest(
            dict(cache=cache["identity"], heads=self.heads, learning=self.learning)
        )
        journal = root / "offline.sqlite"
        if resume and not journal.exists():
            shutil.copyfile(Path(resume) / "journal.sqlite", journal)
        elif journal.exists() and not resume:
            raise ValueError("Existing offline journal requires resume or a new run directory.")
        replay = Replay(
            journal,
            spec,
            self.heads.chunk_size,
            **cfg.data.replay,
            training_scope=self.switch.training_scope,
            updates_per_new_transition=self.learning.updates_per_new_transition,
            allow_terminal_chunks=True,
            task_fingerprint=task,
            seed=self.learning.seed,
        )
        try:
            learner = create_learner(
                spec, self.heads, self.learning, replay, device=cfg.runtime.get("device", "cpu")
            )
            if resume:
                run_id = load_resume(resume, learner, task_fingerprint=task)
            else:
                run_id = uuid.uuid4().hex
                for row in cache["train"]:
                    replay.commit(row)
            if len(replay.rows) != len(cache["train"]):
                raise ValueError(
                    "Prewarm Replay capacity and max_bytes must cover all training rows."
                )
            metrics = {}
            checkpoint = str(resume or "")
            while learner.update_step < cfg.train.max_steps:
                metrics = learner.train_rows(replay._sample_views(self.learning.batch_size))
                if (
                    learner.update_step % cfg.train.save_every
                    and learner.update_step != cfg.train.max_steps
                ):
                    continue
                if cache["validation"]:
                    metrics["validation_loss"] = validation_loss(learner, cache["validation"])
                save_prewarm(
                    root / "prewarm.pt",
                    learner,
                    task_identity=self.task_identity,
                    feature_manifest="features/manifest.yaml",
                    cache_id=cache["identity"],
                    metrics=metrics,
                )
                checkpoint = str(
                    save_resume(
                        root / "checkpoints",
                        learner,
                        task_fingerprint=task,
                        run_id=run_id,
                        feature_manifest="../../features/manifest.yaml",
                        feature_manifest_sha256=feature_manifest_sha256,
                        keep_last=cfg.train.get("keep_last_checkpoints"),
                    )
                )
                self.experiment.logger.info("RLToken prewarm step=%d", learner.update_step)
            return TrainResult(
                str(self.experiment.run_dir), checkpoint, learner.update_step, metrics
            )
        finally:
            replay.close()


RLTokenPretrainTrainer = RLTokenPrewarmTrainer
