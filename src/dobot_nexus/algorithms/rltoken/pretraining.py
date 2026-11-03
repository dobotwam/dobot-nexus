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
from .artifacts import feature_metadata, load_provider, retain_feature_bundle
from .checkpoint import load_resume, save_resume
from .configuration import HeadConfig, LearningConfig, validate_recipe
from .identity import content_digest
from .learner import create_learner
from .offline_data import build_cache, load_cache
from .replay import Replay
from .transitions import residual_batch
from .routing import PolicySwitch


def save_stage(path, learner, *, feature_manifest, task_identity, cache_id, validation, accepted):
    components = set(learner.initialized_components)
    if accepted:
        if learner.phase == "actor_bc":
            components.add("actor")
        if learner.phase == "base_critic":
            components.add("base")
    state = learner.state_dict()
    payload = dict(format="rltoken_stage_v1", spec=learner.spec.to_dict(), heads=learner.heads,
                   training_scope=learner.replay.training_scope,
                   task_identity=task_identity, cache_id=cache_id, validation=validation,
                   accepted=bool(accepted), components=sorted(components), feature_manifest=feature_manifest, state=state)
    path = Path(path)
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    torch.save({"digest": content_digest(payload), "payload": payload}, temporary)
    from .checkpoint import _fsync, _sync_dir
    _fsync(temporary)
    temporary.replace(path)
    _sync_dir(path.parent)


def read_stage(path, learner, *, task_identity, require_accepted=True):
    wrapper = torch.load(path, map_location="cpu", weights_only=False)
    payload = wrapper["payload"]
    if (content_digest(payload) != wrapper["digest"] or payload["format"] != "rltoken_stage_v1"
            or content_digest(payload["spec"]) != content_digest(learner.spec.to_dict())
            or content_digest(payload["heads"]) != content_digest(learner.heads)
            or payload["task_identity"] != task_identity
            or payload.get("training_scope", "all") != learner.replay.training_scope
            or (require_accepted and not payload["accepted"])):
        raise ValueError("Stage artifact identity/integrity/acceptance mismatch.")
    return payload


def initialize_from_stage(learner, path, *, task_identity, component="all"):
    """New task initialization: weights only; fresh optimizers, targets, counters and budget."""
    payload = read_stage(path, learner, task_identity=task_identity)
    if component not in {"all", "actor", "base"}:
        raise ValueError("Initialization component must be all, actor or base.")
    state = payload["state"]
    required = {"actor", "base"} if component == "all" else {component}
    if not required.issubset(payload.get("components", [])):
        raise ValueError("Stage artifact lacks accepted Actor BC/Base Critic prerequisites.")
    learner.initialized_components = sorted(set(learner.initialized_components) | required)
    if component in {"all", "actor"}:
        learner.actor.load_state_dict(state["actor"])
    if component == "all":
        learner.critic.load_state_dict(state["critic"])
    elif component == "base":
        learner.critic.base.load_state_dict({k.removeprefix("base."): v for k, v in state["critic"].items()
                                            if k.startswith("base.")})
    learner.set_phase(learner.phase)
    if component in {"all", "actor"}:
        learner.actor_revision += 1
        learner.pretrained_ready = True
        learner.ready_revision = learner.actor_revision
    return payload


def validation_loss(learner, rows):
    # Validation must not advance target smoothing or training sample RNG.
    generator = torch.Generator(device=learner.device).manual_seed(0)
    total, count = 0., 0
    for start in range(0, len(rows), learner.config.batch_size):
        subset = rows[start:start + learner.config.batch_size]
        batch = residual_batch(subset, learner.heads.chunk_size, learner.device)
        with torch.no_grad():
            if learner.phase == "actor_bc":
                if not batch["supervision_mask"].any():
                    continue
                loss, _ = learner.actor_objective(batch)
            else:
                loss, _ = learner.critic_objective(batch, learner.target_values(batch, generator=generator), generator=generator)
        total += float(loss) * len(subset)
        count += len(subset)
    if count == 0:
        raise ValueError("Validation split has no eligible samples for this phase.")
    return total / count


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
                raise ValueError("Token validation report missing; declare an explicit existing-provider ablation to bypass.")
            validation = json.loads(report.read_text())
            if not validation.get("accepted") or validation.get("feature_id") != spec.feature_id:
                raise ValueError("Token dependence gate or feature identity failed.")
        provider, _, _ = load_provider(root)
        provider.to(cfg.runtime.get("device", "cpu"))
        episodes = import_from_string(cfg.data.episode_reader)(cfg.data)
        path = build_cache(provider, episodes, self.experiment.run_dir / "cache", **cfg.data.cache,
                           policy_switch=cfg.algorithm.get("policy_switch", {}))
        return TrainResult(str(self.experiment.run_dir), str(path), 0, {})


class RLTokenPretrainTrainer:
    def __init__(self, config, experiment):
        self.config, self.experiment = config, experiment
        PolicySwitch.from_config(config)
        self.heads, self.learning = HeadConfig(**config.model.rltoken), LearningConfig(**config.algorithm.learning)
        validate_recipe(config.algorithm.recipe, self.heads, self.learning)
        self.stage = config.algorithm.pretrain
        self.validation_policy = self.stage.get("validation_policy", "loss_threshold")
        if self.validation_policy not in {"loss_threshold", "diagnostic"}:
            raise ValueError("pretrain.validation_policy must be loss_threshold or diagnostic.")
        if self.stage.phase not in {"actor_bc", "base_critic", "residual_critic", "offline_joint"}:
            raise ValueError("Offline trainer requires an explicit offline phase.")
        if min(config.train.max_steps, config.train.save_every) <= 0:
            raise ValueError("Offline step budgets must be positive.")
        if self.validation_policy == "loss_threshold" and (
                "max_validation_loss" not in self.stage
                or not math.isfinite(self.stage.max_validation_loss)
                or self.stage.max_validation_loss < 0):
            raise ValueError("Declare pretrain.max_validation_loss as the initialization acceptance gate.")
        if not self.stage.get("task_identity"):
            raise ValueError("Declare versioned pretrain.task_identity for reward/task semantics.")
        if config.train.get("resume_from") and self.stage.get("initialize"):
            raise ValueError("Offline initialization and full resume are mutually exclusive.")

    def run(self):
        cfg, root = self.config, Path(self.experiment.run_dir) / "rltoken"
        root.mkdir(parents=True, exist_ok=True)
        resume = cfg.train.get("resume_from")
        feature_source = cfg.model.features.artifact
        if resume:
            resume_path = Path(resume)
            manifest = json.loads((resume_path / "manifest.json").read_text())
            feature_source = str((resume_path / manifest["feature_manifest"]).parent.resolve())
        retained = retain_feature_bundle(feature_source, root / "features")
        _, spec, _ = feature_metadata(retained)
        stage = self.stage
        gamma_step = self.learning.gamma ** (1 / self.heads.chunk_size) if self.learning.td3.get("discount_unit", "chunk") == "chunk" else self.learning.gamma
        cache = load_cache(cfg.data.cache_path, spec, chunk_size=self.heads.chunk_size,
                           reward_definition=stage.task_identity, gamma_step=gamma_step,
                           reference_num_steps=cfg.algorithm.get("reference_num_steps", 10),
                           policy_switch=cfg.algorithm.get("policy_switch", {}))
        task = content_digest({"cache": cache["identity"], "stage": {k: v for k, v in stage.items() if k != "initialize"},
                               "heads": self.heads, "learning": self.learning})
        resume = cfg.train.get("resume_from")
        journal = root / "offline.sqlite"
        if resume and not journal.exists():
            shutil.copyfile(Path(resume) / "journal.sqlite", journal)
        elif journal.exists() and not resume:
            raise ValueError("Existing offline journal requires resume or a new run directory.")
        replay = Replay(journal, spec, self.heads.chunk_size, **cfg.data.replay,
                        training_scope=PolicySwitch.from_config(cfg).training_scope,
                        updates_per_new_transition=self.learning.updates_per_new_transition,
                        allow_terminal_chunks=True, task_fingerprint=task, seed=self.learning.seed)
        try:
            learner = create_learner(spec, self.heads, self.learning, replay,
                                     device=cfg.runtime.get("device", "cpu"))
            learner.set_phase(stage.phase)
            run_id = uuid.uuid4().hex
            if resume:
                run_id = load_resume(resume, learner, task_fingerprint=task)
            else:
                for component, path in stage.get("initialize", {}).items():
                    initialize_from_stage(learner, path, task_identity=stage.task_identity, component=component)
                if stage.phase in {"residual_critic", "offline_joint"} and not {"actor", "base"}.issubset(learner.initialized_components):
                    raise ValueError("Residual/offline joint phases require accepted Actor and Base stage artifacts.")
                for row in cache["train"]:
                    if stage.phase == "actor_bc" and not residual_batch([row], self.heads.chunk_size, "cpu")["supervision_mask"].any():
                        continue
                    replay.commit(row)
            best = min([entry["validation_loss"] for entry in learner.stage_history] or [float("inf")])
            last_metrics = {}
            checkpoint = str(resume or "")
            for _ in range(learner.stage_steps, cfg.train.max_steps):
                last_metrics = learner.train_rows(replay.sample(self.learning.batch_size))
                if learner.stage_steps % cfg.train.save_every and learner.stage_steps != cfg.train.max_steps:
                    continue
                score = validation_loss(learner, cache["validation"])
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite offline validation loss.")
                accepted = self.validation_policy == "diagnostic" or score <= stage.max_validation_loss
                validation = dict(loss=score, policy=self.validation_policy)
                if self.validation_policy == "loss_threshold":
                    validation["max_loss"] = stage.max_validation_loss
                learner.stage_history.append(dict(phase=learner.phase, step=learner.stage_steps,
                    validation_loss=score, cache_id=cache["identity"], accepted=accepted,
                    validation_policy=self.validation_policy,
                    training_episodes=cache["metadata"]["training_episodes"]))
                feature_manifest = "features/manifest.yaml"
                arguments = dict(feature_manifest=feature_manifest, task_identity=stage.task_identity,
                                 cache_id=cache["identity"], validation=validation, accepted=accepted)
                save_stage(root / "last.pt", learner, **arguments)
                if score < best:
                    best = score
                    save_stage(root / "best.pt", learner, **arguments)
                checkpoint = str(save_resume(root / "checkpoints", learner, task_fingerprint=task,
                                             run_id=run_id, feature_manifest="../../features/manifest.yaml"))
                self.experiment.logger.info("RLToken phase=%s step=%d validation=%g accepted=%s",
                                            learner.phase, learner.stage_steps, score, accepted)
                last_metrics["validation_loss"] = score
            return TrainResult(str(self.experiment.run_dir), checkpoint, learner.stage_steps, last_metrics)
        finally:
            replay.close()
