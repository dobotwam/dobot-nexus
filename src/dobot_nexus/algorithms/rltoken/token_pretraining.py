"""Token-only job using the public model/data-adapter contracts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from dobot_nexus.core.config import clone_config, resolve_autocast
from dobot_nexus.core.contracts import TrainResult
from dobot_nexus.core.import_utils import import_from_string
from dobot_nexus.training.resume import capture_rng_state, restore_rng_state
from .artifacts import write_model_bundle, feature_metadata
from .checkpoint import _fsync, _publication_lock, _sync_dir
from .serialization import load_payload, save_payload
from .identity import content_digest


def _episode_ids(values):
    if not isinstance(values, (list, tuple)) or not values or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise ValueError("Token episode IDs must be nonempty lists of nonnegative integers.")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate token episode IDs.")
    return tuple(values)


def _step_number(name):
    if not re.fullmatch(r"step-[0-9]{9,}", name):
        raise ValueError(f"Invalid token publication step: {name!r}")
    return int(name[5:])


def _require_publisher_control(metadata):
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise ValueError("Token staging must be owned by the publisher and writable only by its owner.")


@contextmanager
def _regular_input(path, *, dir_fd=None, check_owner=False):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags, dir_fd=dir_fd), "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Token publication requires a regular file: {path}")
        if check_owner:
            _require_publisher_control(metadata)
        yield stream


def _checkpoint_identity(step, stream):
    # Hash the exact serialized generation, including for pre-generation checkpoints.
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return step, digest.hexdigest()


def _published_parent(root, expected_parent):
    checkpoint = root / "resume.pt"
    try:
        with _regular_input(root / "last") as stream:
            step = _step_number(stream.read().decode("ascii"))
    except FileNotFoundError:
        if not os.path.lexists(checkpoint):
            return None
        # A copied standalone checkpoint can seed a run without pointers. Its
        # digest must still match the exact file from which we loaded this step.
        if expected_parent is None:
            raise ValueError("Stale token writer: a resume checkpoint already exists.")
        step = expected_parent[0]
    with _regular_input(checkpoint) as stream:
        return _checkpoint_identity(step, stream)


def _validate_pending(fd):
    pending = os.fstat(fd)
    _require_publisher_control(pending)
    # Older publications can be traversable. Their children must also be
    # protected against modifications by another user during recovery.
    traversable = bool(pending.st_mode & 0o011)
    entries = set(os.listdir(fd))
    if entries - {"artifact", "best", "last", "resume.pt"}:
        raise ValueError("Unexpected token publication entries.")
    for entry in entries:
        mode = os.stat(entry, dir_fd=fd, follow_symlinks=False).st_mode
        expected = stat.S_ISDIR if entry == "artifact" else stat.S_ISREG
        if not expected(mode):
            raise ValueError(f"Invalid token publication entry: {entry}")

    def raise_walk_error(exc):
        raise exc

    if "artifact" in entries:
        if not {"last", "resume.pt"} <= entries:
            raise ValueError("Incomplete staged token publication.")
        for _, dirs, files, directory in os.fwalk("artifact", dir_fd=fd, onerror=raise_walk_error):
            if traversable:
                _require_publisher_control(os.fstat(directory))
            for name in dirs + files:
                mode = os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(f"Symlink or special file in staged token Artifact: {name}")
                if stat.S_ISREG(mode):
                    with _regular_input(name, dir_fd=directory, check_owner=traversable):
                        pass
    pointers = {}
    for name in entries - {"artifact"}:
        with _regular_input(name, dir_fd=fd, check_owner=traversable) as stream:
            if name in {"best", "last"}:
                pointers[name] = stream.read().decode("ascii")
                _step_number(pointers[name])
    if "best" in pointers and pointers["best"] != pointers.get("last"):
        raise ValueError("Inconsistent staged token pointers.")
    return entries, pointers


def _finish_publication(root):
    """Roll a committed staging directory forward under the run's publication lock."""
    pending = root / ".publishing"
    try:
        fd = os.open(pending, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError("Token .publishing must be a real directory.") from exc
    try:
        entries, pointers = _validate_pending(fd)
        if "artifact" in entries:
            os.rename("artifact", root / pointers["last"], src_dir_fd=fd)
            _sync_dir(root)
            os.fsync(fd)
        for name in ("best", "last", "resume.pt"):
            if name in entries:
                os.replace(name, root / name, src_dir_fd=fd)
        _sync_dir(root)
        os.fsync(fd)
        pending.rmdir()
        _sync_dir(root)
    finally:
        os.close(fd)
    return True


def _resume_state(root, resume, trusted):
    with _publication_lock(root):
        trust_selected = bool(resume) and trusted is True
        recovered = _finish_publication(root)
        if not resume and not recovered:
            return None, "", None
        if resume == "auto" or not resume:
            resume = root / "resume.pt"
            checkpoint = _regular_input(resume)
        else:
            checkpoint = open(resume, "rb")
        # An external checkpoint may seed an empty run. In an existing run the
        # checkpoint selected by this writer must still be the published parent.
        with checkpoint as stream:
            state = load_payload(stream, trusted_checkpoint=trust_selected)
            expected_parent = (_checkpoint_identity(state["step"], stream)
                               if (root / "resume.pt").exists() else None)
    return state, str(resume), expected_parent


def _publish_step(root, model, config, state, report, improved, *, expected_parent):
    name = f"step-{state['step']:09d}"
    with tempfile.TemporaryDirectory(dir=root, prefix=".token-") as private:
        staging = Path(private) / "publication"
        staging.mkdir(mode=0o700)
        bundle = write_model_bundle(staging / "artifact", model, config)
        _, spec, _ = feature_metadata(bundle)
        (bundle / "token_validation.json").write_text(json.dumps(
            {**report, "feature_id": spec.feature_id}, indent=2
        ))
        (staging / "last").write_text(name)
        if improved:
            (staging / "best").write_text(name)
        save_payload(dict(state, generation=uuid.uuid4().hex, rng=capture_rng_state()), staging / "resume.pt")
        with _regular_input(staging / "resume.pt") as stream:
            generation = _checkpoint_identity(state["step"], stream)
        for path in sorted(staging.rglob("*"), reverse=True):
            (_sync_dir if path.is_dir() else _fsync)(path)
        _sync_dir(staging)
        with _publication_lock(root):
            current = _published_parent(root, expected_parent)
            if current != expected_parent or (current is not None and state["step"] <= current[0]):
                raise ValueError(f"Stale token writer: expected parent {expected_parent}, published {current}.")
            if (root / name).exists() or os.path.lexists(root / ".publishing"):
                raise FileExistsError(f"Token checkpoint already published or pending: {name}")
            staging.rename(root / ".publishing")
            _sync_dir(root)
            _finish_publication(root)
    return generation


class TokenPretrainTrainer:
    def __init__(self, config, experiment):
        self.config, self.experiment = config, experiment
        settings = config.algorithm.token
        self.train_episodes = _episode_ids(settings.train_episodes)
        self.validation_episodes = _episode_ids(settings.validation_episodes)
        if set(self.train_episodes).intersection(self.validation_episodes):
            raise ValueError("Token train/validation must be nonempty disjoint episode sets.")
        if config.model.rltoken.get("training_mode") != "token_only":
            raise ValueError("TokenPretrainTrainer requires model.rltoken.training_mode=token_only.")
        if not config.data.get("norm_stats_path"):
            raise ValueError("Token training requires the frozen provider's existing normalization asset.")
        if min(config.train.max_steps, config.train.save_every, config.data.batch_size) <= 0:
            raise ValueError("Token budgets and batch size must be positive.")
        for key in ("min_zero_ratio", "min_shuffle_ratio"):
            if settings.get(key, 0) <= 1:
                raise ValueError(f"Declare token.{key} > 1 for the dependence gate.")

    def run(self):
        cfg, root = self.config, Path(self.experiment.run_dir) / "token"
        root.mkdir(parents=True, exist_ok=True)
        state, checkpoint, expected_parent = _resume_state(
            root, cfg.train.get("resume_from"), cfg.train.get("trust_resume_checkpoint", False)
        )
        settings = cfg.algorithm.token
        device = torch.device(cfg.runtime.get("device", "cpu"))
        precision = cfg.runtime.get("precision", cfg.model.get("precision", "fp32"))
        if precision not in {"fp32", "bf16"}:
            raise ValueError("TokenPretrainTrainer supports fp32 or bf16 precision.")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        autocast = dtype != torch.float32 and resolve_autocast(cfg) != "off"
        torch.manual_seed(int(cfg.exp.get("seed", 0)))
        model = import_from_string(cfg.model.target).from_config(cfg, mode="train").to(device)
        train_cfg, val_cfg = clone_config(cfg), clone_config(cfg)
        train_cfg.data.episode_ids = list(self.train_episodes)
        val_cfg.data.episode_ids = list(self.validation_episodes)
        train_cfg.data.num_workers = val_cfg.data.num_workers = 0
        loader_target = settings.get("data_loader")
        build_loader = (lambda conf: import_from_string(loader_target)(model, conf)) if loader_target else model.build_train_dataloader
        train_loader = build_loader(train_cfg)
        validation_loader = build_loader(val_cfg)
        # Final model config describes training data; normalization/prompt assets are shared.
        if not loader_target:
            model.build_train_dataloader(train_cfg)
        fingerprint = content_digest({"model": dict(cfg.model), "data": dict(cfg.data),
                                      "token": dict(settings), "optim": dict(cfg.get("optim", {})),
                                      "precision": precision, "seed": cfg.exp.get("seed", 0),
                                      **({"autocast": "off"} if resolve_autocast(cfg) == "off" else {})})
        optimizer = torch.optim.AdamW(model.get_param_groups(cfg.get("optim", {})),
            lr=cfg.get("optim", {}).get("lr", 1e-4), weight_decay=cfg.get("optim", {}).get("weight_decay", 0.))
        generator = torch.Generator().manual_seed(int(cfg.exp.get("seed", 0)))
        start, best = 0, float("inf")
        metrics = {}
        if state is not None:
            if state["fingerprint"] != fingerprint:
                raise ValueError("Token resume configuration mismatch.")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            generator.set_state(state["sampler"].cpu())
            restore_rng_state(state["rng"])
            start = state["step"]
            if (root / "best").is_file():
                best = state["best"]
            metrics = state.get("metrics", {})
            del state
        for step in range(start, cfg.train.max_steps):
            indices = torch.randint(len(train_loader.dataset), (cfg.data.batch_size,), generator=generator)
            raw = train_loader.collate_fn([train_loader.dataset[int(i)] for i in indices])
            model.train()
            batch = model.adapter.encode_train_batch(raw, device)
            with torch.autocast(device.type, dtype=dtype, enabled=autocast):
                output = model.training_step(batch)
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % cfg.train.get("log_every", 20) == 0:
                self.experiment.logger.info("Token step=%d train_loss=%g", step + 1, float(output.loss.detach()))
            if (step + 1) % cfg.train.save_every and step + 1 != cfg.train.max_steps:
                continue
            model.eval()
            values, count = {}, 0
            with torch.no_grad():
                for offset in range(0, len(validation_loader.dataset), cfg.data.batch_size):
                    rows = [validation_loader.dataset[i] for i in range(offset,
                            min(offset + cfg.data.batch_size, len(validation_loader.dataset)))]
                    raw = validation_loader.collate_fn(rows)
                    with torch.autocast(device.type, dtype=dtype, enabled=autocast):
                        result = model.training_step(model.adapter.encode_train_batch(raw, device))
                    for key, value in result.metrics.items():
                        values[key] = values.get(key, 0.) + float(value) * len(rows)
                    count += len(rows)
            if not count:
                raise ValueError("Token validation is empty.")
            metrics = {key: value / count for key, value in values.items()}
            accepted = (metrics["zero_ratio"] >= settings.min_zero_ratio and
                        metrics["shuffle_ratio"] >= settings.min_shuffle_ratio)
            improved = metrics["token_loss"] < best
            best = min(best, metrics["token_loss"])
            state = dict(fingerprint=fingerprint, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), sampler=generator.get_state(),
                         step=step + 1, best=best, metrics=metrics)
            report = dict(metrics, accepted=accepted, train_episodes=list(self.train_episodes),
                          validation_episodes=list(self.validation_episodes))
            expected_parent = _publish_step(root, model, train_cfg, state, report, improved,
                                            expected_parent=expected_parent)
            checkpoint = str(root / "resume.pt")
            self.experiment.logger.info("Token step=%d validation=%s accepted=%s", step+1, metrics, accepted)
        return TrainResult(str(self.experiment.run_dir), checkpoint, cfg.train.max_steps, metrics)
