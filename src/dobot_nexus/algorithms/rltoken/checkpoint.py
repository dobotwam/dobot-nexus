"""Private multi-optimizer resume format. Published only at a complete update barrier."""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shutil
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


from dobot_nexus.training.resume import capture_rng_state, restore_rng_state

from .artifacts import _file_identity, _read_feature_manifest, file_digest
from .serialization import load_payload, save_payload
from .identity import content_digest


def _fsync(path):
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_CHECKPOINT = re.compile(r"step-([0-9]{9,})-[a-f0-9]{8}")
_DEPLOYMENT = re.compile(r"actor-(0|[1-9][0-9]*)-[a-f0-9]{8}")
_PRUNE_DIR = re.compile(r"\.prune-[a-f0-9]{32}")
_PENDING = re.compile(r"\.pending-actor-(?:0|[1-9][0-9]*)-[a-f0-9]{8}")
_PENDING_ENTRY = re.compile(rf"({_PENDING.pattern})(?:\.reserve|\.staging|\.stage\.lock)?")
_logger = logging.getLogger(__name__)


@contextmanager
def _publication_lock(root):
    """Serialize publishers across coordinators sharing this run directory."""
    import fcntl

    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"Publication lock requires a real directory: {root}") from exc
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _replace_text(path, text):
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        temporary.write_text(text, encoding="utf-8")
        _fsync(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_pointer(pointer, relative):
    _replace_text(pointer, str(relative))


def _pointer_state(pointer, expected_target):
    try:
        mode = pointer.lstat().st_mode
    except FileNotFoundError:
        return "inactive"
    except OSError:
        return "unknown"
    if not stat.S_ISREG(mode):
        return "unknown"
    try:
        target = pointer.read_text(encoding="utf-8").rstrip("\n")
    except (OSError, UnicodeError):
        return "unknown"
    return "active" if target == str(expected_target) else "inactive"


def _discard_directory(path):
    isolated = path.with_name(".prune-" + uuid.uuid4().hex)
    try:
        path.rename(isolated)
        shutil.rmtree(isolated)
    except OSError as exc:
        _logger.warning("Could not prune %s: %s", path, exc)


def _mark_published(root, published_dir, pattern):
    newest = 0
    for path in root.iterdir():
        if pattern.fullmatch(path.name):
            entry = path.lstat()
            if stat.S_ISDIR(entry.st_mode):
                newest = max(newest, entry.st_mtime_ns)
    stamp = max(time.time_ns(), newest + 1)
    os.utime(published_dir, ns=(stamp, stamp))
    _sync_dir(root)


def _managed_directories(root, pattern):
    managed = {}
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            entry = path.lstat()
            if stat.S_ISDIR(entry.st_mode):
                managed[path.name] = (entry.st_mtime_ns, int(match[1]), path.name)
    return managed


def _read_published_history(root):
    path = root / ".published.json"
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise ValueError(f"Expected a regular publication history: {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Invalid publication history: {path}")
    return entries


def _published_directories(entries, managed):
    """Keep the last occurrence of each surviving managed directory."""
    valid = (name for name in reversed(entries) if isinstance(name, str) and name in managed)
    return list(reversed(dict.fromkeys(valid)))


def _write_published_history(root, history):
    _replace_text(root / ".published.json", json.dumps(history))
    _sync_dir(root)


def _active_name(pointer, prefix):
    if not stat.S_ISREG(pointer.lstat().st_mode):
        raise ValueError("expected a regular pointer file")
    target = pointer.read_text(encoding="utf-8").rstrip("\n")
    if not target.startswith(prefix):
        raise ValueError(f"invalid active target {target!r}")
    return target[len(prefix):]


def _reconcile_published_history(root, pointer, prefix, pattern):
    """Persist the old pointer's publication before the next candidate becomes visible."""
    managed = _managed_directories(root, pattern)
    saved = _read_published_history(root)
    history = (sorted(managed, key=managed.__getitem__) if saved is None
               else _published_directories(saved, managed))
    try:
        active = _active_name(pointer, prefix)
    except (FileNotFoundError, ValueError):
        active = None
    if active in managed and history[-1:] != [active]:
        history = [name for name in history if name != active] + [active]
    if history != saved:
        _write_published_history(root, history)
    return history


def validate_keep_last(value, field_name):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{field_name} must be null or a positive integer.")


def validate_retention_config(train_cfg):
    for field_name in ("keep_last_checkpoints", "keep_last_deployments"):
        validate_keep_last(train_cfg.get(field_name), "train." + field_name)


def _prune(root, pointer, prefix, pattern, keep_last, expected):
    if keep_last is None:
        return
    validate_keep_last(keep_last, "keep_last")
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise ValueError("expected a real publication directory")
        active = _active_name(pointer, prefix)
        managed = _managed_directories(root, pattern)
        if active not in managed or (expected is not None and prefix + active != str(expected)):
            raise ValueError(f"invalid active target {active!r}")
        history = _read_published_history(root)
        ordered = (sorted(managed, key=managed.__getitem__) if history is None
                   else _published_directories(history, managed))
        others = [name for name in reversed(ordered) if name != active]
        survivors = [*reversed(others[:keep_last - 1]), active]
        keep = set(survivors)
        if history is not None and history != survivors:
            _write_published_history(root, survivors)
        isolated = [p for p in root.iterdir()
                    if _PRUNE_DIR.fullmatch(p.name) and stat.S_ISDIR(p.lstat().st_mode)]
    except (OSError, ValueError) as exc:
        _logger.warning("Skipping retention for %s: %s", pointer, exc)
        return

    for path in isolated:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            _logger.warning("Could not finish pruning %s: %s", path, exc)
    for old in sorted(managed.keys() - keep):
        _discard_directory(root / old)


def prune_checkpoints(root, keep_last, *, active_name=None):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        _logger.warning("Skipping retention for invalid checkpoint directory %s", root)
        return
    with _publication_lock(root):
        _prune(root, root / "latest_resume", "", _CHECKPOINT, keep_last, active_name)


def _discard_pending(pending):
    try:
        mode = pending.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        _discard_directory(pending)
    else:
        pending.unlink()


def _reservation_epoch(pending):
    reserve = pending.with_suffix(".reserve")
    try:
        mode = reserve.lstat().st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISREG(mode):
        return reserve.read_text(encoding="utf-8")
    return None


def _acquire_file_lock(path, flags):
    import fcntl

    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"Expected a regular export lock file: {path}")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        try:
            if flags & os.O_EXCL:
                path.unlink(missing_ok=True)
        finally:
            os.close(fd)
        raise


@contextmanager
def _export_reservation(pending, destination, epoch):
    root = pending.parent
    reserve = pending.with_suffix(".reserve")
    fd = None
    try:
        with _publication_lock(root):
            for path in (pending, destination, pending.with_suffix(".staging"),
                         pending.with_suffix(".stage.lock")):
                if os.path.lexists(path):
                    raise FileExistsError(path)
            fd = _acquire_file_lock(reserve, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            with os.fdopen(fd, "w", closefd=False, encoding="utf-8") as handle:
                handle.write(epoch)
                handle.flush()
                os.fsync(fd)
            _sync_dir(root)
        yield reserve
    except BaseException:
        if fd is not None:
            with _publication_lock(root):
                try:
                    current = reserve.lstat()
                except FileNotFoundError:
                    current = None
                if current is None or os.path.samestat(os.fstat(fd), current):
                    reserve.unlink(missing_ok=True)
                    _discard_pending(pending)
                    _sync_dir(root)
        raise
    finally:
        if fd is not None:
            os.close(fd)


@contextmanager
def _export_staging(path, epoch):
    coordinated = _PENDING.fullmatch(path.name)
    staging = (path.with_suffix(".staging") if coordinated
               else path.with_name("." + path.name + "-" + uuid.uuid4().hex))
    lock = path.with_suffix(".stage.lock")
    fd = None
    if coordinated:
        with _publication_lock(path.parent):
            if _reservation_epoch(path) != epoch:
                raise RuntimeError(f"Deployment export reservation missing or cancelled: {path}")
            fd = _acquire_file_lock(lock, os.O_CREAT | os.O_RDWR)
    try:
        yield staging
    finally:
        try:
            if staging.exists():
                shutil.rmtree(staging)
        finally:
            if fd is not None:
                try:
                    with _publication_lock(path.parent):
                        lock.unlink()
                finally:
                    os.close(fd)


@contextmanager
def _try_claim_dead_lock(path):
    """Called under the directory lock. Hold the claimed inode throughout cleanup."""
    fd = None
    try:
        try:
            if stat.S_ISREG(path.lstat().st_mode):
                fd = _acquire_file_lock(path, os.O_RDONLY)
        except FileNotFoundError:
            yield True
            return
        except BlockingIOError:
            yield False
            return
        yield fd is not None and os.path.samestat(os.fstat(fd), path.lstat())
    finally:
        if fd is not None:
            os.close(fd)


def _prune_pending_exports(root):
    # Recover each export independently, using the locks even for the current epoch.
    names = {m[1] for p in root.iterdir() if (m := _PENDING_ENTRY.fullmatch(p.name))}
    for name in sorted(names):
        pending = root / name
        reserve = pending.with_suffix(".reserve")
        staging = pending.with_suffix(".staging")
        lock = pending.with_suffix(".stage.lock")
        try:
            if reserve.is_file() and not reserve.is_symlink():
                with _try_claim_dead_lock(reserve) as dead:
                    if not dead:
                        continue
                    reserve.unlink(missing_ok=True)
            if pending.is_dir() and not pending.is_symlink():
                _discard_directory(pending)
            with _try_claim_dead_lock(lock) as dead:
                if dead:
                    if staging.is_dir() and not staging.is_symlink():
                        _discard_directory(staging)
                    lock.unlink(missing_ok=True)
        except OSError as exc:
            _logger.warning("Could not reclaim pending export %s: %s", pending, exc)


def _prune_deployments_locked(root, keep_last, active_relative):
    deployments = root / "deployments"
    _prune_pending_exports(deployments)
    _prune(deployments, root / "latest_deployment.txt", "deployments/",
           _DEPLOYMENT, keep_last, active_relative)


def prune_deployments(root, keep_last, *, active_relative=None, current_epoch=None):
    root = Path(root)
    deployments = root / "deployments"
    if deployments.is_symlink() or not deployments.is_dir():
        _logger.warning("Skipping retention for invalid deployments directory %s", deployments)
        return
    with _publication_lock(root), _publication_lock(deployments):
        _prune_deployments_locked(root, keep_last, active_relative)


@contextmanager
def _after_publication(pointer):
    try:
        yield
    except OSError as exc:
        _logger.warning("Could not update publication history for %s; skipping retention: %s",
                        pointer, exc)


def resume_feature_root(checkpoint, manifest):
    relative = manifest.get("feature_manifest")
    if not relative:
        raise FileNotFoundError("Resume requires its retained frozen feature bundle.")
    path = checkpoint / relative
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected a regular resume feature manifest: {path}")
    if "feature_manifest_sha256" in manifest:
        if path.name != "manifest.yaml" or manifest["feature_manifest_sha256"] != file_digest(path):
            raise ValueError("Resume feature manifest checksum mismatch.")
    return path.parent.resolve()


def save_resume(root, learner, *, task_fingerprint, run_id, feature_manifest=None, keep_last=None,
                feature_manifest_sha256=None):
    validate_keep_last(keep_last, "train.keep_last_checkpoints")
    state = learner._checkpoint_state_dict()
    state.update(rng=capture_rng_state(), task_fingerprint=task_fingerprint, run_id=run_id)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{learner.critic_updates:09d}-{uuid.uuid4().hex[:8]}"
    staging = root / ("." + name)
    staging.mkdir()
    try:
        save_payload(state, staging / "state.pt")
        backup = sqlite3.connect(staging / "journal.sqlite")
        try:
            learner.replay.connection.backup(backup)
        finally:
            backup.close()
        manifest = {
            "format": getattr(learner, "resume_format", "rltoken_resume_v1"),
            "state_digest": content_digest(state),
            "run_id": run_id,
            "feature_manifest": feature_manifest,
            "watermark": learner.replay.watermark,
            "task_fingerprint": task_fingerprint,
            "critic_updates": learner.critic_updates,
            "actor_updates": learner.actor_updates,
        }
        if feature_manifest is not None:
            target = staging / feature_manifest
            if target.name != "manifest.yaml" or not stat.S_ISDIR(target.parent.lstat().st_mode):
                raise ValueError(f"Expected a retained feature manifest: {target}")
            _file_identity(target)
            _, digest = _read_feature_manifest(target.parent, run_owned=True)
            if feature_manifest_sha256 is not None and feature_manifest_sha256 != digest:
                raise ValueError("Resume feature manifest checksum mismatch.")
            feature_manifest_sha256 = digest
        if feature_manifest_sha256 is not None:
            manifest["feature_manifest_sha256"] = feature_manifest_sha256
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for path in staging.iterdir():
            _fsync(path)
        _sync_dir(staging)
        final = root / name
        pointer = root / "latest_resume"
        with _publication_lock(root):
            history = _reconcile_published_history(root, pointer, "", _CHECKPOINT)
            staging.rename(final)
            try:
                _mark_published(root, final, _CHECKPOINT)
                _replace_pointer(pointer, name)
                _sync_dir(root)
            except BaseException:
                pointer_state = _pointer_state(pointer, name)
                if pointer_state == "inactive":
                    _discard_directory(final)
                else:
                    _sync_dir(root)
                raise
            with _after_publication(pointer):
                _write_published_history(root, [*history, name])
                _prune(root, pointer, "", _CHECKPOINT, keep_last, name)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


def load_resume(path, learner, *, task_fingerprint, trusted_checkpoint=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != getattr(learner, "resume_format", "rltoken_resume_v1"):
        raise ValueError("Not an RLToken multi-optimizer checkpoint.")
    state = load_payload(path / "state.pt", trusted_checkpoint=trusted_checkpoint)
    if content_digest(state) != manifest["state_digest"]:
        raise ValueError("Incomplete or corrupt RLToken checkpoint.")
    if (
        state["task_fingerprint"] != task_fingerprint
        or manifest["task_fingerprint"] != task_fingerprint
    ):
        raise ValueError("RLToken task/reward fingerprint mismatch.")
    learner._load_checkpoint_state_dict(state)
    restore_rng_state(state["rng"])
    return state["run_id"]
