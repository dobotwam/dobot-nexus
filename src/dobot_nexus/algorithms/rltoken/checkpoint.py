"""Private multi-optimizer resume format. Published only at a complete update barrier."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path

import torch

from dobot_nexus.training.resume import capture_rng_state, restore_rng_state

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


def save_resume(root, learner, *, task_fingerprint, run_id, feature_manifest=None):
    state = learner.state_dict()
    state.update(rng=capture_rng_state(), task_fingerprint=task_fingerprint, run_id=run_id)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{learner.critic_updates:09d}-{uuid.uuid4().hex[:8]}"
    staging = root / ("." + name)
    staging.mkdir()
    torch.save(state, staging / "state.pt")
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
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in staging.iterdir():
        _fsync(path)
    _sync_dir(staging)
    final = root / name
    staging.rename(final)
    pointer = root / (".latest-" + uuid.uuid4().hex)
    pointer.write_text(name, encoding="utf-8")
    _fsync(pointer)
    pointer.replace(root / "latest_resume")
    _sync_dir(root)
    return final


def load_resume(path, learner, *, task_fingerprint):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != getattr(learner, "resume_format", "rltoken_resume_v1"):
        raise ValueError("Not an RLToken multi-optimizer checkpoint.")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if content_digest(state) != manifest["state_digest"]:
        raise ValueError("Incomplete or corrupt RLToken checkpoint.")
    if (
        state["task_fingerprint"] != task_fingerprint
        or manifest["task_fingerprint"] != task_fingerprint
    ):
        raise ValueError("RLToken task/reward fingerprint mismatch.")
    learner.load_state_dict(state)
    restore_rng_state(state["rng"])
    return state["run_id"]
