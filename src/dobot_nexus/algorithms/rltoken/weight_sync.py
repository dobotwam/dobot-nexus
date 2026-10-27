"""Immutable Actor snapshots and atomic request-boundary application, without RPC imports."""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .identity import content_digest


@dataclass(frozen=True)
class ActorVersion:
    run_id: str
    learner_epoch: str
    actor_update: int

    def __post_init__(self):
        if (
            not self.run_id
            or not self.learner_epoch
            or type(self.actor_update) is not int
            or self.actor_update < 0
        ):
            raise ValueError("Actor version requires run/epoch and a nonnegative update count.")

    def as_tuple(self):
        return self.run_id, self.learner_epoch, self.actor_update


@dataclass(frozen=True)
class ActorSnapshot:
    version: ActorVersion
    feature_id: str
    state: dict[str, torch.Tensor]
    digest: str

    @classmethod
    def capture(cls, actor, version, feature_id):
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in actor.state_dict().items()
        }
        digest = content_digest((version, feature_id, state))
        return cls(version, feature_id, state, digest)


class WeightSlots:
    def __init__(self, actor, feature_id, run_id, learner_epoch):
        self.slots = [
            actor.eval().requires_grad_(False),
            copy.deepcopy(actor).eval().requires_grad_(False),
        ]
        self.active = 0
        self.feature_id, self.run_id, self.learner_epoch = feature_id, run_id, learner_epoch
        self.version = self.digest = None
        self.lock = threading.RLock()

    def apply(self, snapshot):
        if (snapshot.version.run_id, snapshot.version.learner_epoch, snapshot.feature_id) != (
            self.run_id,
            self.learner_epoch,
            self.feature_id,
        ):
            raise ValueError("Actor snapshot run, epoch or feature identity mismatch.")
        if (
            content_digest((snapshot.version, snapshot.feature_id, snapshot.state))
            != snapshot.digest
        ):
            raise ValueError("Actor snapshot checksum mismatch.")
        with self.lock:
            if self.version is not None:
                if snapshot.version.actor_update < self.version.actor_update:
                    raise ValueError("Stale Actor version.")
                if snapshot.version == self.version:
                    if snapshot.digest != self.digest:
                        raise ValueError("Conflicting snapshot for the same Actor version.")
                    return self.version
            standby = self.slots[1 - self.active]
            expected = standby.state_dict()
            if set(snapshot.state) != set(expected) or any(
                value.shape != expected[key].shape
                or value.dtype != expected[key].dtype
                or not torch.isfinite(value).all()
                for key, value in snapshot.state.items()
            ):
                raise ValueError("Actor snapshot parameter schema or values mismatch.")
            standby.load_state_dict(snapshot.state, strict=True)
            device = next(standby.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.active = 1 - self.active
            self.version, self.digest = snapshot.version, snapshot.digest
            return self.version  # Applied ACK, never just a receive ACK.

    @contextmanager
    def request(self):
        with self.lock:
            if self.version is None:
                raise RuntimeError("Initial Actor synchronization has not completed.")
            actor = self.slots[self.active]
            try:
                yield actor, self.version
            finally:
                # A retired slot cannot be overwritten while its kernels are still running.
                device = next(actor.parameters()).device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
