"""Single-owner CPU Replay with SQLite durable journal and explicit UTD budget."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sqlite3
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .serialization import load_payload, save_payload
from .identity import content_digest


def _dump(value):
    stream = io.BytesIO()
    save_payload(value, stream)
    return stream.getvalue()


def _load(payload, *, trusted_checkpoint=False):
    return load_payload(io.BytesIO(payload), trusted_checkpoint=trusted_checkpoint)


@dataclass(frozen=True)
class DurableAck:
    key: tuple[str, ...]
    status: str
    reason: str
    sequence: int
    digest: str


class _IndexedRing:
    """A FIFO with amortized constant-time append and constant-time random access."""

    def __init__(self):
        self._slots = [None] * 8
        self._head = self._size = 0

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if not 0 <= index < self._size:
            raise IndexError(index)
        return self._slots[(self._head + index) % len(self._slots)]

    def append(self, row):
        if self._size == len(self._slots):
            self._slots = [self[i] for i in range(self._size)] + [None] * self._size
            self._head = 0
        self._slots[(self._head + self._size) % len(self._slots)] = row
        self._size += 1

    def popleft(self):
        row = self[0]
        self._slots[self._head] = None
        self._head = (self._head + 1) % len(self._slots)
        self._size -= 1
        return row


class Replay:
    def __init__(
        self,
        path,
        spec,
        chunk_size,
        *,
        capacity=100000,
        max_bytes=2**30,
        max_record_bytes=16 * 2**20,
        updates_per_new_transition=1,
        seed=0,
        task_fingerprint="",
        allow_terminal_chunks=False,
        sampling="uniform",
        demo_capacity=100000,
        demo_max_bytes=2**30,
        training_scope="all",
        trusted_checkpoint=False,
        resume_journal=None,
    ):
        if training_scope not in {"all", "critical_phase"}:
            raise ValueError("Replay training_scope must be all or critical_phase.")
        self.trusted_checkpoint = trusted_checkpoint
        self._trusted_payloads = None
        if trusted_checkpoint is True and resume_journal is not None:
            # The live journal can extend (or differ from) the selected checkpoint.
            # Only bytes actually present in its snapshot inherit resume trust.
            uri = Path(resume_journal).resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as snapshot:
                self._trusted_payloads = {
                    hashlib.sha256(payload).digest()
                    for payload, in snapshot.execute("SELECT payload FROM events")
                }
        self.training_scope = training_scope
        if min(capacity, max_bytes, max_record_bytes) <= 0:
            raise ValueError("Replay bounds must be positive.")
        if type(updates_per_new_transition) is not int or updates_per_new_transition < 0:
            raise ValueError("UTD budget must be a nonnegative integer.")
        self.spec, self.chunk_size = spec, chunk_size
        self.capacity, self.max_bytes, self.max_record_bytes = capacity, max_bytes, max_record_bytes
        self.utd = updates_per_new_transition
        if type(allow_terminal_chunks) is not bool:
            raise ValueError("allow_terminal_chunks must be a boolean.")
        self.allow_terminal_chunks = allow_terminal_chunks
        if sampling not in {"uniform", "demo_joint_50"} or min(demo_capacity, demo_max_bytes) <= 0:
            raise ValueError("Invalid Replay pool configuration.")
        self.sampling, self.demo_capacity, self.demo_max_bytes = sampling, demo_capacity, demo_max_bytes
        self.last_sample_demo_fraction = 0.0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)")
        identity = content_digest(
            {"spec": spec, "chunk_size": chunk_size, "utd": self.utd, "task": task_fingerprint,
             **({"pools": (sampling, capacity, max_bytes, demo_capacity, demo_max_bytes)}
                if sampling != "uniform" else {}),
             **({"training_scope": training_scope} if training_scope != "all" else {}),
             **({"terminal_chunks": "candidate_completion_v2"} if allow_terminal_chunks else {})}
        )
        old = self.connection.execute("SELECT identity FROM metadata").fetchone()
        if old and old[0] != identity:
            self.connection.close()
            raise ValueError("Replay feature/recipe identity mismatch; create a new partition.")
        if not old:
            self.connection.execute("INSERT INTO metadata VALUES (?)", (identity,))
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "key TEXT NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL, "
            "reason TEXT NOT NULL, payload BLOB NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS latest (key TEXT PRIMARY KEY, sequence INTEGER NOT NULL)"
        )
        self.connection.commit()
        self.identity = identity
        self.rows = OrderedDict()
        self.episode_outcomes: dict[tuple[str, ...], bool] = {}
        self.bytes = self.budget = self.watermark = self.admitted = self.online_admitted = 0
        self.status_counts = {"pending": 0, "admitted": 0, "rejected": 0}
        self.generator = torch.Generator().manual_seed(seed)
        self._rebuild_rings()
        try:
            self._replay_after(0)
        except BaseException:
            self.connection.close()
            raise

    def _rebuild_rings(self):
        if self.sampling == "uniform":
            self._ring = _IndexedRing()
            for row, _ in self.rows.values():
                self._ring.append(row)
        else:
            self._demo_ring, self._online_ring = _IndexedRing(), _IndexedRing()
            self._demo_bytes = self._online_bytes = 0
            for row, size in self.rows.values():
                if self.is_demo(row):
                    self._demo_ring.append(row)
                    self._demo_bytes += size
                else:
                    self._online_ring.append(row)
                    self._online_bytes += size

    def _append_bounded(self, ring, row, pool_bytes, capacity, byte_limit):
        ring.append(row)
        while len(ring) > capacity or pool_bytes > byte_limit:
            _, size = self.rows.pop(ring.popleft().key)
            self.bytes -= size
            pool_bytes -= size
        return pool_bytes

    def _check_episode_outcome(self, row, status, reason) -> bool | None:
        """Return the finalized label, including feedback from outside the training scope."""
        if status != "admitted" and (status, reason) != ("rejected", "outside_training_scope"):
            return None
        if not row.execution_confirmed or not row.reward_finalized:
            return None
        if type(row.episode_success) is not bool:
            return None
        previous = self.episode_outcomes.get(row.observation_id.session)
        if previous is not None and previous != row.episode_success:
            raise ValueError(f"Conflicting episode outcome for {row.observation_id.session}.")
        return row.episode_success

    def _index_episode_outcome(self, row, status, reason):
        outcome = self._check_episode_outcome(row, status, reason)
        if outcome is not None:
            self.episode_outcomes[row.observation_id.session] = outcome

    def _apply(self, sequence, status, reason, row, size):
        self._index_episode_outcome(row, status, reason)
        self.watermark = sequence
        self.status_counts[status] += 1
        if status == "admitted":
            self.rows[row.key] = (row, size)
            self.bytes += size
            self.admitted += 1
            if row.origin not in {"demo", "vla"}:
                self.budget += self.utd
                self.online_admitted += 1
            if self.sampling == "uniform":
                self._append_bounded(self._ring, row, self.bytes, self.capacity, self.max_bytes)
            elif self.is_demo(row):
                self._demo_bytes = self._append_bounded(
                    self._demo_ring, row, self._demo_bytes + size,
                    self.demo_capacity, self.demo_max_bytes,
                )
            else:
                self._online_bytes = self._append_bounded(
                    self._online_ring, row, self._online_bytes + size,
                    self.capacity, self.max_bytes,
                )

    def _load_event(self, payload):
        trusted = self.trusted_checkpoint is True and (
            self._trusted_payloads is None or hashlib.sha256(payload).digest() in self._trusted_payloads
        )
        return _load(payload, trusted_checkpoint=trusted)

    def _replay_after(self, watermark):
        if watermark == 0:
            self.episode_outcomes.clear()
        for sequence, status, reason, payload, digest in self.connection.execute(
            "SELECT sequence,status,reason,payload,digest FROM events "
            "WHERE sequence>? ORDER BY sequence",
            (watermark,),
        ):
            row = self._load_event(payload)
            if content_digest(row) != digest:
                raise ValueError(f"Corrupt Replay journal event at sequence {sequence}.")
            self._apply(sequence, status, reason, row, len(payload))

    def pending(self):
        return [self._load_event(row[0]) for row in self.connection.execute(
            "SELECT e.payload FROM latest l JOIN events e ON e.sequence=l.sequence WHERE e.status='pending'")]

    def commit(self, transition, *, finalize=False):
        from .transitions import Finalization
        if isinstance(transition, Finalization):
            if type(transition.accepted) is not bool:
                raise ValueError("Finalization accepted must be a boolean.")
            prior = self.connection.execute(
                "SELECT e.payload FROM latest l JOIN events e ON e.sequence=l.sequence WHERE l.key=?",
                (json.dumps(transition.key),)).fetchone()
            if prior is None:
                raise ValueError("Cannot finalize an unknown pending transition.")
            previous = self._load_event(prior[0])
            transition = replace(previous, rewards=transition.rewards, accepted=transition.accepted,
                                 reward_finalized=True, episode_success=(transition.episode_success
                                     if transition.episode_success is not None else previous.episode_success))
            finalize = True
        row = transition.cpu_copy()
        key = json.dumps(row.key)
        digest = content_digest(row)
        prior = self.connection.execute(
            "SELECT e.sequence,e.digest,e.status,e.reason,e.payload FROM latest l "
            "JOIN events e ON e.sequence=l.sequence WHERE l.key=?",
            (key,),
        ).fetchone()
        if prior:
            sequence, previous_digest, status, reason, payload = prior
            if digest == previous_digest:
                return DurableAck(row.key, status, reason, sequence, digest)
            if not finalize or status != "pending":
                raise ValueError("Conflicting transition payload for an existing identity.")
            previous = self._load_event(payload)
            unchanged = replace(
                row,
                rewards=previous.rewards,
                reward_finalized=previous.reward_finalized,
                accepted=previous.accepted,
                episode_success=previous.episode_success,
            )
            if (
                content_digest(unchanged) != previous_digest
                or not row.reward_finalized
                or row.accepted is None
            ):
                raise ValueError("Finalization may only resolve reward/outcome labels and the record gate.")
        elif finalize:
            raise ValueError("Cannot finalize an unknown pending transition.")
        status, reason = row.admission(
            self.spec, self.chunk_size, allow_terminal_chunks=self.allow_terminal_chunks,
            training_scope=self.training_scope,
        )
        self._check_episode_outcome(row, status, reason)
        payload = _dump(row)
        if len(payload) > min(self.max_record_bytes, self.demo_max_bytes if self.sampling == "demo_joint_50" and self.is_demo(row) else self.max_bytes):
            raise ValueError("Transition exceeds Replay byte bounds.")
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO events(key,digest,status,reason,payload) VALUES (?,?,?,?,?)",
                (key, digest, status, reason, payload),
            )
            sequence = cursor.lastrowid
            self.connection.execute("INSERT OR REPLACE INTO latest VALUES (?,?)", (key, sequence))
        # ACK is returned only after the durable transaction commits.
        self._apply(sequence, status, reason, row, len(payload))
        return DurableAck(row.key, status, reason, sequence, digest)

    @staticmethod
    def is_demo(row):
        return row.origin == "demo" or bool(row.takeover.any()) or row.source == "human"

    def sample(self, batch_size):
        return [row.cpu_copy() for row in self._sample_views(batch_size)]

    def _sample_views(self, batch_size):
        """Borrow rows for read-only collation by the single Replay owner."""
        if not self.rows or batch_size <= 0:
            raise ValueError("Cannot sample empty Replay or nonpositive batch size.")

        def draw(pool, count):
            indices = torch.randint(len(pool), (count,), generator=self.generator).tolist()
            return [pool[i] for i in indices]

        if self.sampling == "demo_joint_50":
            half = batch_size // 2
            selected = draw(self._demo_ring or self._online_ring, half)
            selected.extend(draw(self._online_ring or self._demo_ring, batch_size - half))
        else:
            selected = draw(self._ring, batch_size)
        self.last_sample_demo_fraction = sum(self.is_demo(row) for row in selected) / batch_size
        return [
            replace(row, episode_success=self.episode_outcomes[row.observation_id.session])
            if row.episode_success is None and row.observation_id.session in self.episode_outcomes
            else row
            for row in selected
        ]

    def state_dict(self):
        return copy.deepcopy(self._checkpoint_state_dict())

    def _checkpoint_state_dict(self):
        """Borrow row payloads only for synchronous serialization at an update barrier."""
        return {
            "identity": self.identity,
            "rows": OrderedDict(self.rows),
            "episode_outcomes": dict(self.episode_outcomes),
            "episode_outcomes_version": 2,
            "bytes": self.bytes,
            "budget": self.budget,
            "watermark": self.watermark,
            "watermark_digest": self._watermark_digest(self.watermark),
            "admitted": self.admitted,
            "online_admitted": self.online_admitted,
            "status_counts": dict(self.status_counts),
            "rng": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state):
        self._load_checkpoint_state_dict(copy.deepcopy(state))

    def _load_checkpoint_state_dict(self, state):
        """Take ownership of row payloads from a freshly deserialized private snapshot."""
        if state["identity"] != self.identity:
            raise ValueError("Replay checkpoint identity mismatch.")
        latest = self.connection.execute("SELECT COALESCE(MAX(sequence),0) FROM events").fetchone()[
            0
        ]
        if latest < state["watermark"]:
            raise ValueError("Replay journal is missing checkpoint records.")
        if self._watermark_digest(state["watermark"]) != state["watermark_digest"]:
            raise ValueError("Replay journal checkpoint watermark does not match.")
        self.rows = OrderedDict(state["rows"])
        self.status_counts = dict(state["status_counts"])
        for name in ("bytes", "budget", "watermark", "admitted"):
            setattr(self, name, state[name])
        self.online_admitted = state.get("online_admitted", state["admitted"])
        if state.get("episode_outcomes_version") == 2 and "episode_outcomes" in state:
            self.episode_outcomes = dict(state["episode_outcomes"])
        else:
            # Older snapshots omitted outcomes from outside the training scope.
            self.episode_outcomes.clear()
            for status, reason, payload in self.connection.execute(
                "SELECT status,reason,payload FROM events WHERE sequence<=? AND "
                "(status='admitted' OR (status='rejected' AND reason='outside_training_scope')) "
                "ORDER BY sequence",
                (self.watermark,),
            ):
                self._index_episode_outcome(self._load_event(payload), status, reason)
        self.generator.set_state(state["rng"])
        self._rebuild_rings()
        self._replay_after(self.watermark)

    def _watermark_digest(self, sequence):
        if sequence == 0:
            return None
        row = self.connection.execute(
            "SELECT digest FROM events WHERE sequence=?", (sequence,)
        ).fetchone()
        if row is None:
            raise ValueError("Replay journal is missing its watermark record.")
        return row[0]

    def close(self):
        self.connection.close()
