"""Single-owner CPU Replay with SQLite durable journal and explicit UTD budget."""

from __future__ import annotations

import copy
import io
import json
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .identity import content_digest


def _dump(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _load(payload):
    # Journal is a local trusted training asset, never an untrusted network pickle.
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class DurableAck:
    key: tuple[str, ...]
    status: str
    reason: str
    sequence: int
    digest: str


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
    ):
        if training_scope not in {"all", "critical_phase"}:
            raise ValueError("Replay training_scope must be all or critical_phase.")
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
        self.bytes = self.budget = self.watermark = self.admitted = self.online_admitted = 0
        self.status_counts = {"pending": 0, "admitted": 0, "rejected": 0}
        self.generator = torch.Generator().manual_seed(seed)
        self._replay_after(0)

    def _apply(self, sequence, status, payload):
        self.watermark = sequence
        self.status_counts[status] += 1
        if status == "admitted":
            row = _load(payload)
            self.rows[row.key] = (row, len(payload))
            self.bytes += len(payload)
            self.admitted += 1
            if row.origin not in {"demo", "vla"}:
                self.budget += self.utd
                self.online_admitted += 1
            if self.sampling == "demo_joint_50":
                for demo, capacity, byte_limit in ((True, self.demo_capacity, self.demo_max_bytes),
                                                   (False, self.capacity, self.max_bytes)):
                    keys = [k for k, (r, _) in self.rows.items() if self.is_demo(r) == demo]
                    pool_bytes = sum(self.rows[k][1] for k in keys)
                    while len(keys) > capacity or pool_bytes > byte_limit:
                        key = keys.pop(0)
                        _, size = self.rows.pop(key)
                        self.bytes -= size
                        pool_bytes -= size
                return
            while len(self.rows) > self.capacity or self.bytes > self.max_bytes:
                _, (_, size) = self.rows.popitem(last=False)
                self.bytes -= size

    def _replay_after(self, watermark):
        for sequence, status, payload, digest in self.connection.execute(
            "SELECT sequence,status,payload,digest FROM events WHERE sequence>? ORDER BY sequence",
            (watermark,),
        ):
            if content_digest(_load(payload)) != digest:
                raise ValueError(f"Corrupt Replay journal event at sequence {sequence}.")
            self._apply(sequence, status, payload)

    def pending(self):
        return [_load(row[0]) for row in self.connection.execute(
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
            previous = _load(prior[0])
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
            previous = _load(payload)
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
        self._apply(sequence, status, payload)
        return DurableAck(row.key, status, reason, sequence, digest)

    @staticmethod
    def is_demo(row):
        return row.origin == "demo" or bool(row.takeover.any()) or row.source == "human"

    def sample(self, batch_size):
        if not self.rows or batch_size <= 0:
            raise ValueError("Cannot sample empty Replay or nonpositive batch size.")
        rows = list(self.rows.values())
        if self.sampling == "demo_joint_50":
            demo = [r for r in rows if self.is_demo(r[0])]
            joint = [r for r in rows if not self.is_demo(r[0])]
            pools = [(demo or joint, batch_size // 2), (joint or demo, batch_size - batch_size // 2)]
            selected = [pool[i][0].cpu_copy() for pool, count in pools for i in
                        torch.randint(len(pool), (count,), generator=self.generator).tolist()]
            self.last_sample_demo_fraction = sum(self.is_demo(r) for r in selected) / batch_size
            return selected
        indices = torch.randint(len(rows), (batch_size,), generator=self.generator).tolist()
        return [rows[index][0].cpu_copy() for index in indices]

    def state_dict(self):
        return copy.deepcopy(
            {
                "identity": self.identity,
                "rows": self.rows,
                "bytes": self.bytes,
                "budget": self.budget,
                "watermark": self.watermark,
                "watermark_digest": self._watermark_digest(self.watermark),
                "admitted": self.admitted,
                "online_admitted": self.online_admitted,
                "status_counts": self.status_counts,
                "rng": self.generator.get_state(),
            }
        )

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Replay checkpoint identity mismatch.")
        latest = self.connection.execute("SELECT COALESCE(MAX(sequence),0) FROM events").fetchone()[
            0
        ]
        if latest < state["watermark"]:
            raise ValueError("Replay journal is missing checkpoint records.")
        if self._watermark_digest(state["watermark"]) != state["watermark_digest"]:
            raise ValueError("Replay journal checkpoint watermark does not match.")
        for name in ("rows", "bytes", "budget", "watermark", "admitted", "status_counts"):
            setattr(self, name, copy.deepcopy(state[name]))
        self.online_admitted = state.get("online_admitted", state["admitted"])
        self.generator.set_state(state["rng"])
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
