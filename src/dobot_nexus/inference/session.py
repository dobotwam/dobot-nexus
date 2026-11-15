from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions


@dataclass(frozen=True)
class SessionOptions:
    execute_steps: int
    seed: int | None = None
    num_steps: int | None = None


class PolicySession:
    """Stateful action-buffer wrapper for synchronous robot control."""

    def __init__(self, policy, options: SessionOptions) -> None:
        if options.execute_steps <= 0:
            raise ValueError("SessionOptions.execute_steps must be positive.")
        if options.execute_steps > policy.spec.action_horizon:
            raise ValueError(
                "execute_steps cannot exceed the policy action horizon: "
                f"{options.execute_steps} > {policy.spec.action_horizon}."
            )
        self.policy = policy
        self.options = options
        self._buffer: np.ndarray | None = None
        self._cursor = 0
        self._batch_size: int | None = None
        self._seed = options.seed
        self._identity = None
        self._lock = RLock()

    @property
    def empty(self) -> bool:
        return self._buffer is None or self._cursor >= self._buffer.shape[1]

    @property
    def remaining_steps(self) -> int:
        return 0 if self.empty else int(self._buffer.shape[1] - self._cursor)

    def reset(self, seed: int | None = None) -> None:
        with self._lock:
            self._reset(seed)

    def _reset(self, seed=None):
        self._identity = None
        self._buffer = None
        self._cursor = 0
        self._batch_size = None
        if seed is not None:
            self._seed = int(seed)

    def act(self, observation: ObservationBatch | dict) -> np.ndarray:
        with self._lock:
            return self._act(observation)

    def _act(self, observation):
        canonical = ObservationBatch.from_value(observation)
        identity = (tuple(canonical.env_ids or []), tuple(canonical.prompt))
        if self._identity is not None and identity != self._identity:
            raise ValueError("PolicySession env_ids or prompt changed; reset before changing tasks.")
        self._identity = identity
        if self._batch_size is not None and canonical.batch_size != self._batch_size:
            raise ValueError(
                "PolicySession batch size changed without reset: "
                f"{canonical.batch_size} != {self._batch_size}."
            )
        if self.empty:
            output = self.policy.predict_chunk(
                canonical,
                SamplingOptions(seed=self._seed, num_steps=self.options.num_steps),
            )
            self._buffer = output.actions[:, : self.options.execute_steps]
            self._cursor = 0
            self._batch_size = canonical.batch_size
        assert self._buffer is not None
        action = self._buffer[:, self._cursor]
        self._cursor += 1
        return np.ascontiguousarray(action, dtype=np.float32)

