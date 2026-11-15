"""Ordinary prediction composition using the existing InferencePolicy contract."""

from __future__ import annotations

import threading
import time

from dobot_nexus.core.contracts import SamplingOptions


class PredictionHandler:
    """Return canonical batched actions; callers own wire-specific transformations.

    Sharing this handler across connections preserves a single seed sequence.
    Stateful model sessions should supply their own handler instead.
    """

    def __init__(self, policy, *, seed=42, num_steps=10, vary_seed=True):
        self.policy = policy
        self.seed, self.num_steps, self.vary_seed = int(seed), int(num_steps), bool(vary_seed)
        self._request_index = 0
        self._lock = threading.Lock()

    def __call__(self, observation):
        with self._lock:
            index = self._request_index
            self._request_index += 1
        seed = self.seed + index if self.vary_seed else self.seed
        started = time.monotonic()
        output = self.policy.predict_chunk(
            observation, SamplingOptions(seed=seed, num_steps=self.num_steps)
        )
        timing = {"infer_ms": (time.monotonic() - started) * 1000}
        model_ms = output.timing.get("model_ms", output.timing.get("total_ms"))
        if model_ms is not None:
            timing["model_ms"] = float(model_ms)
        if output.timing.get("total_ms") is not None:
            timing["backend_total_ms"] = float(output.timing["total_ms"])
        return {"actions": output.actions, "policy_timing": timing, "seed": seed}
