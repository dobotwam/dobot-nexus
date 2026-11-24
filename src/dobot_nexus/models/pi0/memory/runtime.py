"""Per-environment state outside the shared policy/model."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sized
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from itertools import islice
from threading import RLock

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch
from dobot_nexus.inference.session import PolicySession, SessionOptions

from .sampling import assemble_memory, select_frames
from .types import MemoryObservationBatch, MemorySnapshot


def _step_index(step):
    if not isinstance(step, (int, np.integer)) or isinstance(step, (bool, np.bool_)) or step < 0:
        raise ValueError("History step must be a nonnegative integer.")
    return int(step)


class HistoryBuffer:
    def __init__(self, config):
        self.config = config
        self.capacity = config.window_frames or config.max_frames
        self.reset()

    @property
    def nbytes(self):
        return len(self.tokens) * 16 * 2048 * 4

    def reset(self):
        self.tokens = OrderedDict()
        self.first_step = None
        self.last_step = None

    def check_step(self, step):
        step = _step_index(step)
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError(
                "History must be dense and ordered; send missing frames or reset explicitly."
            )
        if (
            self.first_step is not None
            and self.config.window_frames is None
            and step - self.first_step >= self.capacity
        ):
            raise ValueError(
                "Global history capacity exceeded; "
                "reset or use a trained sliding-window configuration."
            )

    def append(self, tokens, step):
        step = _step_index(step)
        self.check_step(step)
        tokens = np.asarray(tokens)
        if (
            tokens.shape != (16, 2048)
            or tokens.dtype != np.float32
            or not np.isfinite(tokens).all()
        ):
            raise ValueError("Encoded frame must be finite float32 [16,2048].")
        if self.first_step is None:
            self.first_step = step
        self.tokens[step % self.capacity] = tokens.copy()
        self.tokens.move_to_end(step % self.capacity)
        if len(self.tokens) > self.capacity:
            self.tokens.popitem(last=False)
        self.last_step = step

    def snapshot(self):
        if self.last_step is None:
            raise ValueError("Append the current frame before requesting memory.")
        indices, positions = select_frames(
            self.last_step,
            self.config.sampled_frames,
            segment_start=self.first_step,
            window_frames=self.config.window_frames,
        )
        return assemble_memory(
            np.stack([self.tokens[int(i) % self.capacity] for i in indices]), positions, self.config
        )


class MemorySession:
    def __init__(
        self,
        policy,
        encoder,
        config,
        options: SessionOptions,
        *,
        max_environments=4,
        max_history_bytes=256 * 1024 * 1024,
        max_batch_frames=128,
    ):
        if any(
            type(n) is not int or n < 1
            for n in (max_environments, max_history_bytes, max_batch_frames)
        ):
            raise ValueError("Memory runtime limits must be positive integers.")
        self.max_environments = max_environments
        self.max_history_bytes = max_history_bytes
        self.max_batch_frames = max_batch_frames
        self.policy, self.encoder, self.config, self.options = policy, encoder, config, options
        self.buffers, self.actions, self.prompts = {}, {}, {}
        self._seeds = {}
        self._lock = RLock()

    @contextmanager
    def _transaction(self, env_ids):
        """Publish request-local histories and action cursors together on success."""
        with self._lock:
            self._check_environments(env_ids)
            staged = copy(self)
            staged.buffers = self.buffers.copy()
            staged.actions = self.actions.copy()
            staged.prompts = self.prompts.copy()
            staged._seeds = self._seeds.copy()
            for key in env_ids:
                if key in self.buffers:
                    # Appends replace whole frame arrays. Cached action chunks are
                    # also read-only during act(), so only their cursor needs staging.
                    buffer = copy(self.buffers[key])
                    buffer.tokens = buffer.tokens.copy()
                    staged.buffers[key] = buffer
                    staged.actions[key] = copy(self.actions[key])
            yield staged
            self.buffers, self.actions = staged.buffers, staged.actions
            self.prompts, self._seeds = staged.prompts, staged._seeds

    def reset(self, env_id=None, seed=None):
        with self._lock:
            seed = int(seed) if seed is not None else None
            if env_id is None:
                self._seeds.clear()
                if seed is not None:
                    self.options = replace(self.options, seed=seed)
            elif seed is not None:
                self._check_environments([env_id])
                self._seeds[env_id] = seed
            else:
                self._seeds.pop(env_id, None)
            keys = list(self.buffers) if env_id is None else [env_id]
            for key in keys:
                self.buffers.pop(key, None)
                self.prompts.pop(key, None)
                session = self.actions.pop(key, None)
                if session is not None:
                    session.reset(seed)

    def _check_environments(self, env_ids):
        if any(not isinstance(key, str) or not key.strip() or len(key) > 256 for key in env_ids):
            raise ValueError("env_id must be a nonempty string of at most 256 characters.")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("Batched memory requires unique explicit env_ids.")
        if len(set(self.buffers).union(self._seeds, env_ids)) > self.max_environments:
            raise ValueError("Memory environment limit reached; reset existing environments.")

    def _prepare_environment(self, env_id, prompt):
        if env_id in self.prompts and self.prompts[env_id] != prompt:
            self.reset(env_id)
        if env_id not in self.buffers:
            self.buffers[env_id] = HistoryBuffer(self.config)
            options = replace(self.options, seed=self._seeds.get(env_id, self.options.seed))
            self.actions[env_id] = PolicySession(self.policy, options)
        self.prompts[env_id] = prompt
        return self.buffers[env_id]

    def push_frames(self, frames, steps, *, env_id="default", prompt):
        """Ingest frames executed between action-chunk requests."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("History prompt must be a nonempty string.")
        limit = self.max_batch_frames
        if any(isinstance(items, Sized) and len(items) > limit for items in (frames, steps)):
            raise ValueError("History batch exceeds frame limit.")
        frames = list(islice(frames, limit + 1))
        if len(frames) > limit:
            raise ValueError("History batch exceeds frame limit.")
        steps = list(islice(steps, limit + 1))
        if len(steps) > limit:
            raise ValueError("History batch exceeds frame limit.")
        if not frames or len(frames) != len(steps):
            raise ValueError("History frames and steps must be nonempty and aligned.")
        with self._transaction([env_id]) as staged:
            buffer = staged._prepare_environment(env_id, prompt)
            staged._push_frames(buffer, frames, [_step_index(s) for s in steps])

    def _push_frames(self, buffer, frames, steps):
        buffer.check_step(steps[0])
        if any(s != steps[0] + i for i, s in enumerate(steps)):
            raise ValueError("Every history batch must have consecutive step indices.")
        first = buffer.first_step if buffer.first_step is not None else steps[0]
        if self.config.window_frames is None and steps[-1] - first >= buffer.capacity:
            raise ValueError("Global history capacity exceeded.")
        retained = min(buffer.capacity, len(buffer.tokens) + len(frames))
        projected = (
            sum(b.nbytes for b in self.buffers.values())
            + (retained - len(buffer.tokens)) * 16 * 2048 * 4
        )
        if projected > self.max_history_bytes:
            raise ValueError("Memory history byte limit reached; reset or raise the runtime limit.")
        encoded = np.asarray(self.encoder(frames), dtype=np.float32)
        if encoded.shape != (len(frames), 16, 2048) or not np.isfinite(encoded).all():
            raise ValueError("History encoder returned invalid features.")
        for tokens, step in zip(encoded, steps, strict=True):
            buffer.append(tokens, step)

    @contextmanager
    def _prepared(self, observation, step_indices):
        observation = ObservationBatch.from_value(observation)
        ids = observation.env_ids or (["default"] if observation.batch_size == 1 else None)
        if ids is None:
            raise ValueError("Batched memory requires unique explicit env_ids.")
        if step_indices is not None and len(step_indices) != len(ids):
            raise ValueError("One step index is required per environment.")
        with self._transaction(ids) as staged:
            yield staged, staged._prepare(observation, ids, step_indices)

    def prepare(self, observation, *, step_indices=None):
        with self._prepared(observation, step_indices) as (_, prepared):
            return prepared

    def _prepare(self, observation, ids, step_indices):
        histories = []
        for i, key in enumerate(ids):
            buffer = self._prepare_environment(key, observation.prompt[i])
            step = (
                (0 if buffer.last_step is None else buffer.last_step + 1)
                if step_indices is None
                else step_indices[i]
            )
            step = _step_index(step)
            # Explicitly reusing the last ingested step supports dense add_buffer clients.
            if step != buffer.last_step:
                self._push_frames(
                    buffer,
                    [observation.images[self.config.camera][i]],
                    [step],
                )
            histories.append(buffer.snapshot())
        return MemoryObservationBatch(
            images=observation.images,
            state=observation.state,
            prompt=observation.prompt,
            env_ids=ids,
            timestamps=observation.timestamps,
            memory=MemorySnapshot.stack(histories),
        )

    def predict_chunk(self, observation, *, step_indices=None, options=None):
        with self._prepared(observation, step_indices) as (_, prepared):
            return self.policy.predict_chunk(prepared, options)

    def act(self, observation, *, step_indices=None):
        with self._prepared(observation, step_indices) as (staged, prepared):
            results = []
            for i, key in enumerate(prepared.env_ids):
                single = MemoryObservationBatch(
                    images={k: v[i : i + 1] for k, v in prepared.images.items()},
                    state=prepared.state[i : i + 1],
                    prompt=[prepared.prompt[i]],
                    env_ids=[key],
                    memory=MemorySnapshot(
                        prepared.memory.features[i : i + 1],
                        prepared.memory.positions[i : i + 1],
                        prepared.memory.mask[i : i + 1],
                    ),
                )
                results.append(staged.actions[key].act(single))
            return np.concatenate(results)
