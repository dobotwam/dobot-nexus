"""Bounded delivery of time-resampled RGB history for the Memory bridge.

This queue owns unacknowledged time slots with their source capture timestamps.
RGB compression is lossless. Feature encoding, memory-token selection and long-
term history belong to the policy server. No hardware/model imports.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
import zlib

import numpy as np


HISTORY_PROTOCOL = "rgb_history_v1"
HISTORY_CODEC = "rgb_delta_zlib_v1"
MAX_HISTORY_CHUNK = 128
_FRAME_SHAPE = (224, 224, 3)
_FRAME_BYTES = 224 * 224 * 3
_LOGGER = logging.getLogger(__name__)


def encode_history_frames(frames):
    """Lossless temporal deltas; each chunk is independently decodable."""
    if not 1 <= len(frames) <= MAX_HISTORY_CHUNK:
        raise ValueError("Invalid history chunk length.")
    blobs, previous = [], None
    for frame in frames:
        if not isinstance(frame, np.ndarray) or frame.shape != _FRAME_SHAPE or frame.dtype != np.uint8:
            raise ValueError("History images must be uint8 RGB [224,224,3].")
        delta = frame if previous is None else np.subtract(frame, previous, dtype=np.uint8)
        blobs.append(zlib.compress(delta.tobytes(), level=1))
        previous = frame
    return blobs


def decode_history_frames(codec, blobs):
    if codec != HISTORY_CODEC or not 1 <= len(blobs) <= MAX_HISTORY_CHUNK:
        raise ValueError("Unsupported history codec or chunk length; update both Memory scripts.")
    frames, previous = [], None
    for blob in blobs:
        if not isinstance(blob, bytes):
            raise ValueError("Compressed history frames must be bytes.")
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(blob, _FRAME_BYTES + 1)
        except zlib.error as exc:
            raise ValueError("Corrupt compressed history frame.") from exc
        if len(raw) != _FRAME_BYTES or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Invalid decompressed history frame length or trailing data.")
        delta = np.frombuffer(raw, np.uint8).reshape(_FRAME_SHAPE)
        frame = delta.copy() if previous is None else np.add(delta, previous, dtype=np.uint8)
        frames.append(frame)
        previous = frame
    return frames


class SerializedEnvironment:
    """Serialize SDK access and detach observations from reusable camera buffers."""

    def __init__(self, environment):
        self._environment = environment
        self._lock = threading.RLock()

    def __getattr__(self, name):
        value = getattr(self._environment, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            with self._lock:
                result = value(*args, **kwargs)
                return copy.deepcopy(result) if name == "get_observation" else result

        return call


class ObservationHistory:
    """Resample SDK observations onto the training clock; retain slots until ACK.

    Timestamps are host monotonic read times, not hardware exposure timestamps.
    Camera freshness therefore still depends on the SDK's observation API.
    Each completed time slot takes the nearest of the two bracketing reads,
    as in the dataset conversion. Slow reads can be reused for multiple slots;
    their original capture timestamps remain explicit. max_gap_s bounds actual
    source gaps, not accumulated deviation from an unattainable SDK frame rate.
    """

    def __init__(self, read_observation, history_image, *, fps=30.0,
                 max_pending=512, max_gap_s=0.1, on_error=None):
        if not np.isfinite(fps) or fps <= 0 or max_pending < 1:
            raise ValueError("History FPS and capacity must be positive.")
        if not np.isfinite(max_gap_s) or max_gap_s < 1 / fps:
            raise ValueError("History maximum gap must be at least one frame period.")
        self.read_observation = read_observation
        self.history_image = history_image
        self.fps = float(fps)
        self.max_pending = int(max_pending)
        self.max_gap_s = float(max_gap_s)
        self.on_error = on_error
        self.episode_id = uuid.uuid4().hex
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._pending = []
        self._latest = None
        self._first_timestamp = None
        self._last_timestamp = None
        self._previous_source = None
        self._source_reads = 0
        self._repeated_slots = 0
        self._last_selected_source = None
        self._max_source_gap_s = 0.0
        self._next_frame = 0
        self._acked = -1
        self._offered = -1

    def start(self):
        if self._thread is not None:
            raise RuntimeError("A history collector can only be started once.")
        self._thread = threading.Thread(target=self._run, daemon=True, name="memory-observations")
        self._thread.start()

    def _run(self):
        deadline = time.monotonic()
        first = True
        try:
            while not self._stop.is_set():
                if self._stop.wait(max(0.0, deadline - time.monotonic())):
                    break
                observation = self.read_observation()
                timestamp = time.monotonic()
                if self._stop.is_set():
                    break
                self.append(observation, timestamp)
                # Initial imports/SDK startup occur before the episode timebase.
                deadline = time.monotonic() + 1 / self.fps if first else deadline + 1 / self.fps
                first = False
                # Do not burst camera reads to catch up. append resamples the
                # actual read times onto the training timeline instead.
                now = time.monotonic()
                if deadline < now:
                    deadline = now + 1 / self.fps
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            # The cancellation callback may close a socket blocked in recv.
            # Report the cause first, before it becomes a generic socket error.
            try:
                _LOGGER.error("Memory observation collection failed: %s", exc)
            finally:
                if self.on_error is not None:
                    self.on_error()

    def check(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"Memory observation collection failed: {self._error}") from self._error

    def append(self, observation, timestamp):
        """Append one detached snapshot; also used by deterministic replay tests."""
        image = np.array(self.history_image(observation), copy=True, order="C")
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError("History images must be uint8 RGB [224,224,3].")
        if not np.isfinite(timestamp):
            raise ValueError("Invalid history capture timestamp.")
        with self._condition:
            self.check()
            if self._last_timestamp is not None:
                gap = timestamp - self._last_timestamp
                if not 0 < gap <= self.max_gap_s + 1e-9:
                    raise ValueError(f"Invalid history capture interval: {gap:.6f}s.")
            else:
                gap = 0.0
            origin = timestamp if self._first_timestamp is None else self._first_timestamp
            # A tiny tolerance avoids losing a slot to floating point subtraction
            # of large monotonic timestamps. Never round a slow read's index down
            # to its source sequence number, which would compress elapsed time.
            last_slot = int(np.floor((timestamp - origin) * self.fps + 1e-7))
            new_slots = last_slot - self._next_frame + 1
            if len(self._pending) + new_slots > self.max_pending:
                oldest = self._pending[0][1] if self._pending else origin + self._next_frame / self.fps
                raise RuntimeError(
                    f"Unacknowledged history queue is full ({len(self._pending)}/{self.max_pending}, "
                    f"incoming_slots={new_slots}); "
                    f"last_ack={self._acked}, latest={self._next_frame - 1}, "
                    f"oldest_age_s={timestamp - oldest:.3f}. "
                    "History delivery is not keeping up; refusing to drop frames."
                )
            source = (float(timestamp), observation, image, self._source_reads)
            for index in range(self._next_frame, last_slot + 1):
                target = origin + index / self.fps
                chosen = source
                if (self._previous_source is not None
                        and abs(target - self._previous_source[0]) <= abs(timestamp - target)):
                    chosen = self._previous_source
                captured_at, selected_observation, selected_image, source_index = chosen
                self._pending.append((index, target, selected_image, captured_at))
                self._latest = (index, selected_observation)
                if source_index == self._last_selected_source:
                    self._repeated_slots += 1
                self._last_selected_source = source_index
            self._next_frame = last_slot + 1
            self._first_timestamp = float(origin)
            self._last_timestamp = float(timestamp)
            self._previous_source = source
            self._source_reads += 1
            self._max_source_gap_s = max(self._max_source_gap_s, gap)
            self._condition.notify_all()

    def snapshot(self, timeout=5.0):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._error is not None or self._stop.is_set()
                or (self._latest is not None and self._latest[0] > self._acked), timeout
            )
            self.check()
            if self._stop.is_set():
                raise RuntimeError("History collector is stopped.")
            if not ready:
                raise TimeoutError("No new camera observation for Memory inference.")
            index, observation = self._latest
            self._offered = index
            return observation, {
                "protocol": HISTORY_PROTOCOL,
                "episode_id": self.episode_id,
                "fps": self.fps,
                "frame_index": index,
                "step_indices": [x[0] for x in self._pending],
                "timestamps": [x[1] for x in self._pending],
                "capture_timestamps": [x[3] for x in self._pending],
                "sampling": "nearest_timestamp_v1",
                "frames": [x[2] for x in self._pending],
            }

    def acknowledge(self, episode_id, frame_index):
        with self._condition:
            self.check()
            if (episode_id != self.episode_id or isinstance(frame_index, (bool, np.bool_))
                    or not isinstance(frame_index, (int, np.integer)) or frame_index > self._offered):
                raise ValueError("History acknowledgement does not match the sent observation.")
            if frame_index <= self._acked:
                raise ValueError("History acknowledgement was already consumed.")
            self._pending = [item for item in self._pending if item[0] > frame_index]
            self._acked = frame_index

    def status(self):
        with self._condition:
            return {"pending": len(self._pending), "capacity": self.max_pending, "acked": self._acked}

    def capture_status(self):
        with self._condition:
            elapsed = (self._last_timestamp - self._first_timestamp) if self._source_reads else 0.0
            return {"reads": self._source_reads, "slots": self._next_frame,
                    "observed_fps": (self._source_reads - 1) / elapsed if elapsed > 0 else 0.0,
                    "repeated_slots": self._repeated_slots, "max_source_gap_s": self._max_source_gap_s}

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                raise RuntimeError("SDK observation reader did not stop; cannot reset or reuse hardware.")
