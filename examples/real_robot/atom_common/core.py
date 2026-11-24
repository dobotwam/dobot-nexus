"""Hardware-free trajectory, freshness and stop contracts for ATOM inference."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

STATE_NAMES = tuple(
    side + "_" + joint
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_pitch_joint",
        "elbow_roll_joint",
        "wrist_pitch_joint",
        "wrist_yaw_joint",
        "gripper_opening",
    )
) + ("head_yaw_joint", "head_pitch_joint")
ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


class RuntimeFault(RuntimeError):
    """An execution gate failed; stop advancing policy commands."""


def negotiate_horizon(server_horizon, requested=None):
    """Use the declared full chunk length; never silently crop a different H."""
    if type(server_horizon) is not int or not 1 <= server_horizon <= 256:
        raise RuntimeFault("Policy must declare a finite action horizon between 1 and 256")
    if requested is not None and (type(requested) is not int or requested != server_horizon):
        raise RuntimeFault(
            f"Requested horizon {requested} differs from policy horizon {server_horizon}"
        )
    return server_horizon


def finite_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise RuntimeFault(f"{name} must be finite real numbers with shape {shape}")
    return np.array(array, dtype=np.float64, copy=True)


def require_fresh(timestamp, now, maximum_age, name):
    """Read a shared timestamp BEFORE sampling now; concurrent producers may advance it."""
    if (
        not all(math.isfinite(x) for x in (timestamp, now, maximum_age))
        or maximum_age <= 0
        or not 0 <= now - timestamp <= maximum_age
    ):
        raise RuntimeFault(
            f"{name} freshness failed: age={now - timestamp:.6f}s, maximum={maximum_age:.6f}s"
        )


def state_vector(upper_q, openings):
    q = finite_array(upper_q, (17,), "upper feedback")
    hands = finite_array(openings, (2,), "hand feedback")
    if np.any(hands < 0) or np.any(hands > 1):
        raise RuntimeFault("Invalid hand opening feedback")
    return np.r_[q[1:8], hands[0], q[8:15], hands[1], q[15:17]].astype(np.float32)


@dataclass(frozen=True)
class ActionChunk:
    sequence: int
    observation_time: float
    received_time: float
    actions: np.ndarray

    @classmethod
    def create(cls, sequence, observation_time, received_time, actions, *, horizon=20):
        if type(sequence) is not int or sequence < 1:
            raise RuntimeFault("Invalid action sequence")
        if (
            not all(math.isfinite(t) for t in (observation_time, received_time))
            or received_time < observation_time
        ):
            raise RuntimeFault("Invalid action timestamps")
        data = finite_array(actions, (horizon, 18), "policy action chunk")
        data.flags.writeable = False
        return cls(sequence, observation_time, received_time, data)


def actuator_targets(actions, upper_reference, joint_limits, *, head_mode="policy", saturate_joints=False):
    """Map the 18D layout; hold the waist and optionally the measured head pose."""
    if head_mode not in ("hold", "policy"):
        raise RuntimeFault("Head mode must be hold or policy")
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 18:
        raise RuntimeFault("Invalid action layout")
    actions = finite_array(actions, actions.shape, "actions")
    reference = finite_array(upper_reference, (17,), "held reference")
    limits = finite_array(joint_limits, (17, 2), "joint limits")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise RuntimeFault("Invalid joint interval")
    points = np.tile(np.r_[reference, 0.0, 0.0], (len(actions), 1))
    points[:, 1:8] = actions[:, :7]
    points[:, 8:15] = actions[:, 8:15]
    if head_mode == "policy":
        points[:, 15:17] = actions[:, 16:18]
    if saturate_joints:
        controlled = slice(1, 17 if head_mode == "policy" else 15)
        points[:, controlled] = np.clip(points[:, controlled], limits[controlled, 0], limits[controlled, 1])
    if np.any(points[:, :17] < limits[:, 0]) or np.any(points[:, :17] > limits[:, 1]):
        raise RuntimeFault("Policy waypoint exceeds reviewed joint limits")
    hands = actions[:, [7, 15]]
    # DH's physical domain is [0, 1]. The hands use the same domain on both platforms.
    points[:, 17:] = np.clip(hands, 0, 1)
    return points


class FixedRateChunk:
    """Publish each native motor reference for one dataset period, in order.

    This is zero-order hold, matching SDK command_joint_state/send_cmd. It
    neither modifies raw joint targets nor retimes them based on displacement.
    A 50-row chunk at 20 Hz always occupies 2.5 seconds.
    """

    def __init__(self, points, *, fps=20.0):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 19 or not len(points):
            raise RuntimeFault("Empty or invalid actuator path")
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeFault("Invalid dataset action frequency")
        self.points = finite_array(points, points.shape, "actuator path")
        self.fps = float(fps)
        self.duration = len(points) / self.fps

    def frame_index(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeFault("Invalid action clock")
        return min(int(math.floor(elapsed * self.fps + 1e-9)), len(self.points) - 1)

    def sample(self, elapsed):
        row = self.frame_index(elapsed)
        completed = min(int(math.floor(elapsed * self.fps + 1e-9)), len(self.points))
        return self.points[row].copy(), completed


class Trajectory:
    """Consume all H waypoints with bounded reference velocity, no target clipping.

    Time adaptation is explicit: nominal dataset cadence is 20 Hz, but reference
    velocity stays within the independently reviewed actuator safety envelope.
    """

    def __init__(self, start, points, *, fps=20.0, joint_speed=0.1, hand_speed=0.5):
        self.start = finite_array(start, (19,), "initial actuator reference")
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 19 or not len(points):
            raise RuntimeFault("Empty or invalid actuator path")
        self.points = finite_array(points, points.shape, "actuator path")
        if not all(math.isfinite(v) and v > 0 for v in (fps, joint_speed, hand_speed)):
            raise RuntimeFault("Invalid path timing")
        self.joint_speed = joint_speed
        deltas = np.abs(np.diff(np.vstack([self.start, self.points]), axis=0))
        self.durations = np.maximum.reduce(
            [
                np.full(len(points), 1 / fps),
                1.875 * np.max(deltas[:, :17], axis=1) / joint_speed,
                1.875 * np.max(deltas[:, 17:], axis=1) / hand_speed,
            ]
        )
        self.ends = np.cumsum(self.durations)
        self.duration = float(self.ends[-1])

    def sample(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeFault("Invalid trajectory clock")
        completed = int(np.searchsorted(self.ends, elapsed, side="right"))
        if completed >= len(self.points):
            return self.points[-1].copy(), len(self.points)
        begin = 0.0 if completed == 0 else self.ends[completed - 1]
        start = self.start if completed == 0 else self.points[completed - 1]
        u = float(np.clip((elapsed - begin) / self.durations[completed], 0, 1))
        blend = u**3 * (10 + u * (-15 + 6 * u))
        return start + (self.points[completed] - start) * blend, completed


class PlaybackClock:
    """Delay playback after scheduler jitter instead of jumping the reference.

    The path's peak speed is an upper bound on every joint derivative. Limiting
    each virtual-time advance therefore preserves both the full path and the
    per-command step envelope, even after a late Linux scheduling wakeup.
    The separate live servo watchdog still rejects gaps over 50 ms.
    """

    def __init__(self, now, joint_speed, reference_step_deg=0.04):
        if (
            not all(math.isfinite(v) for v in (now, joint_speed, reference_step_deg))
            or not 0 < joint_speed <= 0.1
            or not 0 < reference_step_deg <= 0.04
        ):
            raise RuntimeFault("Invalid bounded playback clock")
        self.previous = now
        self.elapsed = 0.0
        self.delayed = 0.0
        self.maximum_advance = math.radians(reference_step_deg) / joint_speed

    def advance(self, now, scale=1.0):
        if (
            not math.isfinite(now)
            or now < self.previous
            or not math.isfinite(scale)
            or not 0 <= scale <= 1
        ):
            raise RuntimeFault("Non-monotonic playback clock")
        actual = now - self.previous
        accepted = min(actual * scale, self.maximum_advance)
        self.previous = now
        self.elapsed += accepted
        self.delayed += actual - accepted
        return self.elapsed


class ChunkGate:
    def __init__(self, maximum_observation_age=0.7):
        self.last_sequence = 0
        self.maximum_age = maximum_observation_age

    def accept(self, chunk, now):
        if chunk.sequence <= self.last_sequence:
            raise RuntimeFault("Duplicate or out-of-order action chunk")
        require_fresh(chunk.observation_time, now, self.maximum_age, "action observation")
        self.last_sequence = chunk.sequence


class LoopGuard:
    """Stop repeated action patterns only when measured arm state makes no progress."""

    def __init__(
        self,
        repetitions=3,
        action_tolerance_deg=1.0,
        state_tolerance_deg=0.5,
        hand_tolerance=0.03,
        max_period=3,
    ):
        if type(repetitions) is not int or repetitions < 2:
            raise ValueError("At least two repetitions are required")
        self.repetitions = repetitions
        self.action_tol = math.radians(action_tolerance_deg)
        self.state_tol = math.radians(state_tolerance_deg)
        self.hand_tol = hand_tolerance
        if type(max_period) is not int or max_period < 1:
            raise ValueError("Invalid maximum loop period")
        self.max_period = max_period
        self.history = deque(maxlen=repetitions * max_period)

    def observe(self, actions, measured_state):
        actions = np.asarray(actions)
        actions = finite_array(actions, actions.shape, "loop actions")
        if actions.ndim != 2 or actions.shape[1] != 18 or not len(actions):
            raise RuntimeFault("Invalid loop action layout")
        state = finite_array(measured_state, (18,), "loop state")
        self.history.append((actions, state))
        if len(self.history) < self.repetitions:
            return False
        arms = np.array(ARM_COLUMNS)

        def matches(reference, other):
            actions, state = reference
            other_actions, other_state = other
            return (
                actions.shape == other_actions.shape
                and np.max(
                    np.sqrt(np.mean((actions[:, arms] - other_actions[:, arms]) ** 2, axis=0))
                )
                <= self.action_tol
                and np.max(np.abs(actions[:, [7, 15]] - other_actions[:, [7, 15]])) <= self.hand_tol
                and np.max(np.abs(state[arms] - other_state[arms])) <= self.state_tol
                and np.max(np.abs(state[[7, 15]] - other_state[[7, 15]])) <= self.hand_tol
            )

        for period in range(1, self.max_period + 1):
            if len(self.history) < period * self.repetitions:
                continue
            window = list(self.history)[-period * self.repetitions :]
            if all(
                matches(window[phase], window[cycle * period + phase])
                for cycle in range(1, self.repetitions)
                for phase in range(period)
            ):
                return True
        return False


class BlockedDirectionGuard:
    """Two-horizon obstruction gate using TCP feedback and an optional table plane.

    First confirmed blockage pauses playback. The NEXT fresh horizon is checked
    before any of its commands are sent. This class does not claim a calibrated
    table plane exists; the runtime must validate calibration provenance.
    """

    def __init__(self, direction_cosine=0.8, minimum_push_m=0.003, dimension=3):
        self.direction_cosine = direction_cosine
        self.minimum_push = minimum_push_m
        self.dimension = dimension
        self.blocked_direction = None
        self.blocked_position = None
        self.source_sequence = None

    def record_blockage(self, sequence, actual_tcp, requested_tcp):
        actual = finite_array(actual_tcp, (self.dimension,), "obstruction feedback")
        requested = finite_array(requested_tcp, (self.dimension,), "obstruction reference")
        delta = requested - actual
        distance = float(np.linalg.norm(delta))
        if distance < self.minimum_push:
            raise RuntimeFault("Insufficient blocked-direction evidence")
        self.blocked_direction = delta / distance
        self.blocked_position = actual
        self.source_sequence = sequence

    def next_horizon_pushes_into_obstruction(self, sequence, tcp_targets):
        if self.blocked_direction is None:
            return False
        if sequence <= self.source_sequence:
            raise RuntimeFault("Blockage confirmation needs the next fresh horizon")
        targets = np.asarray(tcp_targets)
        if targets.ndim != 2 or targets.shape[1] != self.dimension:
            raise RuntimeFault("Invalid obstruction target sequence")
        targets = finite_array(targets, targets.shape, "TCP path")
        delta = targets - self.blocked_position
        distance = np.linalg.norm(delta, axis=1)
        projection = delta @ self.blocked_direction
        return bool(
            np.any(
                (projection >= self.minimum_push) & (projection >= distance * self.direction_cosine)
            )
        )
