"""URDF wrist forward kinematics and measured obstruction evidence."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

from .core import RuntimeFault, finite_array


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm < 1e-12:
        raise RuntimeFault("Invalid URDF rotation axis")
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ]
    )


class Kinematics:
    def __init__(self, urdf, joint_names, base="torso_link", tool_offsets=None):
        root = ET.parse(urdf).getroot()
        self.index = {
            n.replace("wrist_roll_joint", "wrist_yaw_joint"): i for i, n in enumerate(joint_names)
        }
        parents = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.chains = {}
        self.offsets = tool_offsets or {"left": [0, 0, 0], "right": [0, 0, 0]}
        for side in ("left", "right"):
            link = side + "_wrist_yaw_link"
            chain = []
            visited = set()
            while link != base:
                if link in visited or link not in parents:
                    raise RuntimeFault("URDF does not connect wrist to " + base)
                visited.add(link)
                joint = parents[link]
                chain.append(joint)
                link = joint.find("parent").get("link")
            self.chains[side] = list(reversed(chain))

    def position(self, q, side):
        q = finite_array(q, (17,), "FK upper state")
        transform = np.eye(4)
        for joint in self.chains[side]:
            origin = joint.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin is not None:
                xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            local = np.eye(4)
            local[:3, 3] = xyz
            local[:3, :3] = (
                rotation([0, 0, 1], rpy[2])
                @ rotation([0, 1, 0], rpy[1])
                @ rotation([1, 0, 0], rpy[0])
            )
            transform = transform @ local
            kind = joint.get("type")
            if kind != "fixed":
                name = joint.get("name")
                if name not in self.index or kind not in ("revolute", "continuous"):
                    raise RuntimeFault("Unsupported or unmapped wrist URDF joint: " + str(name))
                axis = joint.find("axis")
                axis = [
                    float(v) for v in (axis.get("xyz") if axis is not None else "1 0 0").split()
                ]
                motion = np.eye(4)
                motion[:3, :3] = rotation(axis, float(q[self.index[name]]))
                transform = transform @ motion
        offset = finite_array(self.offsets[side], (3,), "tool offset")
        return (transform @ np.r_[offset, 1.0])[:3]


class BlockageMonitor:
    """Confirm commanded movement + absent feedback progress + load/tracking evidence.

    This detects an obstruction, not its material or identity. A calibrated table
    plane is needed to label or preempt a particular desktop surface.
    """

    def __init__(
        self, window_s=0.4, minimum_reference_movement=0.004, maximum_feedback_progress=0.001
    ):
        self.window = window_s
        self.min_reference = minimum_reference_movement
        self.max_progress = maximum_feedback_progress
        self.history = {side: deque() for side in ("left", "right")}

    def observe(self, now, side, actual, reference, tracking_error_deg, load_fraction):
        actual = finite_array(actual, (3,), "TCP feedback")
        reference = finite_array(reference, (3,), "TCP reference")
        if not all(math.isfinite(v) for v in (now, tracking_error_deg, load_fraction)):
            raise RuntimeFault("Non-finite obstruction evidence")
        history = self.history[side]
        history.append((now, actual, reference, tracking_error_deg, load_fraction))
        while len(history) > 2 and history[1][0] < now - self.window:
            history.popleft()
        if len(history) < 2 or now - history[0][0] < self.window * 0.9:
            return None
        first = history[0]
        command = reference - first[2]
        distance = float(np.linalg.norm(command))
        if distance < self.min_reference:
            return None
        direction = command / distance
        progress = float((actual - first[1]) @ direction)
        error = float(np.linalg.norm(reference - actual))
        effort_change = load_fraction - first[4]
        if (
            progress <= self.max_progress
            and error >= self.min_reference
            and (tracking_error_deg >= 0.7 or effort_change >= 0.05)
        ):
            return {
                "side": side,
                "actual_tcp": actual.tolist(),
                "requested_tcp": reference.tolist(),
                "reference_displacement_m": distance,
                "feedback_progress_m": progress,
                "tracking_error_deg": tracking_error_deg,
                "effort_fraction_change": effort_change,
                "window_seconds": now - first[0],
            }
        return None


class JointBlockageMonitor:
    """Offset-independent stalled-joint evidence when absolute table calibration is absent."""

    def __init__(self, window_s=0.4):
        self.window = window_s
        self.history = {s: deque() for s in ("left", "right")}

    def observe(self, now, side, actual, reference, load_fraction):
        actual = finite_array(actual, (7,), "arm feedback")
        reference = finite_array(reference, (7,), "arm reference")
        history = self.history[side]
        history.append((now, actual, reference, load_fraction))
        while len(history) > 2 and history[1][0] < now - self.window:
            history.popleft()
        if len(history) < 2 or now - history[0][0] < self.window * 0.9:
            return None
        first = history[0]
        command = reference - first[2]
        distance = float(np.linalg.norm(command))
        if distance < math.radians(0.5):
            return None
        progress = float((actual - first[1]) @ (command / distance))
        tracking = float(np.max(abs(reference - actual)))
        if (
            progress <= math.radians(0.1)
            and tracking >= math.radians(0.7)
            and (load_fraction - first[3] >= 0.05 or tracking >= math.radians(1.0))
        ):
            return {
                "side": side,
                "feedback_q_rad": actual.tolist(),
                "reference_q_rad": reference.tolist(),
                "reference_movement_deg": math.degrees(distance),
                "feedback_progress_deg": math.degrees(progress),
                "tracking_error_deg": math.degrees(tracking),
                "effort_fraction_change": load_fraction - first[3],
                "evidence_type": "stalled_joint_direction; surface identity is not inferred",
            }
        return None
