"""Run a short check or a full closed-loop Pi0.5 episode on ATOM D. Defaults to observation only."""

import argparse
import datetime
import json
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from dobot_nexus.inference.transport import msgpack_numpy
from examples.real_robot.atom_common.controller_limits import (
    controller_bounds,
    read_controller_settings,
)
from examples.real_robot.atom_common.core import (
    STATE_NAMES,
    ActionChunk,
    ChunkGate,
    LoopGuard,
    actuator_targets,
    state_vector,
)
from examples.real_robot.atom_common.lease import atom_d_control_lease
from examples.real_robot.atom_common.sensors import CameraRig, CameraSpec, RealSenseCamera
from examples.real_robot.atom_d import home
from examples.real_robot.atom_d.console import CANCELLED, OperatorCancelled, operator_session
from examples.real_robot.atom_d.policy import policy_session

ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIG = Path(
    os.environ.get(
        "ATOM_D_CAMERA_CONFIG",
        Path.home()
        / "ATOM/Embodylink_v1.1.2_stable/Embodylink/usr/bin"
        / "robotcollectdata/10010/taskconfig/robotconfig.json",
    )
).expanduser()
ARTIFACT = os.environ.get("PI05_ARTIFACT", str(Path.home() / "step_00004000"))
PERIOD = 0.05
VALIDATION_FRAMES = 10
GRIPPER_FORCE_PERCENT = 100
JOINT_COLUMNS = np.r_[0:7, 8:15, 16:18]
GRASP_APPROACH_ROWS = 15
CARRY_REPLAN_ROWS = 16
DH_GRIP_OBJECT_DROPPED = 3
PREFETCH_ROWS = 4


class GraspTracker:
    """Track closing requests across policy horizons, including staggered two-hand grasps."""

    def __init__(self):
        self.since = [None, None]
        self.grasp_checked = [False, False]
        self.open_intent = [None, None]

    def hand_targets(self, raw, measured):
        """Resolve the near-binary DH targets, preserving intermediate starting poses."""
        out = raw.copy()
        releasing = []
        for i, value in enumerate(raw):
            if value >= 0.8:
                if self.open_intent[i] is False or (
                    self.open_intent[i] is None and measured[i] < 0.99
                ):
                    releasing.append(i)
                self.open_intent[i] = True
            elif value <= 0.2:
                self.open_intent[i] = False
            if self.open_intent[i] is not None:
                out[i] = float(self.open_intent[i])
        return out, releasing

    def observe_targets(self, index, targets):
        for side, target in enumerate(targets):
            if target >= 0.5:
                self.since[side] = None
                self.grasp_checked[side] = False
            elif target <= 0.2 and self.since[side] is None:
                self.since[side] = index
        pending = [
            i
            for i, start in enumerate(self.since)
            if start is not None and not self.grasp_checked[i]
        ]
        if pending and all(index - self.since[i] >= GRASP_APPROACH_ROWS for i in pending):
            return [i for i, start in enumerate(self.since) if start is not None]
        return []


def wait_for_hand_state(
    feedback, robot, rig, folder, sides, deadline, *, opening, check_control=lambda: None
):
    # PC1 retains the last motor and gripper references, as it does during inference.
    start = time.monotonic()
    stable_since = None
    last_sample = None
    states = []
    positions = []
    dropped_sides = set()
    while time.monotonic() < min(start + 3.0, deadline):
        check_control()
        if (folder / "stop").exists():
            raise InterruptedError("Operator stopped grasp verification")
        measure(feedback)
        if robot.has_error():
            raise RuntimeError("SDK drive or bus fault during grasp verification")
        rig.snapshot()
        hands = feedback.messages["rt/hands/state"]
        states = [hands.left.grip_state.value, hands.right.grip_state.value]
        positions = [hands.left.realtime_position, hands.right.realtime_position]
        stamp = feedback.arrivals["rt/hands/state"]
        if stamp != last_sample and stamp >= start:
            last_sample = stamp
            ready = (
                all(
                    states[i] in (1, DH_GRIP_OBJECT_DROPPED) and positions[i] >= 0.99 for i in sides
                )
                if opening
                else all(
                    states[i] in (2, DH_GRIP_OBJECT_DROPPED) and positions[i] <= 0.5 for i in sides
                )
            )
            if ready:
                dropped_sides.update(i for i in sides if states[i] == DH_GRIP_OBJECT_DROPPED)
                if stable_since is None:
                    stable_since = stamp
                if stamp - stable_since >= (0.15 if opening else 0.4):
                    return dict(
                        sides=sides,
                        wait_s=round(stamp - start, 3),
                        states=states,
                        openings=positions,
                        dropped_sides=sorted(dropped_sides),
                    )
            else:
                stable_since = None
                dropped_sides.clear()
        time.sleep(0.01)
    operation = "opening" if opening else "grasp"
    raise RuntimeError(
        f"Unconfirmed {operation}: sides={sides}, states={states}, openings={positions}"
    )


class StereoLeftCamera:
    def __init__(self, spec):
        self.spec = spec
        self.count = 0

    def start(self):
        import cv2

        cfg = json.loads(DATA_CONFIG.read_text())
        path = Path(
            os.environ.get(
                "ATOM_D_TOP_DEVICE", "/dev/v4l/by-id/usb-DECXIN_DECXIN_Camera_01.00.00-video-index0"
            )
        )
        if path.resolve() != Path("/dev/video" + str(cfg["top_camera_id"])):
            raise RuntimeError("Top camera differs from the recording configuration")
        self.cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width * 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open DECXIN stereo camera")

    def read(self):
        import cv2

        ok, frame = self.cap.read()
        t = time.monotonic()
        if not ok or frame.shape != (self.spec.height, 2 * self.spec.width, 3):
            raise RuntimeError("Invalid stereo frame")
        self.count += 1
        rgb = cv2.cvtColor(frame[:, : self.spec.width], cv2.COLOR_BGR2RGB)
        return rgb, self.count, t * 1000, t

    def close(self):
        if hasattr(self, "cap"):
            self.cap.release()


def camera_rig():
    cfg = json.loads(DATA_CONFIG.read_text())
    size = cfg["top_camera_resolution"]
    specs = [CameraSpec("top_camera", "DECXIN", size["w"], size["h"], cfg["top_camera_fps"], 60)]
    for side in ("left", "right"):
        specs.append(CameraSpec(side + "_camera", cfg[side + "_hand_camera_sn"], 640, 480, 30, 60))
    return CameraRig(
        specs,
        factory=lambda s: StereoLeftCamera(s) if s.name == "top_camera" else RealSenseCamera(s),
    )


def measure(feedback):
    q, fsm, hands = feedback.sample(lambda: False)
    sides = (hands.left, hands.right)
    if fsm != 2 or any(h.init_result_state.value != 1 for h in sides):
        states = [(h.init_result_state.value, h.grip_state.value) for h in sides]
        raise RuntimeError(
            f"Robot mode or gripper initialization changed: FSM={fsm}, "
            f"left/right (init, grip)={states}"
        )
    motors = feedback.messages["rt/upper/state"].motor_state
    upper = np.array([m.q for m in motors])
    dq = np.array([m.dq for m in motors])
    tau = np.array([m.tau_est for m in motors])
    if not np.isfinite(np.r_[upper, dq, tau]).all():
        raise RuntimeError("Invalid upper motor feedback")
    return state_vector(upper, [h.realtime_position for h in sides]), upper, dq, tau


def hardware_limits():
    params = json.loads((home.DRIVER / "robot_control_dds/controlParams_P2.json").read_text())[:17]
    names = ("torso_joint",) + STATE_NAMES[:7] + STATE_NAMES[8:15] + STATE_NAMES[16:]
    if tuple(row["name"] for row in params) != names:
        raise RuntimeError("Driver joint order changed")
    tree = ET.parse(home.URDF)
    limits = [tree.find(f"./joint[@name='{name}']/limit").attrib for name in names]
    return (
        np.array([row["kp"] for row in params]),
        np.array([row["kd"] for row in params]),
        np.array([float(row["effort"]) for row in limits]),
        np.array([float(row["velocity"]) for row in limits]),
    )


def request_chunk(ws, state, images, replay_seed=None):
    request = {"state": state, "images": images, "prompt": "Fold the T-shirt with both arms."}
    if replay_seed is not None:
        request["replay_seed"] = replay_seed
    ws.send(msgpack_numpy.packb(request))
    reply = ws.recv(timeout=3)
    if isinstance(reply, str):
        raise RuntimeError(reply)
    result = msgpack_numpy.unpackb(reply)
    if replay_seed is not None and result.get("seed") != replay_seed:
        raise RuntimeError("Policy did not reuse the requested sampling seed")
    return np.asarray(result["actions"]), result.get("seed")


def infer_actions(ws, state, images):
    return request_chunk(ws, state, images)[0]


def capture(feedback, rig, after=None):
    images, info = rig.snapshot()
    while after is not None and min(v["received_monotonic"] for v in info.values()) <= after:
        if time.monotonic() - after > 0.7:
            raise RuntimeError("No fresh camera frames after grasp confirmation")
        time.sleep(0.01)
        images, info = rig.snapshot()
    state, upper, _, _ = measure(feedback)
    return state, upper, images, info


class Prediction(NamedTuple):
    chunk: ActionChunk
    points: np.ndarray
    state: np.ndarray
    images: dict
    frame_info: dict
    seed: Optional[int]


def predict_observation(ws, observation, bounds, sequence, replay_seed=None):
    state, upper, images, info = observation
    stamp = min(data["received_monotonic"] for data in info.values())
    actions, seed = request_chunk(ws, state, images, replay_seed)
    chunk = ActionChunk.create(sequence, stamp, time.monotonic(), actions, horizon=50)
    points = actuator_targets(chunk.actions, upper, bounds, saturate_joints=True)
    return Prediction(chunk, points, state, images, info, seed)


def predict(ws, feedback, rig, bounds, sequence=1, after=None, replay_seed=None):
    prediction = predict_observation(
        ws, capture(feedback, rig, after), bounds, sequence, replay_seed
    )
    ChunkGate().accept(prediction.chunk, time.monotonic())
    return prediction


def hand_status(feedback):
    hands = feedback.messages["rt/hands/state"]
    return (hands.left.grip_state.value, hands.right.grip_state.value)


@dataclass
class PendingChunk:
    future: Future
    start_row: int
    started: float
    reference: np.ndarray
    tail: np.ndarray
    hands: tuple
    targets: np.ndarray
    boundary: bool = False
    hand_changed: bool = False

    def bridge_length(self, points, covered):
        """Use only the contiguous prefix inside the held-pose and hand envelope."""
        future = points[covered : covered + PREFETCH_ROWS]
        for count, row in enumerate(future):
            if (
                np.max(abs(row[1:17] - self.tail[-1, 1:17])) > np.deg2rad(2)
                or np.max(abs(row[17:] - self.targets)) > 0.2
            ):
                return count
        return len(future)

    def rejection(self, prediction, covered, elapsed, current, images, hands, maximum_age):
        """Check the part of the new prediction covered by commands already sent."""
        chunk, points, state, original_images, _, _ = prediction
        if covered != len(self.tail) or abs(elapsed - covered * PERIOD) > PERIOD:
            return "timing"
        if time.monotonic() - chunk.observation_time >= maximum_age:
            return "stale"
        if (
            self.hand_changed
            or hands != self.hands
            or np.max(abs(current[[7, 15]] - state[[7, 15]])) > 0.05
        ):
            return "hand_feedback"
        # The early plan may bridge only with the existing hand command.
        # A fresh boundary plan may change the hands on its first unsent row.
        end = covered if self.boundary else covered + 1
        if np.max(abs(points[:end, 17:] - self.targets)) > 0.2:
            return "hand_intent"
        # Both sets of references are in motor space. Encoder positions include
        # gravity deflection, so compare measured displacement, not absolute q.
        drift = current[JOINT_COLUMNS] - state[JOINT_COLUMNS]
        expected = self.tail[-1, 1:17] - self.reference[1:17]
        if np.max(abs(drift - expected)) > np.deg2rad(2):
            return "tracking"
        if np.max(abs(points[:covered, 1:17] - self.tail[:, 1:17])) > np.deg2rad(2):
            return "overlap"
        if not self.boundary and not self.bridge_length(points, covered):
            return "bridge_excursion"
        if np.max(abs(points[covered, 1:17] - self.tail[-1, 1:17])) > np.deg2rad(4):
            return "seam"
        for name, image in images.items():
            delta = image[::8, ::8].astype(np.int16) - original_images[name][::8, ::8]
            if np.mean(abs(delta)) > 8:
                return "scene"
        return None


def save_view(folder, name, rig):
    save_images(folder / name, rig.snapshot()[0])


def save_images(path, images):
    from PIL import Image

    panel = np.concatenate(
        [images[key] for key in ("top_camera", "left_camera", "right_camera")], axis=1
    )
    Image.fromarray(panel).save(path)


def playback(
    ws,
    feedback,
    rig,
    bounds,
    robot,
    limits,
    folder,
    report,
    full,
    initial_prediction,
    pool=None,
    check_control=lambda: None,
):
    kp, kd, rated, velocity = limits
    gate = ChunkGate()
    repeating = LoopGuard()
    grasps = GraspTracker()
    rate = np.minimum(velocity, np.deg2rad(4) / PERIOD)
    _, q, dq, tau = measure(feedback)
    last_sent = np.clip(q + (tau + kd * dq) / kp, bounds[:, 0], bounds[:, 1])
    last_sent[0] = q[0]
    robot.robot.gripper_force.update(
        left_force=GRIPPER_FORCE_PERCENT, right_force=GRIPPER_FORCE_PERCENT
    )
    report.update(
        inference_mode="async" if pool is not None else "sync",
        handoffs=[],
        completed_chunks=0,
        max_rows=1600 if full else VALIDATION_FRAMES,
        max_wall_seconds=120,
        gripper_force_percent=GRIPPER_FORCE_PERCENT,
        gripper_feedback=[],
        peak_tracking_deg=0.0,
        grasp_events=[],
        release_events=[],
        drop_events=[],
        executed_rows_by_chunk=[],
        peak_reference_step_deg=0.0,
        peak_effort_fraction=0.0,
        saturated_joint_values=0,
    )
    began = time.monotonic()
    sequence = 0
    observed_after = None
    last_dropped = set()
    pending = None
    ended = began
    while report["rows_sent"] < report["max_rows"]:
        check_control()
        sequence += 1
        report["active_chunk"] = sequence
        report["executed_rows_by_chunk"].append(0)
        first = 0
        replay_seed = None
        prediction = initial_prediction if sequence == 1 else None
        handoff = dict(chunk=sequence, source="sync", skipped_rows=0)
        bridge = False
        if pending is not None:
            # Drain even a discarded response before reusing the same WebSocket.
            candidate = pending.future.result()
            if type(candidate.seed) is not int:
                raise RuntimeError("Prefetched policy response has no sampling seed")
            covered = report["rows_sent"] - pending.start_row
            reason = "gripper_event"
            if observed_after is None:
                boundary_obs = capture(feedback, rig)
                status = hand_status(feedback)
                reason = pending.rejection(
                    candidate,
                    covered,
                    ended - pending.started,
                    boundary_obs[0],
                    boundary_obs[2],
                    status,
                    gate.maximum_age,
                )
            job, pending = pending, None
            if reason is None:
                prediction, first = candidate, covered
                bridge = not job.boundary
                handoff.update(source="prefetch" if bridge else "boundary", skipped_rows=first)
                if bridge:
                    # Re-anchor on the same boundary observation used by synchronous
                    # playback. Only a short prefix ever comes from the earlier view.
                    count = job.bridge_length(candidate.points, first)
                    tail = candidate.points[first : first + count]
                    started = time.monotonic()
                    future = pool.submit(
                        predict_observation, ws, boundary_obs, bounds, sequence + 1, candidate.seed
                    )
                    pending = PendingChunk(
                        future,
                        report["rows_sent"],
                        started,
                        last_sent,
                        tail,
                        status,
                        job.targets,
                        boundary=True,
                    )
            else:
                handoff["discarded"] = reason
                replay_seed = candidate.seed
        if prediction is None:
            prediction = predict(ws, feedback, rig, bounds, sequence, observed_after, replay_seed)
        observed_after = None
        chunk, points, state, images, _, seed = prediction
        handoff["seed"] = seed
        report["handoffs"].append(handoff)
        np.savez(folder / f"chunk-{sequence:02}.npz", state=state, actions=chunk.actions, **images)
        if not bridge and repeating.observe(chunk.actions, state):
            return "stopped_repeated_predictions"
        points = points[: first + report["max_rows"] - report["rows_sent"]]
        if bridge:
            points = points[: first + len(pending.tail)]
        gate.accept(chunk, time.monotonic())
        for index in range(first, len(points)):
            row = points[index]
            report["active_row"] = index
            hand_targets, releasing = (
                grasps.hand_targets(row[17:19], state[[7, 15]]) if full else (row[17:19], [])
            )
            report["saturated_joint_values"] += int(
                np.count_nonzero(chunk.actions[index, JOINT_COLUMNS] != row[1:17])
            )
            previous = last_sent
            diff = row[:17] - previous
            report["peak_reference_step_deg"] = max(
                report["peak_reference_step_deg"], float(np.rad2deg(max(abs(diff[1:]))))
            )
            duration = max(PERIOD, float(np.max(abs(diff[1:]) / rate[1:])))
            ticks = max(2, int(np.ceil(duration / 0.0125)))
            tick_period = duration / ticks
            row_start = time.monotonic()
            for tick in range(1, ticks + 1):
                if (folder / "stop").exists():
                    return "stopped_by_supervisor"
                if time.monotonic() - began > report["max_wall_seconds"]:
                    return "wall_time_limit"
                if tick == 1 and index % 5 == 0:
                    rig.snapshot()
                current, actual, _, effort = measure(feedback)
                if robot.has_error():
                    raise RuntimeError("ATOM D SDK reports a drive or bus fault")
                hands = feedback.messages["rt/hands/state"]
                if pending is not None and hand_status(feedback) != pending.hands:
                    pending.hand_changed = True
                dropped = {
                    side
                    for side in ("left", "right")
                    if getattr(hands, side).grip_state.value == DH_GRIP_OBJECT_DROPPED
                }
                if dropped - last_dropped:
                    event = dict(
                        chunk=sequence,
                        row=report["rows_sent"],
                        sides=sorted(dropped - last_dropped),
                        feedback_s=round(feedback.arrivals["rt/hands/state"] - began, 3),
                    )
                    report["drop_events"].append(event)
                    print("GRIPPER_DROP_REPORTED", json.dumps(event), flush=True)
                last_dropped = dropped
                command = previous + diff * (tick / ticks)
                tracking_deg = float(np.rad2deg(np.max(abs(command[1:] - actual[1:]))))
                report["peak_tracking_deg"] = max(report["peak_tracking_deg"], tracking_deg)
                report["peak_effort_fraction"] = max(
                    report["peak_effort_fraction"], float(max(abs(effort) / rated))
                )
                if time.monotonic() - (row_start + (tick - 1) * tick_period) >= PERIOD:
                    raise RuntimeError("Servo scheduling deadline missed")
                if tick == 1:
                    hand_command_time = time.monotonic() - began
                    if index == first and sequence > 1:
                        handoff["boundary_wait_ms"] = round(
                            1000 * (began + hand_command_time - ended), 2
                        )
                check_control()
                robot.command_joint_state(
                    command[1:8],
                    command[8:15],
                    hand_targets[:1],
                    hand_targets[1:],
                    command[15:17],
                    send_hand=(tick == 1),
                )
                last_sent = command
                if tick == ticks:
                    report["rows_sent"] += 1
                    report["executed_rows_by_chunk"][-1] += 1
                    report["feedback"].append(current.tolist())
                    report["gripper_feedback"].append(
                        dict(
                            command_s=hand_command_time,
                            feedback_s=feedback.arrivals["rt/hands/state"] - began,
                            target=hand_targets.tolist(),
                            policy_target=row[17:19].tolist(),
                            state=[hands.left.grip_state.value, hands.right.grip_state.value],
                            force_readback=[
                                hands.left.get_force_value,
                                hands.right.get_force_value,
                            ],
                        )
                    )
                time.sleep(max(0.0, row_start + tick * tick_period - time.monotonic()))
            if full:
                sides = grasps.observe_targets(report["rows_sent"] - 1, hand_targets)
                if releasing:
                    release = wait_for_hand_state(
                        feedback,
                        robot,
                        rig,
                        folder,
                        releasing,
                        began + report["max_wall_seconds"],
                        opening=True,
                        check_control=check_control,
                    )
                    release.update(row=report["rows_sent"] - 1, chunk=sequence)
                    report["release_events"].append(release)
                    print("RELEASE_CONFIRMED", json.dumps(release), flush=True)
                if sides:
                    event = wait_for_hand_state(
                        feedback,
                        robot,
                        rig,
                        folder,
                        sides,
                        began + report["max_wall_seconds"],
                        opening=False,
                        check_control=check_control,
                    )
                    for side in sides:
                        grasps.grasp_checked[side] = True
                    event.update(row=report["rows_sent"] - 1, chunk=sequence)
                    report["grasp_events"].append(event)
                    label = (
                        "GRASP_CHECKED_WITH_DROP" if event["dropped_sides"] else "GRASP_CONFIRMED"
                    )
                    print(label, json.dumps(event), flush=True)
                if sides or releasing:
                    observed_after = time.monotonic()
                    break
                segment_end = (
                    min(len(points), CARRY_REPLAN_ROWS)
                    if any(grasps.grasp_checked)
                    else len(points)
                )
                if index + 1 >= segment_end:
                    break
                remaining = segment_end - index - 1
                if (
                    pool is not None
                    and pending is None
                    and remaining == PREFETCH_ROWS
                    and report["rows_sent"] + remaining < report["max_rows"]
                ):
                    tail = points[index + 1 : segment_end]
                    # Only speculate through stable binary hand targets. Pending
                    # closures and releases get an observation after confirmation.
                    if (
                        None not in grasps.open_intent
                        and all(
                            start is None or checked
                            for start, checked in zip(grasps.since, grasps.grasp_checked)
                        )
                        and np.max(abs(tail[:, 17:] - hand_targets)) <= 0.2
                    ):
                        observed = capture(feedback, rig)
                        started = time.monotonic()
                        future = pool.submit(
                            predict_observation, ws, observed, bounds, sequence + 1
                        )
                        pending = PendingChunk(
                            future,
                            report["rows_sent"],
                            started,
                            last_sent,
                            tail,
                            hand_status(feedback),
                            hand_targets.copy(),
                        )
        ended = time.monotonic()
        end = measure(feedback)[0]
        if robot.has_error():
            raise RuntimeError("ATOM D SDK reports a drive or bus fault")
        report["completed_chunks"] += 1
        report["final_state"] = end.tolist()
        report["elapsed_seconds"] = round(time.monotonic() - began, 3)
        if not full:
            measured = np.asarray(report["feedback"] + [end.tolist()])
            report["peak_measured_motion_deg"] = float(
                np.max(abs(np.rad2deg(measured[:, JOINT_COLUMNS] - state[JOINT_COLUMNS])))
            )
            report["playback_seconds"] = report["elapsed_seconds"]
        save_view(folder, f"after-{sequence:02}.jpg", rig)
        print(
            "EPISODE_PROGRESS",
            json.dumps(
                dict(
                    chunk=sequence,
                    rows=report["rows_sent"],
                    elapsed=report["elapsed_seconds"],
                    openings=end[[7, 15]].tolist(),
                    displacement_deg=round(
                        float(np.max(abs(np.rad2deg((end - state)[JOINT_COLUMNS])))), 2
                    ),
                )
            ),
            flush=True,
        )
    return "episode_frame_limit" if full else "hardware_validation_completed"


def run(args, folder, report, check_control=lambda: None):
    def cancelled():
        check_control()
        return (folder / "stop").exists()

    with ExitStack() as stack:
        stack.enter_context(atom_d_control_lease())
        ws = stack.enter_context(
            policy_session(args.weights, folder, report, check_control=check_control)
        )
        if args.observation_npz:
            with np.load(args.observation_npz, allow_pickle=False) as sample:
                state = sample["state"]
                images = {key: sample[key] for key in ("top_camera", "left_camera", "right_camera")}
            actions = infer_actions(ws, state, images)
            now = time.monotonic()
            chunk = ActionChunk.create(1, now, now, actions, horizon=50)
            np.savez(folder / "prediction.npz", state=state, actions=chunk.actions, **images)
            report.update(
                observation_file=str(args.observation_npz.resolve()),
                predicted_shape=list(actions.shape),
            )
            return "recorded_observation_inferred"

        check_control()
        configure_sdk()
        bounds = controller_bounds(read_controller_settings(home.PC1))
        limits = hardware_limits()
        feedback = None if args.home_start else home.Feedback()
        rig = stack.enter_context(camera_rig())
        rig.wait_ready(check_control=check_control)
        check_control()
        robot = None
        if args.home_start:
            report["home_start"], feedback, robot = home.home_session(
                execute=args.execute, cancelled=cancelled
            )
            print("ATOM_HOME_START_RESULT", json.dumps(report["home_start"]), flush=True)
            if args.execute and report["home_start"]["status"] != "home_start_completed":
                raise RuntimeError("Home-start did not complete")
        else:
            measure(feedback)
            if args.execute:
                robot = home.connect_upper_driver(feedback)
        if args.execute and (
            robot.get_robot_type() != "ATOM D" or robot.hand_type != "dh" or robot.has_error()
        ):
            raise RuntimeError("Native ATOM D driver startup failed")
        # Wait for observations captured after the start pose has been established.
        cutoff = time.monotonic()
        while min(v["received_monotonic"] for v in rig.snapshot()[1].values()) <= cutoff:
            if time.monotonic() - cutoff >= 0.7:
                raise RuntimeError("No fresh camera frames for the initial pose")
            time.sleep(0.01)
        check_control()
        initial_prediction = predict(ws, feedback, rig, bounds)
        check_control()
        chunk, points, state, images, info, _ = initial_prediction
        save_images(folder / "views.jpg", images)
        np.savez(folder / "prediction.npz", state=state, actions=chunk.actions, **images)
        target = np.asarray(json.loads(home.CONFIG.read_text())["physical_state"])
        errors_deg = np.rad2deg((state - target)[JOINT_COLUMNS])
        if not args.execute or not args.full_episode:
            _, q, dq, tau = measure(feedback)
            kp, kd, _, _ = limits
            held = q + (tau + kd * dq) / kp
            prefix = points[:VALIDATION_FRAMES]
            steps = np.diff(np.vstack([held, prefix[:, :17]]), axis=0)
            report.update(
                max_step_deg=float(np.rad2deg(abs(steps[:, 1:]).max())),
                max_excursion_deg=float(np.rad2deg(abs(prefix[:, 1:17] - q[1:]).max())),
                full_chunk_excursion_deg=float(np.rad2deg(abs(points[:, 1:17] - q[1:]).max())),
                gripper_range=[prefix[:, 17:].min(0).tolist(), prefix[:, 17:].max(0).tolist()],
            )
        report.update(
            initial_state=state.tolist(),
            frame_info=info,
            predicted_frames=len(points),
            predicted_shape=list(chunk.actions.shape),
            home_peak_joint=STATE_NAMES[JOINT_COLUMNS[np.argmax(abs(errors_deg))]],
            home_max_error_deg=float(np.max(abs(errors_deg))),
            initial_openings=state[[7, 15]].tolist(),
            observation_age_ms=round(1000 * (chunk.received_time - chunk.observation_time), 1),
        )
        print(
            "LIVE_CHUNK",
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "predicted_shape",
                        "home_peak_joint",
                        "home_max_error_deg",
                        "initial_openings",
                    )
                }
            ),
            flush=True,
        )
        if not args.execute:
            return "observation_verified"
        pool = None
        if args.full_episode and args.inference_mode == "async" and report["supports_replay_seed"]:
            pool = stack.enter_context(
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")
            )
        elif args.full_episode and args.inference_mode == "async":
            report["sync_reason"] = "policy_does_not_support_replay_seed"
            print("INFERENCE_MODE sync: policy does not support replay_seed", flush=True)
        try:
            return playback(
                ws,
                feedback,
                rig,
                bounds,
                robot,
                limits,
                folder,
                report,
                args.full_episode,
                initial_prediction,
                pool,
                check_control,
            )
        finally:
            try:
                save_view(folder, "last.jpg", rig)
            except Exception as exc:
                report["last_view_error"] = str(exc)


def configure_sdk():
    os.environ["CYCLONEDDS_URI"] = home.DDS_URI
    sys.path[:0] = [str(home.DRIVER), str(home.DRIVER / "robot_control_dds")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        "--artifact",
        default=ARTIFACT,
        metavar="PATH",
        help="Pi0.5 Artifact directory, checkpoint pointer, or declared deployment weights file",
    )
    parser.add_argument(
        "--observation-npz", type=Path, help="Infer once from recorded state and camera images"
    )
    parser.add_argument(
        "--inference-mode",
        choices=("async", "sync"),
        default="async",
        help="Full episodes prefetch the next chunk by default",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Open the live console: Enter starts inference, h returns home, Ctrl+C exits",
    )
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="Run successive 50-row chunks from fresh observations",
    )
    parser.add_argument(
        "--home-start",
        action="store_true",
        help="Open the grippers and return to the recorded start pose (motion requires --execute)",
    )
    parser.add_argument("--site-clear", action="store_true")
    args = parser.parse_args(argv)
    if args.observation_npz and (args.execute or args.full_episode or args.home_start):
        parser.error(
            "--observation-npz cannot be combined with --execute, --home-start or --full-episode"
        )
    if args.execute and not args.site_clear:
        parser.error("Motion requires --site-clear")

    if args.execute:
        if not sys.stdin.isatty():
            parser.error("Live execution requires an interactive terminal (use ssh -t)")
        return operator_session(args, perform_trial)

    def interrupted(*_):
        raise InterruptedError("Trial interrupted")

    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    return perform_trial(args, "inference", lambda: None)


def perform_trial(args, operation, check_control):
    folder = (
        ROOT / "runs/atom_d_inference" / datetime.datetime.now().strftime("trial-%Y%m%d-%H%M%S-%f")
    )
    folder.mkdir(parents=True)
    report = dict(
        execute=args.execute,
        operation=operation,
        home_start_requested=args.home_start or operation == "home",
        rows_sent=0,
        feedback=[],
        folder=str(folder),
    )
    print("EPISODE_FOLDER", str(folder), flush=True)
    try:
        check_control()
        if operation == "home":
            if not (args.execute and args.site_clear):
                raise ValueError("Home-start requires --execute and --site-clear")
            configure_sdk()

            def cancelled():
                check_control()
                return (folder / "stop").exists()

            with atom_d_control_lease():
                report["home_start"] = home.run_home(execute=True, cancelled=cancelled)
            report["status"] = report["home_start"]["status"]
        else:
            report["status"] = run(args, folder, report, check_control)
    except OperatorCancelled as exc:
        report.update(status="cancelled_by_operator", error=str(exc))
    except Exception as exc:
        report.update(status="stopped", error=str(exc))
    (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "TRIAL_RESULT",
        json.dumps({k: v for k, v in report.items() if k not in ("feedback", "gripper_feedback")}),
        flush=True,
    )
    if report["status"] == "cancelled_by_operator":
        return CANCELLED
    return 2 if report["status"] in ("stopped", "stopped_by_supervisor") else 0


if __name__ == "__main__":
    raise SystemExit(main())
