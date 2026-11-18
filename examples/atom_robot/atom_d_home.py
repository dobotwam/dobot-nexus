"""Return the desktop ATOM D to a recorded physical pose using its installed driver."""

import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .controller_limits import controller_bounds, read_controller_settings
from .core import STATE_NAMES
from .torque import OuterTorqueCompensator

CONFIG = Path(
    os.environ.get("ATOM_D_HOME_CONFIG", Path(__file__).parent / "config/atom_d_home.json")
).expanduser()
DRIVER = Path(
    os.environ.get(
        "ATOM_D_DRIVER",
        Path.home() / "dobot/code/lerobot/src/lerobot/robots/dobot_atom_gripper/robot_control",
    )
).expanduser()
URDF = Path(
    os.environ.get(
        "ATOM_D_URDF", Path.home() / "ATOM/ATOM_teleoperate/assets/ATOM_P2/urdf/atom.urdf"
    )
).expanduser()
PC1 = os.environ.get("ATOM_D_PC1", "192.168.8.234")
DDS_URI = os.environ.get(
    "CYCLONEDDS_URI",
    '<CycloneDDS><Domain Id="0"><General><Interfaces><NetworkInterface '
    'address="192.168.8.13" multicast="true"/></Interfaces><AllowMulticast>spdp'
    "</AllowMulticast><DontRoute>true</DontRoute></General></Domain></CycloneDDS>",
)


class Feedback:
    """Independent subscribers. The installed UpperControl constructor itself issues commands."""

    def __init__(self):
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from dobot_atom.msg import dds_

        self.participant = DomainParticipant(0)
        self.discovery = BuiltinDataReader(self.participant, BuiltinTopicDcpsPublication)
        qos = Qos(Policy.History.KeepLast(1), Policy.Reliability.BestEffort)
        self.readers = {
            name: DataReader(self.participant, Topic(self.participant, name, kind), qos=qos)
            for name, kind in [
                ("rt/upper/state", dds_.UpperState_),
                ("rt/main/nodes/state", dds_.MainNodesState_),
                ("rt/hands/state", dds_.DHGripperState_),
            ]
        }
        self.messages = {}
        self.arrivals = {}
        self.owner = None

    def sample(self, cancelled):
        if cancelled():
            raise InterruptedError("Home-start cancelled")
        for endpoint in self.discovery.take(100):
            if (
                endpoint.sample_info.valid_data
                and endpoint.topic_name in ("rt/upper/cmd", "rt/hands/cmd")
                and str(endpoint.participant_key) != self.owner
            ):
                raise RuntimeError("Another command publisher is present: " + endpoint.topic_name)
        now = time.monotonic()
        for name, reader in self.readers.items():
            for message in reader.take(1):
                if message.sample_info.valid_data:
                    self.messages[name] = message
                    self.arrivals[name] = now
            if now - self.arrivals.get(name, -float("inf")) > (
                0.15 if name == "rt/upper/state" else 0.5
            ):
                raise RuntimeError("Missing or stale feedback: " + name)
        upper = self.messages["rt/upper/state"]
        if "".join(upper.robot_type).rstrip("\x00") != "ATOM D" or upper.is_upper_control:
            raise RuntimeError("Unexpected model or upper-control ownership")
        nodes = self.messages["rt/main/nodes/state"]
        for i, axis in enumerate([*nodes.left_arm, *nodes.right_arm, *nodes.head]):
            if (
                axis.is_virtual
                or axis.servo_state != 3
                or axis.node_state != 5
                or axis.display_op_mode != 11
                or axis.error_code
                or axis.warn_code
                or axis.pos_err_code
                or axis.vel_err_code
                or axis.torque_err_code
            ):
                raise RuntimeError("Drive not ready: upper joint " + str(i + 1))
        q = np.array([m.q for m in upper.motor_state[1:]], dtype=float)
        if q.shape != (16,) or not np.all(np.isfinite(q)):
            raise RuntimeError("Invalid measured joint positions")
        return q, upper.fsm_id, self.messages["rt/hands/state"]


def connect_upper_driver(feedback):
    """Attach the native readers and command methods without restarting DH calibration."""
    from robot_upper_control import Control_sim, UpperControl

    class AttachedUpperControl(UpperControl):
        def __init__(self):
            self.robot = Control_sim()
            self.hand_type = self.robot.hand_type
            self.pc1_controller_version = self.robot.pc1_controller_version
            self.fisrt_control = True  # SDK field used to latch the waist reference
            self.torso = 0
            feedback.owner = str(self.robot.participant.guid)
            self.robot.start()

        def home_grippers(self, sides):
            from DDSHelper import get_dh_gripper_cmd
            from dobot_atom.msg import dds_

            force = self.robot.gripper_force
            cmd = get_dh_gripper_cmd(
                left_force=float(force["left_force"]), right_force=float(force["right_force"])
            )
            for side in sides:
                getattr(cmd, side).init_cmd = dds_.DHGripperInitCmd.DH_INIT_HOMING
            self.robot.inspire_cmd_writer.write(cmd)

    return AttachedUpperControl()


def _hand_status(hands):
    return {
        side: dict(
            init=h.init_result_state.name,
            grip=h.grip_state.name,
            opening=float(h.realtime_position),
        )
        for side, h in [("left", hands.left), ("right", hands.right)]
    }


def _both_open(hands):
    return all(
        h.init_result_state.value == 1
        and h.grip_state.value == 1
        and 0.99 <= h.realtime_position <= 1.0
        for h in (hands.left, hands.right)
    )


def prepare_home_grippers(robot, feedback, cancelled, *, clock=time.monotonic, sleep=time.sleep):
    """Confirm native homing, then explicitly command and verify the DH open position."""

    def sample():
        _, fsm, hands = feedback.sample(cancelled)
        if robot.has_error():
            raise RuntimeError(
                f"SDK fault during Home-start gripper setup: FSM={fsm}, {_hand_status(hands)}"
            )
        return fsm, hands

    def timed_out(stage):
        return RuntimeError(
            f"Home-start {stage} timed out: FSM={fsm}, "
            f"homing_commands={homing_commands}, open_commands={open_commands}, "
            f"{json.dumps(_hand_status(hands))}"
        )

    homing_commands = {"left": 0, "right": 0}
    open_commands = 0
    fsm, hands = sample()
    before = _hand_status(hands)
    print("ATOM_D_GRIPPER_START", json.dumps(dict(fsm=fsm, hands=before)), flush=True)
    if fsm != 2:
        robot.set_fsm_state()
        deadline = clock() + 10.0
        while fsm != 2:
            if clock() >= deadline:
                raise timed_out("FSM transition")
            sleep(0.02)
            fsm, hands = sample()

    # A fresh connection need not recalibrate hands that already know their endpoints.
    start = clock()
    last_homing = -float("inf")
    while True:
        if fsm != 2:
            raise RuntimeError(
                f"Home-start lost FSM 2 while homing grippers: FSM={fsm}, {_hand_status(hands)}"
            )
        init = [h.init_result_state.value for h in (hands.left, hands.right)]
        if init == [1, 1]:
            break
        if clock() - start >= 20.0:
            raise timed_out("gripper homing")
        if 2 not in init and clock() - last_homing >= 2.0:
            pending = [
                side
                for side, state in zip(("left", "right"), init)
                if state == 0 and homing_commands[side] < 2
            ]
            if pending:
                robot.home_grippers(pending)
                for side in pending:
                    homing_commands[side] += 1
                last_homing = clock()
        sleep(0.02)
        fsm, hands = sample()

    # DH_INIT_HOMING can finish at opening=1 with the old HOLDING status still set.
    # Send a distinct DH_INIT_NONE position command, including when already fully open.
    print("ATOM_D_GRIPPER_OPENING", json.dumps(_hand_status(hands)), flush=True)
    start = clock()
    last_stamp = feedback.arrivals["rt/hands/state"]
    robot.robot.open_gripper()
    open_commands = 1
    last_send = clock()
    stable_since = None
    while clock() - start < 10.0:
        sleep(0.02)
        fsm, hands = sample()
        if fsm != 2 or any(h.init_result_state.value != 1 for h in (hands.left, hands.right)):
            raise RuntimeError(
                f"Home-start mode or initialization lost while opening: "
                f"FSM={fsm}, {_hand_status(hands)}"
            )
        stamp = feedback.arrivals["rt/hands/state"]
        if stamp <= last_stamp or stamp <= last_send:
            continue
        last_stamp = stamp
        if _both_open(hands):
            if stable_since is None:
                stable_since = stamp
            elif stamp - stable_since >= 0.3:
                result = dict(
                    before=before,
                    after=_hand_status(hands),
                    homing_commands=homing_commands,
                    open_commands=open_commands,
                    opening_seconds=round(clock() - start, 3),
                )
                print("ATOM_D_GRIPPERS_READY", json.dumps(result), flush=True)
                return result
        else:
            stable_since = None
            if open_commands < 3 and clock() - last_send >= 2.0:
                robot.robot.open_gripper()
                open_commands += 1
                last_send = clock()
    raise timed_out("gripper opening")


def run_home(*, execute, cancelled):
    result, _, _ = home_session(execute=execute, cancelled=cancelled)
    return result


def home_session(*, execute, cancelled):
    """Return the homing result and its live feedback/driver for subsequent inference."""
    config = json.loads(CONFIG.read_text())
    state = np.asarray(config["physical_state"], dtype=float)
    if (
        config["robot_type"] != "ATOM D"
        or config["joint_units"] != "rad"
        or config["state_order"] != "left7,left_opening,right7,right_opening,head_yaw,head_pitch"
        or state.shape != (18,)
        or not np.all(np.isfinite(state))
        or not np.all(state[[7, 15]] == 1)
    ):
        raise ValueError("Invalid ATOM D Home configuration")
    target = np.r_[state[:7], state[8:15], state[16:]]
    action = np.asarray(config["motor_action"], dtype=float)
    if action.shape != (18,) or not np.all(np.isfinite(action)) or not np.all(action[[7, 15]] == 1):
        raise ValueError("Invalid paired ATOM D motor action")
    motor_goal = np.r_[action[:7], action[8:15], action[16:]]
    bounds = controller_bounds(read_controller_settings(PC1))[1:]
    if np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
        raise ValueError("Home target exceeds active controller limits")
    if np.any(motor_goal < bounds[:, 0]) or np.any(motor_goal > bounds[:, 1]):
        raise ValueError("Recorded motor action exceeds PC1 position limits")
    params = json.loads((DRIVER / "robot_control_dds/controlParams_P2.json").read_text())[:17]
    names = ("torso_joint",) + STATE_NAMES[:7] + STATE_NAMES[8:15] + STATE_NAMES[16:]
    if tuple(row["name"] for row in params) != names:
        raise ValueError("Native ATOM D gain-table order differs from the recording")
    kp = np.array([row["kp"] for row in params])
    kd = np.array([row["kd"] for row in params])
    model = ET.parse(URDF)
    rated = np.array(
        [float(model.find(f"./joint[@name='{name}']/limit").attrib["effort"]) for name in names]
    )
    trim = OuterTorqueCompensator(kp, rated, max_bias_deg=2.0)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://" + PC1 + ":22000/settings/grip", timeout=2) as response:
        if json.load(response)["type"] != "dh":
            raise RuntimeError("Expected DH grippers")
    os.environ["CYCLONEDDS_URI"] = DDS_URI
    sys.path.insert(0, str(DRIVER / "robot_control_dds"))
    feedback = Feedback()
    time.sleep(2)
    start, fsm, _ = feedback.sample(cancelled)
    for _ in range(10):
        time.sleep(0.05)
        q, fsm, _ = feedback.sample(cancelled)
        if np.max(np.abs(q - start)) > np.deg2rad(0.3):
            raise RuntimeError("Robot is moving")
    result = dict(
        robot_type="ATOM D",
        source=config["source"],
        frame_index=config["frame_index"],
        initial_fsm=int(fsm),
        initial_q_rad=q.tolist(),
        target_q_rad=target.tolist(),
    )
    print("ATOM_D_HOME_PREFLIGHT", json.dumps(result), flush=True)
    if not execute:
        return dict(result, status="read_only_preflight_passed"), feedback, None
    if cancelled():
        raise InterruptedError("Home-start cancelled")
    sys.path.insert(0, str(DRIVER))
    robot = connect_upper_driver(feedback)
    if (
        cancelled()
        or robot.get_robot_type() != "ATOM D"
        or robot.hand_type != "dh"
        or robot.has_error()
    ):
        raise RuntimeError("Native ATOM D startup failed")
    result["gripper_startup"] = prepare_home_grippers(robot, feedback, cancelled)

    static_effort = []
    initial, _, _ = feedback.sample(cancelled)
    for _ in range(25):
        time.sleep(0.02)
        q, fsm, hands = feedback.sample(cancelled)
        if not _both_open(hands):
            raise RuntimeError(
                f"Home-start open gripper feedback lost before arm motion: {_hand_status(hands)}"
            )
        if np.max(abs(q - initial)) > np.deg2rad(0.3) or fsm != 2:
            raise RuntimeError("Robot is not stationary after device initialization")
        static_effort.append([m.tau_est for m in feedback.messages["rt/upper/state"].motor_state])
    tau = np.median(static_effort, axis=0)
    if not np.isfinite(tau).all() or np.any(abs(tau) > rated):
        raise RuntimeError("Invalid static support torque")
    # Native SDK emits zero torque and fixed PD gains. Convert measured holding
    # torque to its equivalent position reference before starting the ramp.
    start = q.copy()
    motor_start = start + tau[1:] / kp[1:]
    # A smooth cubic has peak slope 1.5. Bound the maximum reference speed to 5.7 deg/s.
    duration = max(
        1.0,
        1.5
        * max(np.max(abs(target - start)), np.max(abs(motor_goal - motor_start)))
        / np.deg2rad(5.7),
    )
    reference = start.copy()
    print("ATOM_D_HOME_MOVING", json.dumps({"duration_s": duration, "fsm": int(fsm)}), flush=True)
    stable_since = None
    progress = 0.0
    last_tick = time.monotonic()
    deadline = last_tick + max(60.0, 3 * duration)
    next_status = last_tick
    while time.monotonic() < deadline:
        q, fsm, hands = feedback.sample(cancelled)
        error_deg = np.rad2deg(q - reference)
        if fsm != 2 or robot.has_error() or np.max(abs(error_deg)) > 2.0:
            raise RuntimeError("Home tracking or drive fault, degrees: " + str(error_deg.tolist()))
        if not _both_open(hands):
            raise RuntimeError(f"Open gripper feedback lost: {_hand_status(hands)}")
        now = time.monotonic()
        dt = now - last_tick
        last_tick = now
        if dt > 0.1:
            raise RuntimeError("Home command loop stalled")
        lag = np.clip((np.max(abs(error_deg)) - 0.5) / 0.7, 0.0, 1.0)
        progress = min(1.0, progress + min(dt, 0.02) * (1 - lag) / duration)
        phase = progress**2 * (3 - 2 * progress)
        reference = start + (target - start) * phase
        if now >= next_status:
            print(
                "ATOM_D_HOME_PROGRESS",
                json.dumps(
                    {
                        "path_percent": round(100 * phase, 1),
                        "tracking_deg": np.round(error_deg, 2).tolist(),
                        "remaining_deg": round(float(np.max(abs(np.rad2deg(q - target)))), 2),
                    }
                ),
                flush=True,
            )
            next_status = now + 5
        adjustment = trim.step(np.r_[0.0, reference], np.r_[0.0, q], dt)[1:] / kp[1:]
        motor = motor_start + (motor_goal - motor_start) * phase + adjustment
        dq = np.array([m.dq for m in feedback.messages["rt/upper/state"].motor_state[1:]])
        predicted = kp[1:] * (motor - q) - kd[1:] * dq
        if (
            not np.isfinite(predicted).all()
            or np.any(abs(predicted) > rated[1:])
            or np.any(motor < bounds[:, 0])
            or np.any(motor > bounds[:, 1])
        ):
            raise RuntimeError("Motor reference exceeds controller position or rated effort limits")
        robot.command_joint_state(
            motor[:7], motor[7:14], state[7:8], state[15:16], motor[14:], send_hand=False
        )
        if progress == 1.0 and np.max(abs(np.rad2deg(q - target))) <= 0.8:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 0.5:
                return (
                    dict(
                        result,
                        status="home_start_completed",
                        final_q_rad=q.tolist(),
                        final_fsm=int(fsm),
                        max_error_deg=float(np.max(abs(np.rad2deg(q - target)))),
                    ),
                    feedback,
                    robot,
                )
        else:
            stable_since = None
        time.sleep(0.02)
    raise RuntimeError(
        f"Home timed out at {100 * phase:.1f}% of trajectory; remaining degrees: "
        + str(np.rad2deg(q - target).tolist())
    )
