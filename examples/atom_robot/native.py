"""Version-pinned ATOM DDS adapter. Readers at startup; writers only on acquire.

The installed SDK's reviewed DDS session supplies wire types, drive/chassis
checks and explicit ownership handover. The demo entry point is never executed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .controller_limits import (
    apply_controller_limits,
    assert_controller_limits_unchanged,
    read_controller_settings,
)
from .core import RuntimeFault, finite_array, state_vector
from .torque import OuterTorqueCompensator

RAD = math.pi / 180
SESSION_SHA = "c4a36fb2dce72b8ef0c3c4539a85f3a39ec4f6fe68ae3c023080f5f1a632467c"
SETTINGS_SHA = "7d9fcac94d6638ac135f203712676fb142f32f8813689b38033e3669786b4eb2"
GAINS_SHA = "28b361d295199e985c9f62dedd6525cbd70c8f065f6728d0d8fc8a9467c20d84"
URDF_SHA = "dca3429558024df1b249751787eef7f897c4426a623fe349a39fb2665fd166c0"


def checked_bytes(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeFault("Reviewed SDK source/config changed: " + str(path))
    return raw


def sdk_reference_gains(sdk_root):
    """Read all 17 SDK rows, including the waist's own 200/100 gains."""
    from .core import STATE_NAMES

    path = Path(sdk_root) / "vendor/robot_yiliang/robot_control_dds/controlParams_P2.json"
    rows = json.loads(checked_bytes(path, GAINS_SHA))[:17]
    expected = ("torso_joint",) + STATE_NAMES[:7] + STATE_NAMES[8:15] + STATE_NAMES[16:]
    if len(rows) != 17 or tuple(r["name"] for r in rows) != expected:
        raise RuntimeFault("SDK controlParams_P2 rows do not match the native 17-axis order")
    kp = np.asarray([r["kp"] for r in rows], dtype=float)
    kd = np.asarray([r["kd"] for r in rows], dtype=float)
    if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp <= 0) or np.any(kd < 0):
        raise RuntimeFault("Invalid SDK native gain table")
    return kp, kd


def load_sdk(sdk_root):
    root = Path(sdk_root)
    source = root / "tools/raise_arms_close_grippers.py"
    checked_bytes(source, SESSION_SHA)
    cfg = json.loads(checked_bytes(root / "configs/robot.0759.motion-reviewed.json", SETTINGS_SHA))
    rows = json.loads(
        checked_bytes(
            root / "vendor/robot_yiliang/robot_control_dds/controlParams_P2.json", GAINS_SHA
        )
    )
    if (cfg["robot_type"], cfg["robot_id"], cfg["mode"]) != ("ATOM W", "0759", 11):
        raise RuntimeFault("Robot identity or mode mismatch")
    if len(cfg["joints"]) != 17 or [j["index"] for j in cfg["joints"]] != list(range(17)):
        raise RuntimeFault("Unexpected upper joint order")
    gains = {r["name"]: r for r in rows}
    for i, joint in enumerate(cfg["joints"]):
        if i == 0:
            continue
        row = gains[joint["name"].replace("wrist_roll_joint", "wrist_yaw_joint")]
        expected = (300, 15) if "shoulder" in joint["name"] else (200, 8)
        if (row["kp"], row["kd"]) != expected:
            raise RuntimeFault("Reviewed P2 gain table changed")
        joint.update(kp=row["kp"], kd=row["kd"])
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("atom_production_sdk_dds", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from atom_training import Config
    from atom_training.transport import interface_ipv4

    transport_config = Config.load(root / "configs/robot.example.json")
    attribute, local_ip = interface_ipv4(transport_config.dds_interface)
    if (local_ip, transport_config.pc1_ip, transport_config.domain_id) != (
        "192.168.8.13",
        "192.168.8.234",
        0,
    ):
        raise RuntimeFault("DDS transport does not identify the reviewed PC2/PC1 pair")
    from xml.sax.saxutils import quoteattr

    os.environ["CYCLONEDDS_URI"] = (
        f'<CycloneDDS><Domain Id="0"><General><Interfaces>'
        f"<NetworkInterface {attribute}={quoteattr(transport_config.dds_interface)} "
        'multicast="true"/>'
        "</Interfaces><AllowMulticast>spdp</AllowMulticast><DontRoute>true</DontRoute>"
        "</General></Domain></CycloneDDS>"
    )
    cfg = apply_controller_limits(
        cfg,
        read_controller_settings(transport_config.pc1_ip),
        pc1_ip=transport_config.pc1_ip,
    )
    return module, cfg


def idle_pc1_publishers():
    """Discover once; admit only the known idle PC1 humanoid controller identity."""
    from atom_training.transport import _close_entity
    from cyclonedds.builtin import (
        BuiltinDataReader,
        BuiltinTopicDcpsParticipant,
        BuiltinTopicDcpsPublication,
    )
    from cyclonedds.domain import DomainParticipant

    participant = DomainParticipant(0)
    pub = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    readers = BuiltinDataReader(participant, BuiltinTopicDcpsParticipant)
    writers = set()
    properties = {}
    deadline = time.monotonic() + 2
    try:
        while time.monotonic() < deadline:
            for endpoint in pub.take(200):
                if endpoint.sample_info.valid_data and endpoint.topic_name == "rt/upper/cmd":
                    writers.add(str(endpoint.participant_key))
            for endpoint in readers.take(100):
                if endpoint.sample_info.valid_data:
                    properties[str(endpoint.key)] = {
                        v.key: v.value
                        for v in endpoint.qos
                        if hasattr(v, "key") and hasattr(v, "value")
                    }
            time.sleep(0.005)
        for key in writers:
            values = properties.get(key, {})
            if (
                values.get("__Hostname") != "atom"
                or values.get("__ProcessName") != "humanoid"
                or not values.get("__NetworkAddresses", "").startswith("udp/192.168.8.234:")
            ):
                raise RuntimeFault("Unreviewed upper publisher identity: " + str(values))
        return list(writers)
    finally:
        for item in (readers, pub, participant):
            _close_entity(item)


def load_last_holding(q, validation):
    """Reuse only a compatible holding record that still describes this pose.

    History is a cache, not a live position constraint. The caller has already
    checked stationary feedback and exclusive ownership before seeding support.
    """
    candidates = []
    for prefix in (
        "atom-runtime-",
        "full-recorded-pose-",
        "model-arm-step-",
        "home-start-",
        "home-recovery-step-",
    ):
        for folder in Path(validation).glob(prefix + "*"):
            path = folder / "result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result.get("commands_sent", 0):
                candidates.append((folder.name[len(prefix) :], folder, result))
    if not candidates:
        return None
    _, folder, result = max(candidates, key=lambda item: item[0])
    if not result.get("upper_control_returned"):
        raise RuntimeFault("Previous controller did not confirm handover")
    # Native SDK q references deliberately include PD/static offsets. They are
    # not physical position goals for the Home compensator, even when close to
    # feedback. Switching control laws must start from current measured support.
    if result.get("command_law", "compensated_position") != "compensated_position":
        return None
    ref = np.asarray(result.get("last_motor_reference"), dtype=float)
    tau = np.asarray(result.get("last_tau_ff", result.get("final_tau_ff")), dtype=float)
    if any(x.shape != (17,) or not np.isfinite(x).all() for x in (ref, tau)):
        raise RuntimeFault("Invalid stored holding state")
    if np.max(abs(ref - np.asarray(q))) >= 2 * RAD or abs(ref[0] - q[0]) > 0.05 * RAD:
        return None
    if np.max(abs(tau[[0, 15, 16]])) > 1e-9:
        raise RuntimeFault("Unexpected non-arm holding torque")
    ref[0] = q[0]
    return {"reference": ref, "tau_ff": tau, "source": str(folder)}


def sdk_effort_limits(urdf_root, joint_settings):
    """The packaged arm URDF omits the locked waist; do not invent an axis limit."""
    joints = {j.get("name"): j for j in urdf_root.findall("joint")}
    effort = np.full(17, np.inf)
    for i, joint in enumerate(joint_settings):
        if i == 0:
            continue
        name = joint["name"].replace("wrist_roll_joint", "wrist_yaw_joint")
        if name not in joints or joints[name].find("limit") is None:
            raise RuntimeFault("Missing reviewed actuator limit: " + name)
        effort[i] = float(joints[name].find("limit").get("effort"))
    if not np.isfinite(effort[1:]).all() or np.any(effort[1:] <= 0):
        raise RuntimeFault("Invalid arm/head effort limits")
    return effort


def release_owned_control(session, *, clock=time.monotonic, sleep=time.sleep):
    """Repeat only this session's SDK handover until fresh PC1 confirmation."""
    if not session.requested or session.released:
        return
    deadline = clock() + 2.0
    next_send = 0.0
    while clock() < deadline:
        now = clock()
        if now >= next_send:
            session.switch.write(session.dds.SwitchUpperControl_(flag=False))
            next_send = now + 0.05
        session.pump(exclusive=False)
        if (
            "upper" in session.last
            and clock() - session.stamp.get("upper", -float("inf")) < 0.12
            and not session.last["upper"].is_upper_control
        ):
            session.released = True
            return
        sleep(0.005)
    raise RuntimeFault("Upper ownership release was not confirmed by fresh PC1 feedback")


class NativeBridge:
    def __init__(
        self,
        sdk_root,
        validation,
        *,
        execute=False,
        allow_head_motion=False,
        fresh_hold=False,
        require_initialized_grippers=True,
        dataset_reference_test=False,
        command_law="compensated_position",
    ):
        self.sdk_root = Path(sdk_root)
        self.validation = Path(validation)
        self.execute = execute
        self.session = None
        self.settings = None
        self.closed = False
        self.command_count = 0
        self.peak_tracking = 0.0
        self.peak_tick_gap = 0.0
        self.reference = None
        self.tau_ff = None
        self.last_send = None
        self.ownership_acquired = False
        self.allow_head_motion = allow_head_motion
        self.fresh_hold = fresh_hold
        self.require_initialized_grippers = require_initialized_grippers
        self.dataset_reference_test = dataset_reference_test
        self.last_expected_feedback = None
        if command_law not in ("compensated_position", "sdk_reference"):
            raise ValueError("Unknown native motor command law")
        self.command_law = command_law

    def open(self):
        if self.session is not None or self.closed:
            raise RuntimeFault("DDS bridge lifecycle cannot be restarted")
        self.native, self.settings = load_sdk(self.sdk_root)
        if self.command_law == "sdk_reference":
            kp, kd = sdk_reference_gains(self.sdk_root)
            for joint, p, d in zip(self.settings["joints"], kp, kd, strict=True):
                joint.update(kp=float(p), kd=float(d))
        self.settings["allowed_idle_publishers"] = idle_pc1_publishers()
        session_type = self.native.Session
        if self.command_law == "sdk_reference":
            from .sdk_session import sdk_reference_session

            session_type = sdk_reference_session(session_type)
        self.session = session_type(self.settings)
        self.limits = np.asarray([j["limits"] for j in self.settings["joints"]], dtype=float)
        self.kp = np.asarray([j["kp"] for j in self.settings["joints"]], dtype=float)
        self.kd = np.asarray([j["kd"] for j in self.settings["joints"]], dtype=float)
        urdf = self.sdk_root / "vendor/robot_model/atom.urdf"
        checked_bytes(urdf, URDF_SHA)
        self.effort = sdk_effort_limits(ET.parse(urdf).getroot(), self.settings["joints"])
        return self

    def observe(self, seconds=3.0):
        s = self.session
        other = 0
        samples = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            other += len(s.readers["upper_cmd"].take(200))
            s.pump(exclusive=False)
            if "upper" in s.last:
                samples.append(
                    (
                        time.monotonic(),
                        s.q(),
                        [float(m.tau_est) for m in s.last["upper"].motor_state],
                    )
                )
            time.sleep(0.005)
        s.check(False)
        if other:
            raise RuntimeFault("Another program is sending upper commands")
        recent = [(q, tau) for stamp, q, tau in samples if stamp > time.monotonic() - 0.5]
        if (
            len(recent) < 10
            or np.max(np.ptp(np.asarray([q for q, _ in recent]), axis=0)) > 0.3 * RAD
        ):
            raise RuntimeFault("Robot is not stationary")
        self.stationary_tau_est = np.median(np.asarray([tau for _, tau in recent]), axis=0)
        s.base = s.q()
        s.hands.read()
        if not self.execute:
            self.reference = np.asarray(s.base)
            self.holding_source = "read_only_feedback; no holding effort is assumed"
            self.tau_ff = np.zeros(17)
            return self.snapshot()
        self.seed_holding(s.base, self.stationary_tau_est)
        return self.snapshot()

    def seed_holding(self, q, tau_est):
        """Seed only after stationary/exclusive observe; fresh mode keeps effort bounds."""
        if not self.execute:
            raise RuntimeFault("Read-only bridge cannot prepare motor holding state")
        q = finite_array(q, (17,), "holding feedback")
        if self.command_law == "sdk_reference":
            # q/dq/tau/Kp/Kd use the supplied SDK's native PD law. Convert the
            # measured static support once into an equivalent zero-tau startup
            # reference; policy rows themselves are subsequently sent unchanged.
            tau = finite_array(tau_est, (17,), "static support feedback")
            dq = np.asarray([float(m.dq) for m in self.session.last["upper"].motor_state])
            self.reference = q + (tau + self.kd * dq) / self.kp
            if np.any(self.reference < self.limits[:, 0]) or np.any(
                self.reference > self.limits[:, 1]
            ):
                raise RuntimeFault("Equivalent SDK startup reference exceeds controller limits")
            self.tau_ff = np.zeros(17)
            self.holding_source = "sdk_zero_tau_equivalent_static_reference"
            return
        self.comp = OuterTorqueCompensator(self.kp, self.effort, max_bias_deg=3.0)
        held = None if self.fresh_hold else load_last_holding(q, self.validation)
        if held:
            self.reference = held["reference"]
            self.comp.seed_latched(self.reference, q, held["tau_ff"])
            self.holding_source = held["source"]
        else:
            self.reference = np.array(q, dtype=float, copy=True)
            self.comp.seed_static(tau_est)
            self.holding_source = (
                "explicit_fresh_bounded_static_effort"
                if self.fresh_hold
                else "bounded_static_effort"
            )
        self.tau_ff = self.comp.current_tau()

    def snapshot(self):
        from atom_training.dh_pc1 import feedback_values

        s = self.session
        s.pump(exclusive=s.requested)
        s.check(True if self.ownership_acquired else False)
        s.hands._poll()
        ages = s.hands._ages()
        if max(ages) > 0.25:
            raise RuntimeFault("Hand feedback is stale")
        hands = feedback_values(s.hands._hands)
        if self.require_initialized_grippers and any(
            hands[side]["init_state"] != 1 for side in ("left", "right")
        ):
            raise RuntimeFault("Grippers must already be initialized")
        q = np.asarray(s.q())
        opening = np.asarray([hands[side]["opening"] for side in ("left", "right")])
        return {
            "time": time.monotonic(),
            "q": q,
            "openings": opening,
            "state": state_vector(q, opening),
            "tau_est": np.asarray([float(m.tau_est) for m in s.last["upper"].motor_state]),
            "hands": hands,
        }

    def acquire(self):
        if not self.execute:
            raise RuntimeFault("Read-only bridge cannot create command writers")
        assert_controller_limits_unchanged(self.settings, pc1_ip="192.168.8.234")
        s = self.session
        s.pump()
        s.check(False)
        s.prepare_writers()
        # First command preserves the exact known holding position and torque.
        self.send(self.reference, ownership=False)
        s.requested = True
        for _ in range(3):
            s.switch.write(s.dds.SwitchUpperControl_(flag=True))
        deadline = time.monotonic() + 2
        while True:
            self.send(self.reference, ownership=None)
            if s.last["upper"].is_upper_control:
                break
            if time.monotonic() > deadline:
                raise RuntimeFault("Upper authority was not granted")
            time.sleep(self.native.PERIOD)
        self.ownership_acquired = True

    def send(self, reference, ownership=True, *, expected_feedback=None):
        if self.command_law == "sdk_reference":
            if expected_feedback is not None:
                raise RuntimeFault("SDK motor rows do not accept a substituted feedback target")
            return self._send_sdk_reference(reference, ownership)
        if not self.execute:
            raise RuntimeFault("Read-only bridge cannot send motor commands")
        s = self.session
        s.pump()
        s.check(ownership)
        now = time.monotonic()
        reference = np.asarray(reference, dtype=float)
        q = np.asarray(s.q())
        if reference.shape != (17,) or not np.isfinite(reference).all():
            raise RuntimeFault("Invalid upper reference")
        if self.last_send is not None:
            gap = now - self.last_send
            self.peak_tick_gap = max(self.peak_tick_gap, gap)
            if gap > 0.05:
                raise RuntimeFault("Servo loop stalled over 50 ms")
        if s.matches["upper"] < 1:
            raise RuntimeFault("Upper command subscriber lost")
        if np.any(reference < self.limits[:, 0]) or np.any(reference > self.limits[:, 1]):
            raise RuntimeFault("Reference exceeds active PC1 position limits")
        if abs(reference[0] - s.base[0]) > 1e-9:
            raise RuntimeFault("Waist reference changed")
        if not self.allow_head_motion and np.max(abs(reference[15:] - self.reference[15:])) > 1e-9:
            raise RuntimeFault("Frozen head reference changed")
        if s.last_ref is not None and np.max(abs(reference - np.asarray(s.last_ref))) > 0.05 * RAD:
            raise RuntimeFault("Adjacent motor reference step exceeds .05 degree")
        tracking_reference = reference
        if expected_feedback is not None:
            if not self.dataset_reference_test:
                raise RuntimeFault(
                    "Recorded feedback tracking is available only in explicit dataset tests"
                )
            tracking_reference = finite_array(
                expected_feedback, (17,), "recorded physical feedback"
            )
            if np.any(tracking_reference < self.limits[:, 0]) or np.any(
                tracking_reference > self.limits[:, 1]
            ):
                raise RuntimeFault("Recorded physical feedback exceeds active PC1 position limits")
        error = float(np.max(abs(tracking_reference - q)))
        self.peak_tracking = max(self.peak_tracking, error)
        if error > 2 * RAD:
            raise RuntimeFault("Measured reference tracking error exceeds 2 degrees")
        dt = 0 if self.last_send is None else now - self.last_send
        tau = self.comp.step(reference, q, dt)
        if not np.isfinite(tau).all() or np.any(abs(tau) > self.effort + 1e-9):
            raise RuntimeFault("Holding torque exceeds SDK rated joint effort")
        dq = np.asarray([float(m.dq) for m in s.last["upper"].motor_state])
        predicted = self.kp * (reference - q) - self.kd * dq + tau
        if np.any(abs(predicted) > self.effort):
            raise RuntimeFault("Predicted joint effort exceeds SDK rated joint effort")
        motors = [
            s.dds.MotorCmd_(
                mode=11,
                q=float(v),
                dq=0.0,
                tau=float(tau[i]),
                kp=float(self.kp[i]),
                kd=float(self.kd[i]),
            )
            for i, v in enumerate(reference)
        ]
        s.own_sent = True
        s.last_ref = reference.tolist()
        s.writer.write(s.dds.UpperCmd_(motor_cmd=motors))
        self.last_expected_feedback = (
            tracking_reference.copy() if expected_feedback is not None else None
        )
        self.last_send = time.monotonic()
        self.command_count += 1
        self.tau_ff = tau

    def _send_sdk_reference(self, reference, ownership):
        if not self.execute:
            raise RuntimeFault("Read-only bridge cannot send motor commands")
        s = self.session
        s.pump()
        s.check(ownership)
        reference = finite_array(reference, (17,), "SDK upper reference")
        if np.any(reference < self.limits[:, 0]) or np.any(reference > self.limits[:, 1]):
            raise RuntimeFault("Reference exceeds active PC1 position limits")
        if abs(reference[0] - self.reference[0]) > 1e-9:
            raise RuntimeFault("Unmodelled waist reference changed")
        if not self.allow_head_motion and np.max(abs(reference[15:] - self.reference[15:])) > 1e-9:
            raise RuntimeFault("Frozen head reference changed")
        now = time.monotonic()
        if self.last_send is not None:
            gap = now - self.last_send
            self.peak_tick_gap = max(self.peak_tick_gap, gap)
            if gap > 0.05:
                raise RuntimeFault("SDK reference publisher stalled over one action period")
        if s.matches["upper"] < 1:
            raise RuntimeFault("Upper-command subscriber lost")
        # Exactly the supplied SDK's Control_sim.send_cmd motor fields. No
        # outer PI, extra feedforward, reference clipping or per-step cap.
        motors = [
            s.dds.MotorCmd_(
                mode=11, q=float(v), dq=0.0, tau=0.0, kp=float(self.kp[i]), kd=float(self.kd[i])
            )
            for i, v in enumerate(reference)
        ]
        s.own_sent = True
        s.last_ref = reference.tolist()
        s.writer.write(s.dds.UpperCmd_(motor_cmd=motors))
        self.peak_tracking = max(
            self.peak_tracking, float(np.max(abs(reference - np.asarray(s.q()))))
        )
        self.last_send = time.monotonic()
        self.command_count += 1
        self.tau_ff = np.zeros(17)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.session is not None:
            try:
                release_owned_control(self.session)
                if self.session.requested and self.session.released:
                    deadline = time.monotonic() + 0.5
                    while time.monotonic() < deadline:
                        self.session.verify_held_pose(
                            self.last_expected_feedback
                            if self.dataset_reference_test
                            and self.last_expected_feedback is not None
                            else self.session.last_ref
                        )
                        time.sleep(0.01)
            finally:
                self.session.close()

    def result(self):
        s = self.session
        upper = getattr(s, "last", {}).get("upper") if s else None
        control_active = None if upper is None else bool(upper.is_upper_control)
        return {
            "commands_sent": self.command_count,
            "upper_ownership_requested": bool(s and s.requested),
            "upper_control_returned": bool(
                s and (s.released or (not s.requested and control_active is False))
            ),
            "upper_control_release_confirmed": bool(s and s.released),
            "upper_control_observed": control_active,
            "last_motor_reference": s.last_ref if s and s.last_ref is not None else None,
            "last_tau_ff": None if self.tau_ff is None else self.tau_ff.tolist(),
            "max_tracking_error_deg": self.peak_tracking / RAD,
            "max_tick_gap_s": self.peak_tick_gap,
            "dds_session_initializations": int(s is not None),
            "motor_writer_created": bool(s and s.writer is not None),
            "holding_source": getattr(self, "holding_source", None),
            "dataset_reference_test": self.dataset_reference_test,
            "command_law": self.command_law,
            "controller_position_limits": (
                self.settings.get("controller_position_limits") if self.settings else None
            ),
        }
