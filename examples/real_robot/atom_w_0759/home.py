"""Smooth Home return at normal SDK reference speed, preserving physical goals."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from ..atom_common.core import PlaybackClock, RuntimeFault, Trajectory, finite_array
from ..atom_common.lease import hardware_lock_path
from ..atom_common.torque import OuterTorqueCompensator
from .native import NativeBridge

HOME_CONFIG = Path(__file__).parent / "config/atom0759_home.json"


def preflight_home(sdk_root, validation, *, home_config=HOME_CONFIG):
    """Check target and measured support without creating command writers."""
    bridge = NativeBridge(sdk_root, validation, execute=False, require_initialized_grippers=False)
    config = json.loads(Path(home_config).read_text())
    lock_path = hardware_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            bridge.open()
            snapshot = bridge.observe()
            physical, target = home_targets(snapshot["q"], config)
            if np.any(target < bridge.limits[:, 0]) or np.any(target > bridge.limits[:, 1]):
                raise RuntimeFault("Home motor target exceeds active PC1 position limits")
            comp = OuterTorqueCompensator(bridge.kp, bridge.effort, max_bias_deg=3.0)
            comp.seed_static(bridge.stationary_tau_est)
            return {
                "status": "read_only_preflight_passed",
                "execute": False,
                "initial_q_rad": snapshot["q"].tolist(),
                "target_q_rad": target.tolist(),
                "physical_target_q_rad": physical.tolist(),
                "delta_deg": np.rad2deg(physical - snapshot["q"]).tolist(),
                "static_support_tau_Nm": comp.current_tau().tolist(),
                "rated_effort_Nm": [float(v) if np.isfinite(v) else None for v in bridge.effort],
                "reference_speed_deg_s": config["reference_speed_deg_s"],
                "gripper_commands_sent": 0,
                "home_config": str(home_config),
                **bridge.result(),
            }
        finally:
            bridge.close()


def tracking_playback_scale(feedback, reference, *, head_pitch_bias_deg=0.0):
    """Slow the virtual clock smoothly before the unchanged 2-degree guard."""
    q = finite_array(feedback, (17,), "Home tracking feedback")
    ref = finite_array(reference, (17,), "Home tracking reference")
    if not np.isfinite(head_pitch_bias_deg):
        raise RuntimeFault("Invalid head reference bias")
    # Compare in physical joint coordinates; the motor reference deliberately
    # contains the separately validated head pitch bias.
    ref[16] -= np.deg2rad(head_pitch_bias_deg)
    error = float(np.max(abs(q - ref)))
    u = float(np.clip((error - np.deg2rad(0.5)) / np.deg2rad(0.7), 0, 1))
    return 1 - u**3 * (10 + u * (-15 + 6 * u))


def home_targets(q, config):
    if (
        config["schema_version"],
        config["robot_type"],
        config["robot_id"],
        config["joint_units"],
    ) != (1, "ATOM W", "0759", "rad"):
        raise RuntimeFault("Home configuration identifies a different robot")
    state = finite_array(config["physical_state"], (18,), "physical Home state")
    q = finite_array(q, (17,), "current upper pose")
    if config["waist"] != "hold" or config["grippers"] != "hold":
        raise RuntimeFault("Home must hold waist and grippers")
    if config["physical_tolerance_deg"] != 0.8 or not 0 < config["reference_speed_deg_s"] <= float(
        np.rad2deg(0.1)
    ):
        raise RuntimeFault("Home tolerance or speed differs from the reviewed envelope")
    physical = np.r_[q[0], state[:7], state[8:15], state[16:18]]
    motor = physical.copy()
    bias = float(config["head_pitch_motor_reference_bias_deg"])
    if not -0.6 <= bias <= 0:
        raise RuntimeFault("Head reference bias is outside the validated range")
    motor[16] += np.deg2rad(bias)
    return physical, motor


def return_home(
    sdk_root,
    validation,
    *,
    blocked=None,
    home_config=HOME_CONFIG,
    fresh_hold=False,
    reset_devices=False,
    gripper_sdk_root=None,
):
    # A confirmed unresolved contact must not be followed by an unverified retreat.
    if blocked is not None:
        return {
            "status": "held_after_obstruction",
            "commands_sent": 0,
            "reason": (
                "Obstruction has not been cleared; automatic Home could apply more contact force"
            ),
        }
    validation = Path(validation)
    validation.mkdir(parents=True, exist_ok=True)
    folder = validation / (
        "atom-runtime-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-home"
    )
    folder.mkdir()
    bridge = NativeBridge(
        sdk_root,
        validation,
        execute=True,
        allow_head_motion=True,
        fresh_hold=fresh_hold,
        require_initialized_grippers=False,
    )
    result = {
        "status": "preparing_home",
        "commands_sent": 0,
        "gripper_commands_sent": 0,
        "calibration_modified": False,
        "fresh_hold_requested": fresh_hold,
        "tracking_clock_delay_s": 0.0,
    }
    config = json.loads(Path(home_config).read_text())
    result["reference_speed_deg_s"] = config["reference_speed_deg_s"]
    interrupted = False
    previous_handlers = {}

    def stop(sig, frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[sig] = signal.signal(sig, stop)
    lock = hardware_lock_path().open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bridge.open()
        snapshot = bridge.observe()
        if reset_devices:
            from .gripper_reset import reset_grippers

            result["gripper_commands_sent"] = None  # unknown until the child report is read
            result["device_reset"] = reset_grippers(
                validation,
                folder,
                lock.fileno(),
                cancelled=lambda: interrupted,
                sdk_root=gripper_sdk_root,
            )
            result["gripper_commands_sent"] = result["device_reset"]["writes"]
            print("ATOM_HOME_DEVICE_RESET", json.dumps(result["device_reset"]), flush=True)
            snapshot = bridge.observe()
        if interrupted:
            raise RuntimeFault("Home interrupted before arm movement")
        physical, target = home_targets(snapshot["q"], config)
        if np.any(target < bridge.limits[:, 0]) or np.any(target > bridge.limits[:, 1]):
            raise RuntimeFault("Home motor target exceeds active PC1 position limits")
        bridge.acquire()
        reference = np.r_[bridge.reference, snapshot["openings"]]
        print("ATOM_HOME_RETURN_STARTED", json.dumps({"report_dir": str(folder)}), flush=True)
        for phase, indices in [
            ("left_arm", range(1, 8)),
            ("right_arm", range(8, 15)),
            ("head", range(15, 17)),
        ]:
            end = reference.copy()
            end[list(indices)] = target[list(indices)]
            if np.max(abs(end - reference)) < 1e-10:
                continue
            path = Trajectory(
                reference, end[None, :], joint_speed=np.deg2rad(config["reference_speed_deg_s"])
            )
            began = time.monotonic()
            clock = PlaybackClock(began, path.joint_speed)
            last_command = reference[:17].copy()
            while True:
                if interrupted:
                    raise RuntimeFault("Home interrupted by operator")
                feedback = bridge.snapshot()
                now = time.monotonic()
                scale = tracking_playback_scale(
                    feedback["q"],
                    last_command,
                    head_pitch_bias_deg=(
                        config["head_pitch_motor_reference_bias_deg"] if phase == "head" else 0.0
                    ),
                )
                command, complete = path.sample(clock.advance(now, scale=scale))
                bridge.send(command[:17])
                last_command = command[:17].copy()
                if complete:
                    break
                if now - began > max(60.0, path.duration * 3):
                    raise RuntimeFault("Home trajectory did not track within its bounded deadline")
                time.sleep(0.0012)
            result["tracking_clock_delay_s"] += clock.delayed
            reference = end
        deadline = time.monotonic() + 60
        stable = None
        while True:
            if interrupted:
                raise RuntimeFault("Home interrupted by operator")
            bridge.send(reference[:17])
            snapshot = bridge.snapshot()
            now = time.monotonic()
            error = np.rad2deg(snapshot["q"] - physical)
            stable = (
                (stable or now)
                if np.max(abs(error[1:])) <= config["physical_tolerance_deg"]
                else None
            )
            if stable is not None and now - stable > 0.5:
                break
            if now > deadline:
                raise RuntimeFault("Physical Home did not reach its unchanged tolerance")
            time.sleep(0.0012)
        result.update(
            status="physical_home_completed",
            full_pose_synchronized=True,
            physical_home_error_deg=error.tolist(),
            final_q_rad=snapshot["q"].tolist(),
            head_pitch_motor_reference_bias_deg=config["head_pitch_motor_reference_bias_deg"],
            home_config_sha256=hashlib.sha256(Path(home_config).read_bytes()).hexdigest(),
        )
    except BaseException as exc:
        result.update(status="home_fault", error=type(exc).__name__ + ": " + str(exc))
        if bridge.session is not None and "upper" in bridge.session.last:
            upper = bridge.session.last["upper"]
            result["feedback_at_fault"] = {
                "q_rad": [float(m.q) for m in upper.motor_state],
                "dq_rad_s": [float(m.dq) for m in upper.motor_state],
                "tau_est": [float(m.tau_est) for m in upper.motor_state],
                "fsm": int(upper.fsm_id),
            }
    finally:
        reset_report = folder / "gripper-reset.json"
        if reset_devices and reset_report.exists():
            try:
                result["device_reset"] = json.loads(reset_report.read_text())
                result["gripper_commands_sent"] = result["device_reset"]["writes"]
            except (OSError, ValueError, KeyError) as exc:
                result.update(status="home_fault", device_reset_report_error=str(exc))
        try:
            bridge.close()
        except BaseException as exc:
            result.update(status="home_fault", cleanup_error=str(exc))
        result.update(bridge.result())
        lock.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        path = folder / "result.tmp"
        path.write_text(json.dumps(result, indent=2) + "\n")
        path.replace(folder / "result.json")
    return dict(result, report_dir=str(folder))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--site-clear", action="store_true")
    parser.add_argument(
        "--reset-devices",
        action="store_true",
        help="Home/open both DH grippers and verify fresh ready feedback before arm Home",
    )
    parser.add_argument(
        "--fresh-hold",
        action="store_true",
        help="After restart/reposition, seed bounded effort from current stationary feedback",
    )
    parser.add_argument("--sdk-root", default=os.environ.get("ATOM_SDK_ROOT"))
    parser.add_argument("--gripper-sdk-root", default=os.environ.get("ATOM_GRIPPER_SDK_ROOT"))
    parser.add_argument(
        "--validation",
        default=str(Path(__file__).resolve().parents[3] / "runs/atom_robot/validation"),
    )
    args = parser.parse_args(argv)
    if not args.sdk_root:
        parser.error("Set ATOM_SDK_ROOT or pass --sdk-root")
    if args.execute and args.reset_devices and not args.gripper_sdk_root:
        parser.error("Device reset requires ATOM_GRIPPER_SDK_ROOT or --gripper-sdk-root")
    if args.execute and not args.site_clear:
        parser.error("Home movement requires explicit execution and on-site clearance")
    if args.execute:
        result = return_home(
            args.sdk_root,
            args.validation,
            fresh_hold=args.fresh_hold or args.reset_devices,
            reset_devices=args.reset_devices,
            gripper_sdk_root=args.gripper_sdk_root,
        )
    else:
        result = preflight_home(args.sdk_root, args.validation)
    print("ATOM_HOME_RETURN_RESULT", json.dumps(result), flush=True)
    return 0 if result["status"] in ("physical_home_completed", "read_only_preflight_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
