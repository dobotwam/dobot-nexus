"""Long-lived ATOM inference client. Default: real sensors, zero command writers."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import multiprocessing as mp
import os
import signal
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

from .core import (
    ActionChunk,
    BlockedDirectionGuard,
    ChunkGate,
    FixedRateChunk,
    LoopGuard,
    RuntimeFault,
    actuator_targets,
    require_fresh,
)
from .kinematics import JointBlockageMonitor
from .lease import hardware_lock_path
from .native import NativeBridge
from .policy import policy_identity, preflight_policy
from .sensors import DEFAULT_CAMERA_CONFIG, load_camera_specs
from .workers import gripper_worker, offer_latest, perception_worker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION = ROOT / "runs/atom_robot/validation"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Telemetry:
    """Disk writes never run in the servo loop; failure stops an observable run."""

    def __init__(self, folder, stop):
        self.folder = folder
        self.stop = stop
        self.queue = Queue(maxsize=32)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, kind, data):
        if self.error:
            raise RuntimeFault("Telemetry writer failed: " + str(self.error))
        try:
            self.queue.put_nowait((kind, data))
        except Full:
            raise RuntimeFault("Telemetry queue is full; refusing to stall servo") from None

    def _run(self):
        try:
            with (self.folder / "events.jsonl").open("a", buffering=1) as log:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    kind, data = item
                    entry = {"kind": kind, "monotonic": time.monotonic(), **data}
                    text = json.dumps(entry, default=json_default)
                    log.write(text + "\n")
                    if kind == "status":
                        path = self.folder / "status.tmp"
                        path.write_text(text + "\n")
                        path.replace(self.folder / "status.json")
        except BaseException as exc:
            self.error = exc
            self.stop.set()

    def close(self):
        self.queue.put(None, timeout=2)
        self.thread.join(timeout=3)
        if self.thread.is_alive() or self.error:
            raise RuntimeFault("Telemetry did not close cleanly: " + str(self.error))


def config_for(args, folder):
    return {
        "sdk_root": args.sdk_root,
        "execute": args.execute,
        "task_guard_mode": args.task_guards,
        "horizon": args.horizon,
        "action_fps": args.action_fps,
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "policy_timeout_s": 2.0,
        "startup_timeout_s": 20.0,
        "camera_max_age_s": 0.35,
        "camera_max_skew_s": 0.15,
        "task": args.task,
        "head_mode": args.head_mode,
        "cameras": str(Path(args.cameras).expanduser().resolve()),
        "record_observation": True,
        "report_dir": str(folder),
    }


def runtime_identity(args):
    from dataclasses import asdict

    return {
        "sdk_root": str(Path(args.sdk_root).expanduser().resolve()) if args.sdk_root else None,
        "gripper_sdk_root": str(Path(args.gripper_sdk_root).expanduser().resolve())
        if args.gripper_sdk_root
        else None,
        "cameras": [asdict(spec) for spec in load_camera_specs(args.cameras)],
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "head_mode": args.head_mode,
        "task": args.task,
        "action_fps": args.action_fps,
        "task_guards": args.task_guards,
    }


def run(args):
    from .validation import source_fingerprint

    if not args.sdk_root:
        raise RuntimeFault("Set ATOM_SDK_ROOT or pass --sdk-root for the reviewed ATOM SDK")
    args.sdk_root = str(Path(args.sdk_root).expanduser().resolve())
    identity = runtime_identity(args)
    sources = source_fingerprint()
    if args.execute:
        from .validation import require_live_validation

        metadata, horizon = preflight_policy(config_for(args, None))
        require_live_validation(
            args.validation,
            expected_horizon=horizon,
            expected_action_fps=args.action_fps,
            expected_runtime=identity,
            expected_policy=policy_identity(metadata),
        )
        if args.home_on_start:
            from .home import return_home

            if not args.gripper_sdk_root:
                raise RuntimeFault(
                    "Initial Home requires ATOM_GRIPPER_SDK_ROOT or --gripper-sdk-root"
                )
            home = return_home(
                args.sdk_root,
                Path(args.validation),
                fresh_hold=True,
                reset_devices=True,
                gripper_sdk_root=args.gripper_sdk_root,
            )
            print("ATOM_RUNTIME_INITIAL_HOME", json.dumps(home), flush=True)
            if home.get("status") != "physical_home_completed":
                raise RuntimeFault("Initial Home prerequisite failed; policy was not started")
    validation = Path(args.validation)
    validation.mkdir(parents=True, exist_ok=True)
    folder = validation / ("atom-runtime-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir()
    context = mp.get_context("spawn")
    stop = context.Event()
    request = context.Event()
    armed = context.Event()
    minimum_source = context.Value("d", 0.0, lock=False)
    camera_health = context.Array("d", 3, lock=False)
    grip_health = context.Value("d", 0.0, lock=False)
    state_queue = context.Queue(maxsize=1)
    chunk_queue = context.Queue(maxsize=1)
    target_queue = context.Queue(maxsize=16)
    events = context.Queue(maxsize=128)
    cfg = config_for(args, folder)
    processes = []
    bridge = None
    telemetry = Telemetry(folder, stop)
    result = {
        "status": "preparing",
        "sources": sources,
        "runtime_identity": identity,
        "execute": args.execute,
        "task_guard_mode": args.task_guards,
        "horizon": None,
        "consumed_actions": 0,
        "completed_horizons": 0,
        "trajectory_clock_delay_s": 0.0,
        "command_law": "sdk_reference",
        "action_fps": args.action_fps,
        "dispatched_action_rows": 0,
        "maximum_row_dispatch_lateness_s": 0.0,
        "commands_sent": 0,
        "gripper_commands_sent": 0,
        "home_required": bool(args.execute),
        "table_plane_calibrated": False,
        "stop_reason": None,
        "worker_events": [],
    }

    def interrupted(sig, frame):
        result["stop_reason"] = "operator_signal_" + str(sig)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    code = 2
    began = time.monotonic()
    last_snapshot = None
    blocked = None
    blocked_side = None
    last_reference = None
    gate = ChunkGate()
    loop_guard = LoopGuard(repetitions=args.loop_repetitions)
    obstruction = BlockedDirectionGuard(minimum_push_m=np.deg2rad(0.3), dimension=7)
    block_monitor = JointBlockageMonitor()
    reported_blocked_sides = set()
    trajectory = None
    active_sequence = None
    previous_consumed = 0
    pending = None
    requested = False
    ready = set()
    last_publish = 0.0
    last_metrics = 0.0
    last_grip = 0.0
    last_monitor = 0.0
    sample_count = 0
    previous_tick = None
    maximum_tick_gap = 0.0
    periods = []
    previous_reference = None
    maximum_reference_step = 0.0
    runtime_horizon = None
    trajectory_started = None
    active_row = -1
    finishing_since = None
    pending_completion = None
    last_gripper_ack = None
    cancelled_sequence = None

    def drain_events():
        nonlocal runtime_horizon, last_gripper_ack
        while True:
            try:
                event = events.get_nowait()
            except Empty:
                break
            result["worker_events"].append(event)
            if len(result["worker_events"]) > 64:
                result["worker_events"] = result["worker_events"][-64:]
            telemetry.put("worker", event)
            if event["kind"] == "gripper_row":
                last_gripper_ack = (event["sequence"], event["row"])
            if event["kind"] == "ready":
                ready.add(event["component"])
                if event["component"] == "perception":
                    result["server_metadata"] = event["metadata"]
                    runtime_horizon = event["horizon"]
                    result["horizon"] = runtime_horizon
            if event["kind"] == "fault":
                result["stop_reason"] = event["component"] + ": " + event["error"]
                stop.set()

    def publish_state(snapshot, now):
        nonlocal last_publish, sample_count
        if now - last_publish >= 0.04:
            sample_count += 1
            offer_latest(
                state_queue,
                {"time": snapshot["time"], "state": snapshot["state"], "sequence": sample_count},
            )
            last_publish = now

    def persist():
        path = folder / "result.tmp"
        path.write_text(json.dumps(result, default=json_default, indent=2) + "\n")
        path.replace(folder / "result.json")

    lock = hardware_lock_path().open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for target, arguments in [
            (
                perception_worker,
                (
                    cfg,
                    state_queue,
                    chunk_queue,
                    request,
                    stop,
                    camera_health,
                    events,
                    minimum_source,
                ),
            ),
            (gripper_worker, (cfg, target_queue, stop, armed, grip_health, events)),
        ]:
            process = context.Process(target=target, args=arguments, daemon=False)
            process.start()
            processes.append(process)
        bridge = NativeBridge(
            args.sdk_root,
            validation,
            execute=args.execute,
            allow_head_motion=args.head_mode == "policy",
            command_law="sdk_reference",
        )
        bridge.open()
        last_snapshot = bridge.observe()
        last_reference = np.r_[bridge.reference, last_snapshot["openings"]]
        startup_deadline = time.monotonic() + 30
        request.set()
        requested = True
        while True:
            now = time.monotonic()
            drain_events()
            if stop.is_set():
                raise RuntimeFault(result["stop_reason"] or "Worker stopped during preparation")
            last_snapshot = bridge.snapshot()
            publish_state(last_snapshot, now)
            try:
                pending = chunk_queue.get_nowait()
            except Empty:
                pass
            if {"perception", "grippers"} <= ready and pending is not None:
                break
            if now > startup_deadline:
                raise RuntimeFault("Runtime preparation timeout")
            time.sleep(0.0012)
        # Validate first full path before creating any motor command writer.
        if runtime_horizon is None:
            raise RuntimeFault("Policy did not negotiate its action horizon")
        if args.execute:
            require_live_validation(
                args.validation,
                expected_horizon=runtime_horizon,
                expected_action_fps=args.action_fps,
                expected_runtime=identity,
                expected_policy=policy_identity(result["server_metadata"]),
            )
        first = ActionChunk.create(
            pending["sequence"],
            pending["observation_time"],
            pending["received_time"],
            pending["actions"],
            horizon=runtime_horizon,
        )
        actuator_targets(first.actions, bridge.reference, bridge.limits, head_mode=args.head_mode)
        if args.execute:
            bridge.acquire()
            armed.set()
        result["status"] = "running"
        telemetry.put("status", dict(result, report_dir=str(folder)))
        print(
            "ATOM_RUNTIME_STARTED",
            json.dumps(
                {"execute": args.execute, "report_dir": str(folder), "horizon": runtime_horizon}
            ),
            flush=True,
        )
        waiting_since = None
        started = time.monotonic()
        finishing = False
        while not stop.is_set():
            tick = time.monotonic()
            row_event = None
            if previous_tick is not None:
                interval = tick - previous_tick
                maximum_tick_gap = max(maximum_tick_gap, interval)
                if len(periods) < 200000:
                    periods.append(interval)
                if args.execute and interval > 0.05:
                    raise RuntimeFault("Control scheduler stalled over 50 ms")
            previous_tick = tick
            drain_events()
            if stop.is_set():
                break
            for process in processes:
                if not process.is_alive():
                    raise RuntimeFault("Required worker exited unexpectedly")
            last_snapshot = bridge.snapshot()
            publish_state(last_snapshot, tick)
            for n, stamp in enumerate(camera_health):
                require_fresh(stamp, time.monotonic(), 0.35, "camera " + str(n))
            require_fresh(grip_health.value, time.monotonic(), 0.35, "gripper worker")
            if pending_completion is not None:
                sequence, final_row, completed_at = pending_completion
                if last_gripper_ack == (sequence, final_row):
                    result["completed_horizons"] += 1
                    minimum_source.value = tick
                    telemetry.put(
                        "chunk_completed",
                        {
                            "sequence": sequence,
                            "motor_completed_at": completed_at,
                            "gripper_final_row": final_row,
                            "observation_floor": tick,
                        },
                    )
                    pending_completion = None
                    if (
                        result["completed_horizons"] >= args.max_horizons
                        or tick - started >= args.max_seconds
                    ):
                        result["stop_reason"] = "episode_limit_reached"
                        finishing = True
                        finishing_since = tick
                elif tick - completed_at > 0.2:
                    raise RuntimeFault(
                        "Final gripper action row was not acknowledged; next inference refused"
                    )
            if pending is None:
                try:
                    pending = chunk_queue.get_nowait()
                except Empty:
                    pass
            if trajectory is None and pending is not None and not finishing:
                chunk = ActionChunk.create(
                    pending["sequence"],
                    pending["observation_time"],
                    pending["received_time"],
                    pending["actions"],
                    horizon=runtime_horizon,
                )
                if blocked is not None and chunk.observation_time < minimum_source.value:
                    result["discarded_pre_obstruction_predictions"] = (
                        result.get("discarded_pre_obstruction_predictions", 0) + 1
                    )
                    pending = None
                    request.set()
                    requested = True
                    continue
                gate.accept(chunk, tick)
                points = actuator_targets(
                    chunk.actions, bridge.reference, bridge.limits, head_mode=args.head_mode
                )
                repeating = loop_guard.observe(chunk.actions, last_snapshot["state"])
                if repeating:
                    result["loop_guard_triggered"] = True
                    if args.execute and args.task_guards == "stop":
                        result["stop_reason"] = "repeated_action_cycle"
                        raise RuntimeFault("Repeated action cycle with no measured progress")
                    telemetry.put("loop_monitor", {"sequence": chunk.sequence, "enforced": False})
                if blocked is not None:
                    indices = slice(1, 8) if blocked_side == "left" else slice(8, 15)
                    if obstruction.next_horizon_pushes_into_obstruction(
                        chunk.sequence, points[:, indices]
                    ):
                        raise RuntimeFault("Next horizon continues into the measured obstruction")
                    blocked = None
                    minimum_source.value = 0.0
                trajectory = FixedRateChunk(points, fps=args.action_fps)
                trajectory_started = tick
                active_row = -1
                active_sequence = chunk.sequence
                previous_consumed = 0
                waiting_since = None
                requested = False
                telemetry.put(
                    "chunk",
                    {
                        "sequence": chunk.sequence,
                        "actions": chunk.actions,
                        "observation_time": chunk.observation_time,
                        "inference_seconds": pending["inference_seconds"],
                        "trajectory_seconds": trajectory.duration,
                        "state": pending["state"],
                        "dispatch_state": last_snapshot["state"],
                        "source_state_sequence": pending["source_state_sequence"],
                        "camera_frames": pending["camera_frames"],
                        "policy_seed": pending.get("policy_seed"),
                        "policy_timing": pending.get("policy_timing", {}),
                        "observation_floor": pending.get("observation_floor", 0.0),
                        "observation_archive": pending.get("observation_archive"),
                        "loop_guard_would_stop": repeating,
                    },
                )
                pending = None
            if trajectory is not None:
                elapsed = tick - trajectory_started
                row = trajectory.frame_index(elapsed)
                if row != active_row:
                    if row != active_row + 1:
                        raise RuntimeFault(
                            "Missed action row; refusing to skip or accelerate playback"
                        )
                    active_row = row
                    row_event = {
                        "sequence": active_sequence,
                        "row": row,
                        "deadline": trajectory_started + row / args.action_fps,
                    }
                reference, consumed = trajectory.sample(elapsed)
                result["consumed_actions"] += consumed - previous_consumed
                previous_consumed = consumed
                last_reference = reference
                # Acquire the next observation only after this chunk finishes.
                # Predictions from a moving, earlier pose are not time-aligned
                # with the next chunk's first reference.
                if consumed == runtime_horizon:
                    trajectory = None
                    pending_completion = (active_sequence, runtime_horizon - 1, tick)
            elif not finishing:
                waiting_since = waiting_since or tick
                if blocked is not None:
                    # Reduce contact reference error without another push. Head
                    # and torso stay fixed; every reference step remains bounded.
                    actual = last_snapshot["q"]
                    last_reference[1:15] += np.clip(
                        actual[1:15] - last_reference[1:15], -0.1 * 0.0012, 0.1 * 0.0012
                    )
                if not requested and pending_completion is None:
                    request.set()
                    requested = True
                if tick - waiting_since > 2.0:
                    raise RuntimeFault("Action buffer underrun or obstruction-confirmation timeout")
            if stop.is_set():
                break
            if previous_reference is not None:
                maximum_reference_step = max(
                    maximum_reference_step,
                    float(np.max(abs(last_reference[:17] - previous_reference))),
                )
            previous_reference = last_reference[:17].copy()
            if args.execute:
                bridge.send(last_reference[:17])
            if row_event is not None:
                sent_at = time.monotonic()
                lateness = sent_at - row_event["deadline"]
                if lateness >= 1 / args.action_fps:
                    raise RuntimeFault("Action row missed its dataset period")
                result["dispatched_action_rows"] += 1
                result["maximum_row_dispatch_lateness_s"] = max(
                    result["maximum_row_dispatch_lateness_s"], lateness
                )
                telemetry.put(
                    "action_row",
                    dict(
                        row_event,
                        sent_at=sent_at,
                        reference=last_reference[:17].copy(),
                        openings=last_reference[17:].copy(),
                        feedback_q=last_snapshot["q"].copy(),
                        execute=args.execute,
                    ),
                )
            if row_event is not None or (trajectory is None and tick - last_grip >= 0.1):
                packet = dict(
                    row_event or {"sequence": active_sequence, "row": active_row, "deadline": tick},
                    time=tick,
                    opening=last_reference[17:].copy(),
                    is_hold=row_event is None,
                    horizon=runtime_horizon,
                )
                if row_event is not None and active_row == 0 and cancelled_sequence is not None:
                    packet["cancelled_previous_sequence"] = cancelled_sequence
                    cancelled_sequence = None
                try:
                    target_queue.put_nowait(packet)
                except Full:
                    raise RuntimeFault(
                        "Gripper row queue overflow; no action row was replaced"
                    ) from None
                last_grip = tick
            if args.execute and blocked is None and tick - last_monitor >= 0.05:
                for side, indices in [("left", slice(1, 8)), ("right", slice(8, 15))]:
                    evidence = block_monitor.observe(
                        tick,
                        side,
                        last_snapshot["q"][indices],
                        last_reference[indices],
                        float(
                            np.max(abs(last_snapshot["tau_est"][indices]) / bridge.effort[indices])
                        ),
                    )
                    if evidence:
                        if side not in reported_blocked_sides:
                            telemetry.put(
                                "obstruction_monitor",
                                dict(evidence, enforced=args.task_guards == "stop"),
                            )
                            reported_blocked_sides.add(side)
                        if args.task_guards == "monitor":
                            # On-site operator supervises task behavior. Native
                            # drive, effort, freshness and position gates remain.
                            continue
                        blocked = evidence
                        blocked_side = side
                        obstruction.record_blockage(
                            active_sequence, evidence["feedback_q_rad"], evidence["reference_q_rad"]
                        )
                        minimum_source.value = tick
                        cancelled_sequence = active_sequence
                        pending_completion = None
                        trajectory = None
                        pending = None
                        waiting_since = tick
                        request.set()
                        requested = True
                        telemetry.put("obstruction", evidence)
                        break
                    else:
                        reported_blocked_sides.discard(side)
                last_monitor = tick
            if finishing:
                if not args.execute:
                    break
                # The SDK consumes motor references, not ideal measured poses.
                # Report static tracking offsets instead of adding a second
                # controller to force measured positions onto training actions.
                if tick - finishing_since >= 0.5:
                    break
            if tick - last_metrics >= 1.0:
                result.update(
                    elapsed_seconds=tick - started,
                    state_sequence=sample_count,
                    maximum_scheduler_gap_s=maximum_tick_gap,
                    **bridge.result(),
                )
                telemetry.put("status", dict(result, report_dir=str(folder)))
                last_metrics = tick
            if tick - started > args.max_seconds + 30:
                raise RuntimeFault("Episode end deadline exceeded")
            time.sleep(max(0.0, 0.0012 - (time.monotonic() - tick)))
        if stop.is_set() and not str(result.get("stop_reason")).startswith("operator_signal"):
            raise RuntimeFault(result.get("stop_reason") or "A required worker stopped")
        result["status"] = "stopped" if stop.is_set() else "completed"
        code = 0
    except BaseException as exc:
        result.update(status="fault", error=type(exc).__name__ + ": " + str(exc))
        code = 2
    finally:
        armed.clear()
        stop.set()
        if bridge is not None:
            try:
                bridge.close()
            except BaseException as exc:
                result.update(status="fault", cleanup_error=str(exc))
                code = 2
            result.update(bridge.result())
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                result["forced_worker_shutdown"] = True
                code = 2
        try:
            drain_events()
        except Exception as exc:
            result["event_drain_error"] = str(exc)
            code = 2
        result["worker_exit_codes"] = [p.exitcode for p in processes]
        faults = [event for event in result["worker_events"] if event["kind"] == "fault"]
        if faults:
            result.update(status="fault", worker_faults=faults)
            code = 2
        result["gripper_commands_sent"] = sum(
            event.get("writes", 0)
            for event in result["worker_events"]
            if event["kind"] == "closed" and event["component"] == "grippers"
        )
        result["gripper_action_rows_consumed"] = sum(
            event.get("action_rows", 0)
            for event in result["worker_events"]
            if event["kind"] == "closed" and event["component"] == "grippers"
        )
        result["maximum_gripper_row_lateness_s"] = max(
            (
                event.get("maximum_row_lateness_s", 0.0)
                for event in result["worker_events"]
                if event["kind"] == "closed" and event["component"] == "grippers"
            ),
            default=0.0,
        )
        result["maximum_gripper_changed_target_lateness_s"] = max(
            (
                event.get("maximum_changed_target_lateness_s", 0.0)
                for event in result["worker_events"]
                if event["kind"] == "closed" and event["component"] == "grippers"
            ),
            default=0.0,
        )
        result["last_gripper_acknowledged_row"] = last_gripper_ack
        result["maximum_scheduler_gap_s"] = maximum_tick_gap
        result["maximum_shadow_reference_step_deg"] = float(np.rad2deg(maximum_reference_step))
        if periods:
            result["scheduler_interval_p99_s"] = float(np.percentile(periods, 99))
        result["last_state"] = last_snapshot
        result["elapsed_seconds"] = time.monotonic() - began
        try:
            telemetry.put("status", dict(result, report_dir=str(folder)))
            telemetry.close()
        except BaseException as exc:
            result.update(status="fault", telemetry_cleanup_error=str(exc))
            code = 2
        persist()
        lock.close()
        for queue in (state_queue, chunk_queue, target_queue, events):
            queue.cancel_join_thread()
            queue.close()
        print(
            "ATOM_RUNTIME_RESULT",
            json.dumps(dict(result, report_dir=str(folder)), default=json_default),
            flush=True,
        )
    if (
        args.execute
        and args.return_home
        and result.get("status") == "completed"
        and result.get("stop_reason") == "episode_limit_reached"
        and result.get("upper_control_returned")
        and result.get("commands_sent", 0) > 0
    ):
        from .home import return_home

        home = return_home(args.sdk_root, validation, blocked=blocked)
        result["home_result"] = home
        if home.get("status") != "physical_home_completed":
            code = 2
        persist()
        print("ATOM_RUNTIME_HOME_RESULT", json.dumps(home, default=json_default), flush=True)
    return code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--site-clear", action="store_true")
    parser.add_argument("--sdk-root", default=os.environ.get("ATOM_SDK_ROOT"))
    parser.add_argument("--gripper-sdk-root", default=os.environ.get("ATOM_GRIPPER_SDK_ROOT"))
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--cameras", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument(
        "--head-mode",
        choices=("hold", "policy"),
        default="hold",
        help="Hold the initial head reference, or execute model head outputs",
    )
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--max-horizons", type=int, default=6)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Require this per-inference action horizon; default negotiates with the policy",
    )
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument(
        "--action-fps",
        type=float,
        default=20.0,
        help="Dataset reference cadence; collect_data was recorded at 20 Hz",
    )
    parser.add_argument("--loop-repetitions", type=int, default=3)
    parser.add_argument(
        "--task-guards",
        choices=("stop", "monitor"),
        default="monitor",
        help="stop on repeated/blocked task behavior, or only log it under on-site supervision",
    )
    parser.add_argument("--task", default="Fold the T-shirt with both arms.")
    parser.add_argument("--return-home", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--home-on-start", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.execute and not args.site_clear:
        parser.error("Execution requires --site-clear after on-site confirmation")
    if (
        not 1 <= args.max_horizons <= 100
        or (args.horizon is not None and not 1 <= args.horizon <= 256)
        or not 0 < args.max_seconds <= 1800
        or not 2 <= args.loop_repetitions <= 10
        or args.action_fps != 20.0
        or not 1 <= args.policy_port <= 65535
    ):
        parser.error("Invalid episode or loop-stop limits")
    return args


def main(argv=None):
    try:
        return run(parse_args(argv))
    except RuntimeFault as exc:
        print(
            "ATOM_RUNTIME_REFUSED",
            # A failed initial Home may already have sent commands. Only the
            # per-phase reports can establish a command count.
            json.dumps({"status": "refused", "error": str(exc), "commands_sent": None}),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
