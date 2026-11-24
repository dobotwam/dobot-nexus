"""Blocking camera, policy and DH I/O live outside the DDS servo process."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from queue import Empty, Full

import numpy as np

from ..atom_common.core import RuntimeFault, negotiate_horizon, require_fresh
from ..atom_common.sensors import ATOM_CAMERAS, CameraRig, load_camera_specs


def offer_latest(queue, value):
    """Nonblocking replacement for observations/metrics; never use for active actions."""
    try:
        queue.put_nowait(value)
        return True
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            return False
        try:
            queue.put_nowait(value)
            return True
        except Full:
            return False


def drain_latest(queue, timeout=0.2):
    value = queue.get(timeout=timeout)
    while True:
        try:
            value = queue.get_nowait()
        except Empty:
            return value


def worker_error(events, stop, component, exc):
    try:
        events.put_nowait(
            {
                "kind": "fault",
                "component": component,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    except Full:
        pass
    stop.set()


def fresh_observation(state_queue, rig, config, minimum_source_time, stop):
    """Every state and image must follow the acknowledged previous chunk."""
    deadline = time.monotonic() + 1
    while not stop.is_set():
        observation = drain_latest(state_queue, timeout=0.5)
        require_fresh(observation["time"], time.monotonic(), 0.2, "robot observation")
        images, frame_info = rig.snapshot(config["camera_max_age_s"], config["camera_max_skew_s"])
        source_time = min(
            observation["time"], *(v["received_monotonic"] for v in frame_info.values())
        )
        if source_time >= minimum_source_time.value:
            return observation, images, frame_info, source_time
        if time.monotonic() > deadline:
            raise RuntimeFault(
                "No fresh observation after the completed chunk or obstruction pause"
            )
        time.sleep(0.01)
    raise RuntimeFault("Observation acquisition interrupted")


def archive_policy_exchange(folder, sequence, packet, actions, details):
    """Keep the exact input values and output for every request, using no pickle."""
    folder = Path(folder)
    records = folder / "observations"
    records.mkdir(exist_ok=True)
    path = records / f"{sequence:06d}.npz"
    # Preserve the legacy HWC archive format; conversion from the actual CHW
    # wire packet is lossless and is recorded explicitly in the JSON sidecar.
    images = {
        name: np.ascontiguousarray(value.transpose(1, 2, 0))
        for name, value in packet["images"].items()
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        state=packet["state"],
        actions=actions,
        prompt=np.asarray([packet["prompt"]]),
        **{"image." + name: value for name, value in images.items()},
    )
    temporary.replace(path)
    manifest = {
        "schema_version": 1,
        "sequence": sequence,
        "image_layout": "HWC_RGB",
        "wire_image_layout": "CHW_RGB",
        **details,
        "state_sha256": hashlib.sha256(packet["state"].tobytes()).hexdigest(),
        "wire_image_sha256": {
            k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in packet["images"].items()
        },
    }
    sidecar = path.with_suffix(".json")
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(sidecar)
    latest = folder / "latest-observation.tmp.npz"
    if latest.exists():
        latest.unlink()
    os.link(path, latest)
    latest.replace(folder / "latest-observation.npz")
    return str(path)


def perception_worker(
    config, state_queue, chunk_queue, request, stop, camera_heartbeat, events, minimum_source_time
):
    """One camera rig and one WebSocket for the entire runtime lifecycle."""
    rig = None
    sequence = 0
    connection_count = 0
    try:
        from dobot_nexus.inference.transport.client import WebsocketPolicyClient

        from ..atom_common.policy import policy_contract

        specs = load_camera_specs(config["cameras"]) if config.get("cameras") else ATOM_CAMERAS
        index = {spec.name: n for n, spec in enumerate(specs)}
        rig = CameraRig(
            specs, heartbeat=lambda name, stamp: camera_heartbeat.__setitem__(index[name], stamp)
        )
        rig.start()
        rig.wait_ready(timeout=config["startup_timeout_s"])
        with WebsocketPolicyClient(
            host=config["policy_host"],
            port=config["policy_port"],
            timeout=config["policy_timeout_s"],
        ) as client:
            connection_count += 1
            metadata = client.get_server_metadata()
            contract, horizon = policy_contract(metadata, config.get("horizon"))
            layout = contract.layout
            events.put(
                {
                    "kind": "ready",
                    "component": "perception",
                    "metadata": metadata,
                    "horizon": horizon,
                    "connection_count": connection_count,
                }
            )
            while not stop.is_set():
                if not request.wait(timeout=0.1):
                    continue
                request.clear()
                if stop.is_set():
                    break
                observation, images, frame_info, source_time = fresh_observation(
                    state_queue, rig, config, minimum_source_time, stop
                )
                packet = contract.observation(
                    {
                        "state": observation["state"],
                        "state_names": list(layout.state_names),
                        "images": images,
                    },
                    config["task"],
                )
                began = time.monotonic()
                reply = client.infer(packet)
                elapsed = time.monotonic() - began
                actions = contract.actions(reply["actions"], horizon)
                if stop.is_set():
                    break
                sequence += 1
                payload = {
                    "sequence": sequence,
                    "observation_time": source_time,
                    "received_time": time.monotonic(),
                    "actions": actions,
                    "state": observation["state"],
                    "inference_seconds": elapsed,
                    "source_state_sequence": observation["sequence"],
                    "camera_frames": frame_info,
                    "policy_seed": reply.get("seed"),
                    "policy_timing": reply.get("policy_timing", {}),
                    "observation_floor": minimum_source_time.value,
                }
                if config.get("record_observation"):
                    payload["observation_archive"] = archive_policy_exchange(
                        config["report_dir"],
                        sequence,
                        packet,
                        actions,
                        {
                            "state_time": observation["time"],
                            "source_state_sequence": observation["sequence"],
                            "observation_time": source_time,
                            "observation_floor": minimum_source_time.value,
                            "camera_frames": frame_info,
                            "policy_seed": reply.get("seed"),
                            "policy_timing": reply.get("policy_timing", {}),
                        },
                    )
                # One pending chunk is allowed. The control process never blocks
                # on inference, and no old prediction can silently replace an active path.
                chunk_queue.put(payload, timeout=0.5)
                try:
                    events.put_nowait(
                        {
                            "kind": "inference",
                            "component": "perception",
                            "sequence": sequence,
                            "inference_seconds": elapsed,
                            "camera_stats": rig.stats,
                            "connection_count": connection_count,
                        }
                    )
                except Full:
                    pass
    except BaseException as exc:
        if not stop.is_set():
            worker_error(events, stop, "perception", exc)
    finally:
        if rig is not None:
            try:
                rig.close()
            except BaseException as exc:
                worker_error(events, stop, "camera_cleanup", exc)
        try:
            events.put(
                {
                    "kind": "closed",
                    "component": "perception",
                    "requests": sequence,
                    "connection_count": connection_count,
                    "camera_stats": None if rig is None else rig.stats,
                },
                timeout=0.2,
            )
        except Full:
            pass


def poll_gripper_state(hands, feedback_values):
    """Consume the persistent stream without discarding/awaiting each sample."""
    hands._poll()
    ages = hands._ages()
    if (
        len(ages) != 2
        or not all(np.isfinite(v) and v >= 0 for v in ages)
        or max(ages) > hands.config.max_state_age_s
    ):
        raise RuntimeFault("Gripper/upper DDS feedback is stale: " + str(ages))
    return {"hands": feedback_values(hands._hands)}


def gripper_worker(config, target_queue, stop, armed, heartbeat, events):
    """DH discovery/read/write cannot block the servo's one-millisecond loop."""
    hands = None
    writes = 0
    writer_creations = 0
    latest = None
    last_write = 0.0
    own_handle = None
    command_reader = None
    action_rows = 0
    maximum_row_lateness = 0.0
    previous_row = None
    stream_horizon = config.get("horizon")
    last_target = None
    duplicate_targets_held = 0
    late_rows_already_held = 0
    maximum_changed_lateness = 0.0
    last_feedback_event = 0.0
    try:
        sys.path.insert(0, config["sdk_root"])
        from pathlib import Path

        from .native import checked_bytes

        checked_bytes(
            Path(config["sdk_root"]) / "atom_training/dh_pc1.py",
            "7d2e2480f4a08a91aaf50600fe18937ed7ab9d3fb5452fe55e31b13970e1e3eb",
        )
        from atom_training.dh_pc1 import (
            DHConfig,
            DualPC1DH,
            command_payload,
            dds_command,
            feedback_values,
            validate_send_state,
        )

        hands = DualPC1DH(
            DHConfig(
                dds_interface="192.168.8.13",
                allow_execute=config["execute"],
                dual_pc1_adaptation_confirmed=True,
            )
        )
        hands.open()
        state = hands.read()
        if any(state["hands"][s]["init_state"] != 1 for s in ("left", "right")):
            raise RuntimeFault("Grippers must be initialized before starting the policy")
        for side in ("left", "right"):
            if state["hands"][side]["grip_state"] == 3:
                raise RuntimeFault(
                    f"{side} gripper reports dropped-object state before startup: "
                    f"{state['hands'][side]}; run Home-start device reset"
                )
        if config["execute"]:
            hands.config.require_execute(True)
            from cyclonedds.core import Listener
            from cyclonedds.pub import DataWriter
            from cyclonedds.qos import Policy, Qos
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic

            hands._discover({"command"})
            hands.Command, types = hands._get_type("command")
            hands.Single, hands.Init = types["DHGripperSingleCmd_"], types["DHGripperInitCmd"]
            topic = hands._own(Topic(hands._participant, "rt/hands/cmd", hands.Command))
            hands._listener = Listener(
                on_publication_matched=lambda entity, status: setattr(
                    hands, "matched", int(status.current_count)
                )
            )
            hands._writer = hands._own(
                DataWriter(
                    hands._participant,
                    topic,
                    listener=hands._listener,
                    qos=Qos(
                        Policy.Reliability.Reliable(max_blocking_time=20_000_000),
                        Policy.History.KeepLast(1),
                    ),
                )
            )
            writer_creations = 1
            deadline = time.monotonic() + 5
            while hands.matched < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            if hands.matched < 1:
                raise RuntimeFault("No compatible PC1 gripper subscriber")
            command_reader = hands._own(
                DataReader(
                    hands._participant,
                    topic,
                    qos=Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(100)),
                )
            )
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if command_reader.take(100):
                    raise RuntimeFault("Another program is commanding the grippers")
                time.sleep(0.01)
            if hands.matched < 2:
                raise RuntimeFault(
                    "PC1 gripper matching was not confirmed separately from the monitor reader"
                )
            # Discovery/matching can block. Require NEW feedback once after it,
            # then keep consuming the stream non-blockingly during playback.
            state = hands.read()
            for side in ("left", "right"):
                if (
                    state["hands"][side]["init_state"] != 1
                    or state["hands"][side]["grip_state"] == 3
                ):
                    raise RuntimeFault(
                        f"{side} gripper not ready after DDS discovery: {state['hands'][side]}"
                    )
        heartbeat.value = time.monotonic()
        events.put(
            {
                "kind": "ready",
                "component": "grippers",
                "state": state,
                "writer_creations": writer_creations,
            }
        )
        while not stop.is_set():
            new_target = False
            try:
                latest = target_queue.get(timeout=0.005)
                new_target = True
            except Empty:
                pass
            state = poll_gripper_state(hands, feedback_values)
            now = time.monotonic()
            heartbeat.value = now
            if stop.is_set():
                break
            if any(
                state["hands"][s]["init_state"] != 1 or state["hands"][s]["grip_state"] == 3
                for s in ("left", "right")
            ):
                raise RuntimeFault("Gripper initialization lost or object drop reported")
            if latest is not None and now - last_feedback_event >= 0.1:
                events.put_nowait(
                    {
                        "kind": "gripper_feedback",
                        "component": "grippers",
                        "observed_at": now,
                        "hands": state["hands"],
                        "dds_received_at": getattr(hands, "_hands_stamp", None),
                        "dds_ages_s": list(hands._ages()),
                    }
                )
                last_feedback_event = now
            if (not config["execute"] or armed.is_set()) and latest is not None:
                require_fresh(latest["time"], now, 0.3, "gripper target heartbeat")
                if new_target:
                    is_row = not latest.get("is_hold", False)
                    target = np.asarray(latest["opening"], dtype=float)
                    if (
                        target.shape != (2,)
                        or not np.isfinite(target).all()
                        or np.any((target < 0) | (target > 1))
                    ):
                        raise RuntimeFault("Invalid gripper target")
                    changed = last_target is None or not np.array_equal(target, last_target)
                    if is_row:
                        stream_horizon = negotiate_horizon(
                            latest.get("horizon", stream_horizon), stream_horizon
                        )
                        key = (latest["sequence"], latest["row"])
                        if previous_row is None:
                            valid = key == (1, 0)
                        elif key[0] == previous_row[0]:
                            valid = key[1] == previous_row[1] + 1
                        else:
                            cancelled = latest.get("cancelled_previous_sequence") == previous_row[0]
                            valid = (
                                key[0] == previous_row[0] + 1
                                and key[1] == 0
                                and (previous_row[1] == stream_horizon - 1 or cancelled)
                            )
                        if not valid or not 0 <= key[1] < stream_horizon:
                            raise RuntimeFault("Missing or duplicate gripper action row")
                        lateness = now - latest["deadline"]
                        held = not changed and last_write <= latest["deadline"]
                        if lateness < 0 or (lateness >= 1 / config["action_fps"] and not held):
                            raise RuntimeFault("Gripper changed target missed its dataset period")
                        if lateness >= 1 / config["action_fps"]:
                            late_rows_already_held += 1
                        maximum_row_lateness = max(maximum_row_lateness, lateness)
                    if config["execute"]:
                        ages = hands._ages()
                        validate_send_state(
                            hands.config,
                            hands._hands,
                            hands._upper,
                            hands_age_s=ages[0],
                            upper_age_s=ages[1],
                            matched=hands.matched - 1,
                        )
                    transmitted = False
                    if changed:
                        if config["execute"]:
                            payload = command_payload(
                                left=float(target[0]),
                                right=float(target[1]),
                                left_force=20.0,
                                right_force=20.0,
                            )
                            message = dds_command(payload, hands.Command, hands.Single, hands.Init)
                            if stop.is_set():
                                break
                            hands._writer.write(message)
                            writes += 1
                            transmitted = True
                        last_write = time.monotonic()
                        last_target = target.copy()
                        if is_row:
                            write_lateness = last_write - latest["deadline"]
                            maximum_changed_lateness = max(maximum_changed_lateness, write_lateness)
                            if write_lateness >= 1 / config["action_fps"]:
                                raise RuntimeFault(
                                    "Gripper changed target DDS write exceeded its dataset period"
                                )
                    else:
                        duplicate_targets_held += 1
                    if is_row:
                        previous_row = key
                        action_rows += 1
                        events.put_nowait(
                            {
                                "kind": "gripper_row",
                                "component": "grippers",
                                "sequence": latest["sequence"],
                                "row": latest["row"],
                                "deadline": latest["deadline"],
                                "processed_at": time.monotonic(),
                                "sent_at": last_write if transmitted else None,
                                "effective_at": last_write,
                                "transmitted": transmitted,
                                "execute": config["execute"],
                                "already_held": not changed,
                                "opening": target.tolist(),
                                "hands": state["hands"],
                            }
                        )
                    # Identify our local writer before any subsequent target can be sent.
                    deadline = time.monotonic() + 0.08
                    while config["execute"] and own_handle is None:
                        for sample in command_reader.take(100):
                            info = getattr(sample, "sample_info", None)
                            handle = int(info.publication_handle) if info else 0
                            if (
                                not handle
                                or abs(sample.left.position - message.left.position) > 1e-5
                                or abs(sample.right.position - message.right.position) > 1e-5
                            ):
                                raise RuntimeFault("Unexpected first gripper command publisher")
                            if own_handle is not None and handle != own_handle:
                                raise RuntimeFault("Multiple gripper command publishers observed")
                            own_handle = handle
                        if own_handle is not None:
                            break
                        if time.monotonic() > deadline:
                            raise RuntimeFault(
                                "Gripper command publisher identity was not observed"
                            )
                        if stop.is_set():
                            break
                        time.sleep(0.001)
            if command_reader is not None and own_handle is not None:
                for sample in command_reader.take(100):
                    info = getattr(sample, "sample_info", None)
                    if info is None or int(info.publication_handle) != own_handle:
                        raise RuntimeFault("Another gripper controller became active")
            time.sleep(0.005)
    except BaseException as exc:
        if not stop.is_set():
            worker_error(events, stop, "grippers", exc)
    finally:
        if hands is not None:
            try:
                hands.close()
            except BaseException as exc:
                worker_error(events, stop, "gripper_cleanup", exc)
        try:
            events.put(
                {
                    "kind": "closed",
                    "component": "grippers",
                    "writes": writes,
                    "writer_creations": writer_creations,
                    "action_rows": action_rows,
                    "maximum_row_lateness_s": maximum_row_lateness,
                    "maximum_changed_target_lateness_s": maximum_changed_lateness,
                    "duplicate_targets_held": duplicate_targets_held,
                    "late_rows_already_held": late_rows_already_held,
                },
                timeout=0.2,
            )
        except Full:
            pass
