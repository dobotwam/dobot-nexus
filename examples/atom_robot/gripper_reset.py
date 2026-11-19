"""Start a new episode through real DH homing/open commands and fresh feedback."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

from .core import RuntimeFault
from .lease import hardware_lock_path


def validate_ready(hands):
    for side in ("left", "right"):
        state = hands[side]
        if (
            state["init_state"] != 1
            or state["grip_state"] != 1
            or not 0.99 <= state["opening"] <= 1
        ):
            raise RuntimeFault(f"{side} gripper reset not confirmed: {state}")


def reset_device(device, *, timeout=10.0, clock=time.monotonic, sleep=time.sleep):
    """Always home, including already-initialized devices with a latched drop flag."""
    before, _ = device.read()
    result = {"before": before, "homed": [], "opened": []}
    for side in ("left", "right"):
        if not device.home(side, timeout):
            raise RuntimeFault(f"{side} gripper homing did not complete")
        result["homed"].append(side)
        if not device.open_hand(side, timeout):
            raise RuntimeFault(f"{side} gripper opening did not complete")
        result["opened"].append(side)
    # A cached 'init=1' is insufficient: require new feedback received after the
    # commands, with both open/in-place, continuously over a settling interval.
    _, previous = device.read()
    deadline, stable = clock() + timeout, None
    while clock() < deadline:
        hands, token = device.read()
        if token != previous:
            previous = token
            try:
                validate_ready(hands)
            except RuntimeFault:
                stable = None
            else:
                now = clock()
                stable = now if stable is None else stable
                if now - stable >= 0.3:
                    return {**result, "after": hands, "status": "reset_confirmed"}
        sleep(0.02)
    raise RuntimeFault(f"Fresh open/in-place gripper feedback timed out: {hands}")


def manufacturer_device(sdk_root=None):
    """Use the installed manufacturer's SDK in an isolated process, without upper writers."""
    from .native import checked_bytes

    sdk_root = sdk_root or os.environ.get("ATOM_GRIPPER_SDK_ROOT")
    if not sdk_root:
        raise RuntimeFault(
            "Set ATOM_GRIPPER_SDK_ROOT to the manufacturer robot_control_dds directory"
        )
    vendor = Path(sdk_root).expanduser().resolve()
    checked_bytes(
        vendor / "script/sdk/DH_sdk.py",
        "2966db8891ec9f1e2e993f4455aba85d944c24ed2d3eb8787e09f36fe1030f4d",
    )
    os.environ["CYCLONEDDS_URI"] = (
        '<CycloneDDS><Domain Id="0"><General><Interfaces>'
        '<NetworkInterface address="192.168.8.13" multicast="true"/>'
        "</Interfaces><AllowMulticast>spdp</AllowMulticast><DontRoute>true</DontRoute>"
        "</General></Domain></CycloneDDS>"
    )
    sys.path.insert(0, str(vendor))
    package = types.ModuleType("sdk")
    package.__path__ = [str(vendor / "script/sdk")]
    sys.modules["sdk"] = package
    module = importlib.import_module("sdk.DH_sdk")
    if module.DEFAULT_FORCE != 20.0 or module._last_cmd != {
        "left": (1.0, 20.0),
        "right": (1.0, 20.0),
    }:
        raise RuntimeFault("Manufacturer reset defaults differ from open/20% targets")
    from cyclonedds.core import Listener
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from sdk.dds_participant import get_qos

    class Device:
        writes = 0
        matched = 0
        last_raw = None

        def writer(self, topic, dtype):
            raw = DataWriter(
                module.get_participant(),
                Topic(module.get_participant(), topic, dtype),
                qos=get_qos(),
                listener=Listener(
                    on_publication_matched=lambda entity, status: setattr(
                        self, "matched", int(status.current_count)
                    )
                ),
            )

            class TrackedWriter:
                def write(proxy, value):
                    raw.write(value)
                    self.writes += 1

            return TrackedWriter()

        def read(self):
            raw = module._raw_dh_state()
            if raw is None:
                raise RuntimeFault("Missing manufacturer gripper feedback")
            self.last_raw = raw  # retain it so object identity cannot be recycled
            parsed = module._parse_dh_state(raw)
            return {
                side: {
                    "init_state": value["init_result_state"],
                    "grip_state": value["grip_state"],
                    "opening": value["realtime_position"],
                }
                for side, value in parsed.items()
            }, id(raw)

        def home(self, side, timeout):
            return self.sdk.DH_Init(side=side, timeout=timeout)

        def open_hand(self, side, timeout):
            return self.sdk.DH_Control(
                1.0, side=side, force=20.0, wait_inplace=True, timeout=timeout
            )

        def neutral(self):
            if self.writes:
                module._write_dh_cmd(
                    left_init=module.InitCmd.DH_INIT_NONE, right_init=module.InitCmd.DH_INIT_NONE
                )

    device = Device()
    module.get_writer = device.writer
    device.sdk = module.DHSdk()
    deadline = time.monotonic() + 5
    while device.matched < 1 or device.sdk.Get_DH_State() is None:
        if time.monotonic() > deadline:
            raise RuntimeFault("No compatible PC1 gripper subscriber or state")
        time.sleep(0.02)
    # Check for competing writers before sending the first reset command.
    monitor = DataReader(
        module.get_participant(),
        Topic(module.get_participant(), "rt/hands/cmd", module.dds_.DHGripperCmd_),
    )
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if monitor.take(100):
            raise RuntimeFault("Another process is commanding the grippers")
        time.sleep(0.01)
    device.monitor = monitor
    return device


def reset_grippers(validation, folder, lock_fd, *, cancelled=lambda: False, sdk_root=None):
    """Keep the caller's hardware lease held while isolating incompatible DDS types."""
    validation, folder = Path(validation).resolve(), Path(folder).resolve()
    report = Path(folder) / "gripper-reset.json"
    command = [
        sys.executable,
        "-u",
        "-m",
        __name__,
        "--execute",
        "--site-clear",
        "--validation",
        str(validation),
        "--report",
        str(report),
        "--lock-fd",
        str(lock_fd),
    ]
    if sdk_root:
        command += ["--sdk-root", str(Path(sdk_root).expanduser().resolve())]
    log = Path(folder) / "gripper-reset.log"
    with log.open("w") as output:
        child = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            pass_fds=(lock_fd,),
            cwd=Path(__file__).resolve().parents[2],
        )
        try:
            deadline = time.monotonic() + 60
            while child.poll() is None:
                if cancelled() or time.monotonic() > deadline:
                    raise RuntimeFault("Device reset interrupted or timed out")
                time.sleep(0.05)
            result = json.loads(report.read_text()) if report.exists() else {}
            if child.returncode or result.get("status") != "reset_confirmed":
                raise RuntimeFault(f"Device reset failed: {result.get('error', 'see ' + str(log))}")
            validate_ready(result["after"])
            return result
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--site-clear", action="store_true")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    parser.add_argument("--sdk-root", default=os.environ.get("ATOM_GRIPPER_SDK_ROOT"))
    args = parser.parse_args(argv)
    if not args.execute or not args.site_clear:
        parser.error("Device reset requires explicit execution and on-site clearance")
    # The child shares the parent's open file description: locking it preserves
    # the same exclusive lease through device reset and arm Home.
    expected = hardware_lock_path().stat()
    inherited = os.fstat(args.lock_fd)
    if (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino):
        raise RuntimeFault("Device reset did not inherit the hardware lease")
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupted(sig, frame):
        raise RuntimeFault("Device reset interrupted by operator")

    handlers = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    device = None
    result = {"status": "reset_failed", "upper_commands_sent": 0, "writes": 0}
    try:
        device = manufacturer_device(args.sdk_root)
        result.update(reset_device(device))
    except BaseException as exc:
        result.update(status="reset_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if device is not None:
            try:
                device.neutral()
            except BaseException as exc:
                result.update(status="reset_failed", cleanup_error=str(exc))
            result["writes"] = device.writes
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print("ATOM_DEVICE_RESET_RESULT", json.dumps(result), flush=True)
    return 0 if result["status"] == "reset_confirmed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
