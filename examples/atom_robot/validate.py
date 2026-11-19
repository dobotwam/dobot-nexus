"""Run two independent real-sensor lifecycles without creating command writers."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from .core import RuntimeFault
from .policy import preflight_policy
from .run import ROOT, config_for, parse_args


def main(argv=None):
    args = parse_args(argv)
    if args.execute:
        raise RuntimeFault("Validation never accepts --execute; use the runtime after validation")
    if not args.sdk_root:
        raise RuntimeFault("Set ATOM_SDK_ROOT or pass --sdk-root")
    _, horizon = preflight_policy(config_for(args, None))
    validation = Path(args.validation).expanduser().resolve()
    folder = validation / (
        "read-only-validation-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    folder.mkdir(parents=True)
    reports = []
    for number in (1, 2):
        log = folder / f"read-only-{number}.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            "examples.atom_robot.run",
            "--sdk-root",
            str(Path(args.sdk_root).expanduser().resolve()),
            "--cameras",
            str(Path(args.cameras).expanduser().resolve()),
            "--validation",
            str(validation),
            "--head-mode",
            args.head_mode,
            "--policy-host",
            args.policy_host,
            "--policy-port",
            str(args.policy_port),
            "--task",
            args.task,
            "--task-guards",
            args.task_guards,
            "--action-fps",
            str(args.action_fps),
            "--horizon",
            str(horizon),
            "--max-horizons",
            "3",
            "--max-seconds",
            str(args.max_seconds),
        ]
        if args.gripper_sdk_root:
            command += [
                "--gripper-sdk-root",
                str(Path(args.gripper_sdk_root).expanduser().resolve()),
            ]
        print(f"Read-only lifecycle {number}/2; log={log}", flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            print(log.read_text(errors="replace")[-8000:], flush=True)
            return completed.returncode
        report = None
        for line in log.read_text().splitlines():
            if line.startswith("ATOM_RUNTIME_RESULT "):
                report = Path(json.loads(line.split(" ", 1)[1])["report_dir"]) / "result.json"
        if report is None:
            raise RuntimeFault("Read-only runtime produced no final report")
        reports.append(report)
    command = [sys.executable, "-m", "examples.atom_robot.verify", "--validation", str(validation)]
    for report in reports:
        command += ["--report", str(report)]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
