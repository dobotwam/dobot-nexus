"""Validate CPU contracts and at least two independent real read-only lifecycles."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import unittest
from pathlib import Path

from .validation import source_fingerprint, validate_read_only_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "runs/atom_robot/validation",
    )
    args = parser.parse_args(argv)
    if len(args.report) < 2 or len(set(p.resolve() for p in args.report)) < 2:
        parser.error("Two independent runtime reports are required")
    root = Path(__file__).resolve().parents[2]
    args.validation.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(
        str(root / "tests/entrypoints/atom"), top_level_dir=str(root)
    )
    with (args.validation / "atom-runtime-unit-tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("CPU contract tests failed; validation receipt was not written")
    runs = []
    for path in args.report:
        raw = path.read_bytes()
        proof = validate_read_only_report(json.loads(raw))
        runs.append(
            {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "proof": proof}
        )
    if len({run["sha256"] for run in runs}) != len(runs):
        raise SystemExit("Copied reports are not independent validation runs")
    runtime = runs[0]["proof"]["runtime_identity"]
    policy = runs[0]["proof"]["policy_identity"]
    if (
        not runtime
        or not policy
        or any(
            run["proof"]["runtime_identity"] != runtime or run["proof"]["policy_identity"] != policy
            for run in runs
        )
    ):
        raise SystemExit("Read-only runs must use the same runtime settings and policy")
    horizons = {run["proof"]["action_horizon"] for run in runs}
    if len(horizons) != 1:
        raise SystemExit("Read-only runs used different policy horizons")
    frequencies = {run["proof"]["action_fps"] for run in runs}
    if len(frequencies) != 1:
        raise SystemExit("Read-only runs used different action frequencies")
    receipt = {
        "status": "passed",
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": source_fingerprint(),
        "repeat_runs": len(runs),
        "runs": runs,
        "unit_tests_passed": True,
        "unit_tests_run": result.testsRun,
        "motion_tested": False,
        "table_collision_hardware_validation": False,
        "inference_horizon": horizons.pop(),
        "action_fps": frequencies.pop(),
        "runtime_identity": runtime,
        "policy_identity": policy,
    }
    output = args.validation / "atom-runtime-validation.json"
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    tmp.replace(output)
    print(
        "ATOM_REUSE_VALIDATION_PASSED",
        json.dumps(
            {
                "receipt": str(output),
                "repeat_runs": len(runs),
                "unit_tests_run": result.testsRun,
                "inference_horizon": receipt["inference_horizon"],
                "robot_commands_sent": 0,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
