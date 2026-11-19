"""A source-bound validation receipt is required before live execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .core import RuntimeFault, negotiate_horizon
from .policy import policy_identity


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    files = list((root / "examples/atom_robot").rglob("*.py")) + list(
        (root / "examples/atom_robot/config").glob("*.json")
    )
    files += list((root / "src/dobot_nexus").rglob("*.py"))
    files += list((root / "tests/entrypoints/atom").glob("*.py"))
    files += list((root / "examples/pi05_real_robot/layouts").glob("*.json"))
    files += [
        root / "examples/pi05_real_robot/_runtime/robot_contract.py",
        root / "examples/recipes/infer/pi05_atom18.yaml",
        root / "tests/fixtures/atom_training_chunk.json",
    ]
    files += list((root / "examples/atom_robot").glob("*.sh"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def validate_read_only_report(report):
    if report.get("sources") != source_fingerprint():
        raise RuntimeFault("Read-only report was produced by a different runtime source")
    if report.get("status") != "completed" or report.get("execute") is not False:
        raise RuntimeFault("Validation run did not complete in read-only mode")
    metadata = report.get("server_metadata", {})
    horizon = negotiate_horizon(metadata.get("policy_spec", {}).get("action_horizon"))
    if report.get("horizon") != horizon or metadata.get("output_schema", {}).get("actions") != [
        horizon,
        18,
    ]:
        raise RuntimeFault("Actual policy and consumed action horizons differ")
    if (
        report.get("commands_sent") != 0
        or report.get("gripper_commands_sent") != 0
        or report.get("motor_writer_created") is not False
    ):
        raise RuntimeFault("Read-only validation created or sent motor commands")
    if report.get("dds_session_initializations") != 1:
        raise RuntimeFault("DDS session was not reused")
    if (
        report.get("completed_horizons", 0) < 3
        or report.get("consumed_actions") != report["completed_horizons"] * horizon
    ):
        raise RuntimeFault("Validation did not consume every row of at least three action chunks")
    if report.get("maximum_scheduler_gap_s", 1) > 0.05:
        raise RuntimeFault("Shadow servo scheduling exceeded the reviewed 50ms bound")
    if report.get("command_law") != "sdk_reference" or report.get("action_fps") != 20.0:
        raise RuntimeFault("Validation did not exercise the 20 Hz SDK reference executor")
    rows = report["completed_horizons"] * horizon
    if (
        report.get("dispatched_action_rows") != rows
        or report.get("gripper_action_rows_consumed") != rows
    ):
        raise RuntimeFault("Joint or gripper playback skipped action rows")
    if (
        max(
            report.get("maximum_row_dispatch_lateness_s", 1),
            report.get(
                "maximum_gripper_changed_target_lateness_s",
                report.get("maximum_gripper_row_lateness_s", 1),
            ),
        )
        >= 0.05
    ):
        raise RuntimeFault("Action dispatch did not meet the dataset period")
    if report.get("worker_exit_codes") != [0, 0] or report.get("forced_worker_shutdown"):
        raise RuntimeFault("Required workers did not cleanly exit")
    closed = {e["component"]: e for e in report["worker_events"] if e["kind"] == "closed"}
    perception = closed.get("perception", {})
    hands = closed.get("grippers", {})
    if (
        perception.get("connection_count") != 1
        or hands.get("writer_creations") != 0
        or hands.get("writes") != 0
    ):
        raise RuntimeFault(
            "Policy connection or read-only hand lifecycle did not match the contract"
        )
    stats = perception.get("camera_stats", {})
    if set(stats) != {"top_camera", "left_camera", "right_camera"} or any(
        v["starts"] != 1 or v["stops"] != 1 or v["frames"] < 60 for v in stats.values()
    ):
        raise RuntimeFault("Camera streams were restarted or not cleanly closed")
    return {
        "horizons": report["completed_horizons"],
        "action_horizon": horizon,
        "actions": report["consumed_actions"],
        "dds_initializations": 1,
        "policy_connections": 1,
        "camera_lifecycle": stats,
        "commands_sent": 0,
        "action_fps": report["action_fps"],
        "runtime_identity": report.get("runtime_identity"),
        "policy_identity": policy_identity(metadata),
    }


def require_live_validation(
    validation,
    expected_horizon=None,
    expected_action_fps=None,
    *,
    expected_runtime=None,
    expected_policy=None,
):
    path = Path(validation) / "atom-runtime-validation.json"
    if not path.exists():
        raise RuntimeFault("Run the reusable read-only validation before live execution")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "passed" or receipt.get("sources") != source_fingerprint():
        raise RuntimeFault("Runtime source changed after validation; repeat the validation gate")
    if receipt.get("repeat_runs", 0) < 2 or receipt.get("unit_tests_passed") is not True:
        raise RuntimeFault("Incomplete reusable runtime validation receipt")
    if expected_horizon is not None and receipt.get("inference_horizon") != expected_horizon:
        raise RuntimeFault(
            "Policy horizon changed after validation; repeat the read-only validation"
        )
    if expected_action_fps is not None and receipt.get("action_fps") != expected_action_fps:
        raise RuntimeFault(
            "Action frequency changed after validation; repeat the read-only validation"
        )
    if expected_runtime is not None and receipt.get("runtime_identity") != expected_runtime:
        raise RuntimeFault(
            "Runtime settings changed after validation; repeat the read-only validation"
        )
    if expected_policy is not None and receipt.get("policy_identity") != expected_policy:
        raise RuntimeFault("Policy changed after validation; repeat the read-only validation")
    return receipt
