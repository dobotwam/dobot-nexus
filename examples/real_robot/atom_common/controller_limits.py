"""Use the connected PC1's active position limits in DDS upper-joint order."""

from __future__ import annotations

import copy
import hashlib
import json
import urllib.request

import numpy as np

from .core import STATE_NAMES, RuntimeFault


def read_controller_settings(pc1_ip, *, timeout=2.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    snapshot = {}
    try:
        for key in ("version", "posLimit", "jointOffset"):
            request = urllib.request.Request(f"http://{pc1_ip}:22000/settings/{key}", method="GET")
            with opener.open(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected HTTP status for {key}")
                raw = response.read(131073)
                if len(raw) > 131072:
                    raise ValueError(f"Oversized controller response for {key}")
                snapshot[key] = json.loads(raw)
        controller_bounds(snapshot)
    except Exception as exc:
        raise RuntimeFault(f"Cannot read PC1 position limits: {exc}") from exc
    return snapshot


def controller_bounds(snapshot):
    try:
        version = snapshot["version"]["controller"]
        positions = snapshot["posLimit"]["jointsPosLimit"]
        offsets = snapshot["jointOffset"]["offset"]
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Controller version is missing")
        if any(len(positions[k]) != n for k, n in (("left_arm", 7), ("right_arm", 7), ("head", 2))):
            raise ValueError("Unexpected PC1 position-limit group lengths")
        bounds = np.asarray(
            [positions["waist"]]
            + positions["left_arm"]
            + positions["right_arm"]
            + positions["head"],
            dtype=float,
        )
        if (
            bounds.shape != (17, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("Invalid PC1 position limits")
        if any(
            len(offsets[k]) != n
            for k, n in (("waist", 1), ("left_arm", 7), ("right_arm", 7), ("head", 2))
        ):
            raise ValueError("Unexpected PC1 calibration group lengths")
        zeroes = np.asarray(
            offsets["waist"] + offsets["left_arm"] + offsets["right_arm"] + offsets["head"],
            dtype=float,
        )
        if zeroes.shape != (17,) or not np.isfinite(zeroes).all():
            raise ValueError("Invalid PC1 calibration metadata")
        return bounds
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeFault(f"Invalid PC1 upper-joint limit contract: {exc}") from exc


def limit_fingerprint(snapshot):
    controller_bounds(snapshot)
    selected = {
        "controller": snapshot["version"]["controller"],
        "limits": snapshot["posLimit"]["jointsPosLimit"],
        "offsets": snapshot["jointOffset"]["offset"],
    }
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def apply_controller_limits(settings, snapshot, *, pc1_ip):
    bounds = controller_bounds(snapshot)
    expected_names = ("torso_joint",) + STATE_NAMES[:7] + STATE_NAMES[8:15] + STATE_NAMES[16:]
    joints = settings["joints"]
    if (
        len(joints) != 17
        or [j["index"] for j in joints] != list(range(17))
        or tuple(j["name"].replace("wrist_roll_joint", "wrist_yaw_joint") for j in joints)
        != expected_names
    ):
        raise RuntimeFault("SDK joint names/order do not match the PC1 upper limit contract")
    result = copy.deepcopy(settings)
    for row, bounds_row in zip(result["joints"], bounds, strict=True):
        row["limits"] = bounds_row.tolist()
    result["controller_position_limits"] = {
        "source": f"http://{pc1_ip}:22000/settings/posLimit",
        "controller_version": snapshot["version"]["controller"],
        "units": "rad",
        "fingerprint": limit_fingerprint(snapshot),
        "bounds_rad": bounds.tolist(),
    }
    return result


def assert_controller_limits_unchanged(settings, *, pc1_ip):
    current = read_controller_settings(pc1_ip)
    if limit_fingerprint(current) != settings["controller_position_limits"]["fingerprint"]:
        raise RuntimeFault(
            "PC1 limits, calibration or controller version changed before control acquisition"
        )
