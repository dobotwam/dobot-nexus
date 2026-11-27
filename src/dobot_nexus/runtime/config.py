"""Driver preflight checks that can run before experiment creation."""

import os
from collections.abc import Mapping


def validate_launcher_env(config):
    if config.get("runtime", {}).get("launcher") == "rlinf":
        if config["runtime"].get("fsdp", {}).get("enabled", False):
            raise ValueError("runtime.launcher=rlinf does not support FSDP.")
        algorithm = config.get("algorithm")
        if not isinstance(algorithm, Mapping) or not algorithm.get("target"):
            raise ValueError("runtime.launcher=rlinf requires algorithm.target.")
        if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
            raise ValueError("RLinf requires one driver; do not launch it with torchrun.")


def validate_ray_tmpdir(path):
    """Validate the actual RAY_TMPDIR, including the driver PID in Ray's socket path."""
    directory = os.path.abspath(path)
    session = f"session_2000-01-01_00-00-00_000000_{os.getpid()}"
    socket = os.path.join(directory, "ray", session, "sockets", "plasma_store")
    if len(directory.encode("utf-8")) > 38 or len(socket.encode("utf-8")) >= 104:
        raise ValueError(f"RAY_TMPDIR too long for AF_UNIX sockets: {directory}")
