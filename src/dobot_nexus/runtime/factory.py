"""Production backend selection."""


def create_runtime(config, logger=None):
    launcher = config.get("runtime", {}).get("launcher")
    if launcher != "rlinf":
        raise ValueError(f"Distributed runtime requires runtime.launcher=rlinf, got {launcher!r}.")
    try:
        from .ray.backend import RLinfRayRuntime
    except ImportError as exc:
        raise ImportError(
            "Install the pinned Ray and RLinf dependencies from "
            "src/dobot_nexus/runtime/ray/requirements.txt "
            "(or algorithms/rltoken/requirements-rlinf.txt)."
        ) from exc
    return RLinfRayRuntime(config, logger)
