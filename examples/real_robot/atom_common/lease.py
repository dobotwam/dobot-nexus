"""One host-wide lease for this hardware, independent of the report directory."""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path


def hardware_lock_path():
    return Path("/tmp/dobot-nexus-atom-w-0759.lock")


@contextmanager
def atom_d_control_lease():
    """Exclude other D runners and coordinate with an installed collection console."""
    import fcntl

    root = Path(os.environ.get("ATOM_D_CONTROL_ROOT", Path(__file__).resolve().parents[3]))
    if not root.is_dir():
        raise FileNotFoundError(f"ATOM_D_CONTROL_ROOT does not exist: {root}")
    paths = [Path("/tmp/dobot-nexus-atom-d.lock")]
    edge = root / "local_script/atom_edge"
    if edge.is_dir():
        validation = edge / "validation"
        validation.mkdir(exist_ok=True)
        paths.append(validation / "home-start.lock")
    collection = root / "local_script/collection_console/runtime"
    if collection.is_dir():
        paths.append(collection / "robot-control.lock")
    with ExitStack() as stack:
        for path in paths:
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
