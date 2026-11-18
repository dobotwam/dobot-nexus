"""Select the requested Pi0.5 weights and own any policy started for this trial."""

import os
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from dobot_nexus.inference.transport import msgpack_numpy
from examples.atom_robot.policy import policy_contract
from examples.pi05_real_robot._runtime.resolve_model import resolve_model_path

ROOT = Path(__file__).resolve().parents[2]
PORTS = (8001, 8000)


def _connect_policy(stack, port, artifact, report=None):
    """Keep the socket on the caller's stack only when it serves the selected model."""
    from websockets.sync.client import connect

    token = os.environ.get("NEXUS_POLICY_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with ExitStack() as attempt:
        ws = attempt.enter_context(
            connect(
                f"ws://127.0.0.1:{port}",
                compression=None,
                additional_headers=headers,
                open_timeout=3,
            )
        )
        meta = msgpack_numpy.unpackb(ws.recv(timeout=3))
        if not meta.get("artifact") or Path(meta["artifact"]).resolve() != artifact:
            return None
        if meta.get("input_schema", {}).get("state") != [18] or meta.get("output_schema", {}).get(
            "actions"
        ) != [50, 18]:
            raise ValueError(
                f"Incompatible policy at port {port}: expected state [18], actions [50, 18]"
            )
        policy_contract(meta, requested_horizon=50)
        if report is not None:
            report["supports_replay_seed"] = meta.get("supports_replay_seed") is True
        stack.enter_context(attempt.pop_all())
        return ws


def _stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def policy_session(weights, folder, report):
    supplied = Path(weights).expanduser().resolve(strict=True)
    model = resolve_model_path(supplied)
    if (
        supplied.is_file()
        and supplied.name not in ("latest.txt", "latest_deployment.txt")
        and supplied != model.torch_checkpoint
    ):
        raise ValueError(
            f"{supplied} is not the weights file declared in {model.artifact_dir}/manifest.yaml"
        )
    report.update(artifact=str(model.artifact_dir), weights=str(model.torch_checkpoint))
    with ExitStack() as stack:
        available = []
        for port in PORTS:
            try:
                ws = _connect_policy(stack, port, model.artifact_dir, report)
            except ConnectionRefusedError:
                available.append(port)
                continue
            if ws is not None:
                report.update(policy_port=port, policy_reused=True)
                yield ws
                return

        if not available:
            raise RuntimeError("Policy ports 8001 and 8000 are in use by other services")
        port = available[0]
        log = folder / "policy.log"
        report.update(policy_port=port, policy_reused=False, policy_log=str(log))
        print("LOADING_POLICY", model.artifact_dir, "port", port, "log", log, flush=True)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
        )
        command = [
            os.environ.get("POLICY_PYTHON", str(ROOT / ".venvs/pi05-edge/bin/python")),
            "-u",
            "-m",
            "dobot_nexus.models.pi05.deploy.server",
            "--artifact",
            str(model.artifact_dir),
            "--config",
            str(ROOT / "examples/recipes/infer/pi05_atom18.yaml"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-torch-compile",
            "--no-cuda-graph",
        ]
        with log.open("w") as output:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        stack.callback(_stop_child, process)
        deadline = time.monotonic() + 180
        while process.poll() is None and time.monotonic() < deadline:
            try:
                ws = _connect_policy(stack, port, model.artifact_dir, report)
            except ConnectionRefusedError:
                time.sleep(0.5)
                continue
            if ws is None:
                raise RuntimeError(f"Another policy took port {port} during startup; see {log}")
            yield ws
            return
        raise RuntimeError(f"Policy did not become ready (exit={process.poll()}); see {log}")
