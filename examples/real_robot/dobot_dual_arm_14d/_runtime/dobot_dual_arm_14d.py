#!/usr/bin/env python3
"""Dobot dual-arm 14D hardware runtime and dimension-aware WebSocket bridge.

Use ../start_dobot_dual_arm_14d_server.sh on the robot and ../start_robot_client.sh on the
GPU host. The launchers supply task, hardware and deployment settings; see
../README.md. Hardware and policy-client dependencies are imported on demand.
"""

from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import os
import queue
import select
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

if __package__:
    from .wire import authenticate, configured_key, send_message, recv_message
    from .dimensions import CommunicationSpec, positive_dimension
else:
    from wire import authenticate, configured_key, send_message, recv_message
    from dimensions import CommunicationSpec, positive_dimension


PROTOCOL_VERSION = "dual_env_socket_v2"
STATE_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
]
IMAGE_NAMES = ["top_camera", "left_camera", "right_camera"]
CONTROL_STATE_PREFIX = "[control-state] "
MODEL_HANDOFF_HINT = (
    "[control] Before starting or reconnecting the client, disable manual follow, "
    "type 'model' and press Enter in this robot-server terminal. "
    "Wait for state=model_waiting; a connected model enters state=model_control."
)
FOLLOW_SIGNS = np.array([1, 1, -1, -1, -1, 1], dtype=np.float32)


class ModelSessionCancelled(RuntimeError):
    """Raised when Web control cancels the active model session."""


class RobotHardwareLease:
    """Exclusive cross-process lease held for the full robot-server lifetime."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = None

    def acquire(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            file.close()
            raise RuntimeError(
                "Robot hardware is already in use by another process"
            ) from exc
        self._file = file

    def release(self) -> None:
        file = self._file
        self._file = None
        if file is None:
            return
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        finally:
            file.close()


class ClientHealthServer:
    """Authenticated, read-only health endpoint owned by one OpenPI Client."""

    _MAX_REQUEST_BYTES = 4096
    _MAX_CONCURRENT_REQUESTS = 8

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        request_timeout: float = 0.2,
    ):
        if not isinstance(token, str) or not token:
            raise ValueError("health token must be a non-empty string")
        if request_timeout <= 0:
            raise ValueError("health request timeout must be positive")
        self._host = host
        self._port = port
        self._token = token.encode("utf-8")
        self._request_timeout = float(request_timeout)
        self._state = "starting"
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._clients_lock = threading.Lock()
        self._clients: set[socket.socket] = set()
        self._workers: set[threading.Thread] = set()
        self._worker_slots = threading.BoundedSemaphore(
            self._MAX_CONCURRENT_REQUESTS
        )
        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._address: tuple[str, int] | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._address is None:
            raise RuntimeError("health server has not started")
        return self._address

    def start(self) -> None:
        if self._accept_thread is not None:
            raise RuntimeError("health server has already started")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self._host, self._port))
            server_socket.listen()
            server_socket.settimeout(self._request_timeout)
        except Exception:
            server_socket.close()
            raise
        self._server_socket = server_socket
        bound_host, bound_port = server_socket.getsockname()[:2]
        self._address = (str(bound_host), int(bound_port))
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="openpi-client-health",
            daemon=True,
        )
        self._accept_thread.start()

    def set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state

    def close(self) -> None:
        self._stop.set()
        server_socket = self._server_socket
        self._server_socket = None
        if server_socket is not None:
            server_socket.close()

        with self._clients_lock:
            clients = list(self._clients)
        for connection in clients:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

        accept_thread = self._accept_thread
        if accept_thread is not None and accept_thread is not threading.current_thread():
            accept_thread.join(self._request_timeout + 1.0)

        with self._clients_lock:
            workers = list(self._workers)
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(self._request_timeout + 1.0)

    def _accept_loop(self) -> None:
        server_socket = self._server_socket
        if server_socket is None:
            return
        while not self._stop.is_set():
            try:
                connection, _ = server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                connection.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )
            except OSError:
                connection.close()
                continue
            self._start_worker(connection)

    def _start_worker(self, connection: socket.socket) -> bool:
        if self._stop.is_set() or not self._worker_slots.acquire(blocking=False):
            connection.close()
            return False
        worker = threading.Thread(
            target=self._serve_connection,
            args=(connection,),
            name="openpi-client-health-request",
            daemon=True,
        )
        try:
            with self._clients_lock:
                self._clients.add(connection)
                self._workers.add(worker)
                worker.start()
        except Exception:
            with self._clients_lock:
                self._clients.discard(connection)
                self._workers.discard(worker)
            self._worker_slots.release()
            connection.close()
            raise
        return True

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            deadline = time.monotonic() + self._request_timeout
            try:
                request = self._read_request(connection, deadline=deadline)
            except EOFError:
                return
            response = self._health_response(request)
            payload = json.dumps(response, separators=(",", ":")).encode(
                "utf-8"
            )
            connection.sendall(payload + b"\n")
            if request is not None and response.get("ok") is True:
                self._wait_for_peer_close(connection, deadline=deadline)
        except (OSError, UnicodeError, ValueError):
            pass
        finally:
            connection.close()
            with self._clients_lock:
                self._clients.discard(connection)
                self._workers.discard(threading.current_thread())
            self._worker_slots.release()

    def _wait_for_peer_close(
        self,
        connection: socket.socket,
        *,
        deadline: float | None = None,
    ) -> None:
        if deadline is None:
            deadline = time.monotonic() + self._request_timeout
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            connection.settimeout(remaining)
            try:
                chunk = connection.recv(1024)
            except socket.timeout:
                return
            if not chunk:
                return

    def _read_request(
        self,
        connection: socket.socket,
        *,
        deadline: float | None = None,
    ) -> bytes | None:
        request = bytearray()
        if deadline is None:
            deadline = time.monotonic() + self._request_timeout
        while len(request) <= self._MAX_REQUEST_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            connection.settimeout(remaining)
            chunk = connection.recv(
                min(1024, self._MAX_REQUEST_BYTES + 1 - len(request))
            )
            if not chunk:
                raise EOFError
            request.extend(chunk)
            newline = request.find(b"\n")
            if newline >= 0:
                if newline > self._MAX_REQUEST_BYTES:
                    return None
                return bytes(request[:newline])
            if len(request) > self._MAX_REQUEST_BYTES:
                return None
        return None

    def _health_response(self, request: bytes | None) -> Dict[str, Any]:
        try:
            message = json.loads(request.decode("utf-8")) if request is not None else None
        except (UnicodeError, ValueError):
            message = None
        supplied_token = message.get("token") if isinstance(message, dict) else None
        try:
            supplied_token_bytes = supplied_token.encode("utf-8")
        except (AttributeError, UnicodeError):
            return {"ok": False}
        if not hmac.compare_digest(supplied_token_bytes, self._token):
            return {"ok": False}
        with self._state_lock:
            state = self._state
        return {"ok": True, "pid": os.getsid(0), "state": state}


# ─────────────────────────── socket helpers ───────────────────────────

def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def log(message: str) -> None:
    print(message, flush=True)


# ─────────────────────────── home hotkey ──────────────────────────────

def _flush_stdin() -> None:
    """Drop anything already typed, so a safety prompt cannot be satisfied by
    keys buffered before it appeared (e.g. a held-down 'h')."""
    if not sys.stdin.isatty():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


class OperatorControl:
    """Own line-based stdin for confirmed action controls.

    A dedicated reader avoids mixing ``input()`` with ``select()``: Python's
    text buffering can otherwise consume a stop command while a prior line is
    being read.  Commands are deliberately queued in order, so ``r\\ns\\n``
    (start hold, then release) cannot turn into an extra action.
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        fd: int | None = None,
        on_command=None,
        on_disconnect=None,
    ):
        self._enabled = enabled
        self._fd = fd
        self._on_command = on_command
        self._on_disconnect = on_disconnect
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._commands: queue.Queue[str] = queue.Queue()
        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if not self._enabled or self._active:
            return
        if self._fd is None:
            if not sys.stdin.isatty():
                log("[control] stdin is not a tty; confirmed action controls are unavailable")
                return
            self._fd = sys.stdin.fileno()
        self._stop.clear()
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="robot-operator-control")
        self._thread.start()
        log("[control] confirmed-action input armed (Enter=one, hold Enter=continuous, h=home)")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    ready, _, _ = select.select([self._fd], [], [], 0.1)
                except (OSError, ValueError):
                    return
                if not ready:
                    continue
                try:
                    data = os.read(self._fd, 4096)
                except OSError:
                    return
                if not data:
                    return
                self.feed(data)
        finally:
            self._active = False
            if not self._stop.is_set() and self._on_disconnect is not None:
                self._on_disconnect()

    def feed(self, data: bytes) -> None:
        """Parse complete terminal lines; kept public for isolated regression tests."""
        parsed = []
        with self._buffer_lock:
            self._buffer.extend(data)
            while True:
                newline = self._buffer.find(b"\n")
                if newline < 0:
                    break
                raw = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                parsed.append(raw.decode(errors="ignore").strip().lower())
        for command in parsed:
            self._commands.put(command)
            if self._on_command is not None:
                self._on_command(command)

    def get_nowait(self) -> str | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def wait(self, prompt: str, *, poll=None) -> str:
        log(prompt)
        while not self._stop.is_set():
            if poll is not None:
                poll()
            try:
                return self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
        raise EOFError("operator control stopped while waiting for a command")

    def clear(self) -> None:
        """Drop queued controls before an explicit safety gate."""
        with self._buffer_lock:
            self._buffer.clear()
        while self.get_nowait() is not None:
            pass
        _flush_stdin()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._active = False


class HomeHotkey:
    """Watch stdin for 'h' so the operator can send the arms home mid-chunk.

    Only runs a reader thread when actions are NOT individually confirmed; with
    --confirm-each-action, OperatorControl owns stdin and forwards `h` at each
    action boundary instead (see RobotRuntimeEnv._execute_one_action).
    Terminal state is restored on stop(), including when the loop dies.
    """

    def __init__(self, enabled: bool = True):
        self.requested = threading.Event()
        self._enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._saved_term = None

    def start(self) -> None:
        if not self._enabled:
            return
        if not sys.stdin.isatty():
            log("[hotkey] stdin is not a tty; 'h' key disabled (use --confirm-each-action instead)")
            return
        try:
            import termios
            import tty
        except ImportError:
            log("[hotkey] termios unavailable; 'h' key disabled")
            return
        self._fd = sys.stdin.fileno()
        try:
            self._saved_term = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except termios.error as exc:
            log(f"[hotkey] could not set cbreak mode ({exc}); 'h' key disabled")
            self._fd = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log("[hotkey] armed: press 'h' at any time to send the arms back to home")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                raw = os.read(self._fd, 1)
            except OSError:
                return
            if not raw:  # EOF (terminal closed) — stop instead of spinning on a ready fd
                return
            ch = raw.decode(errors="ignore")
            if ch.lower() == "h":
                log("\n[hotkey] 'h' pressed -> will return to home at the next step boundary")
                self.requested.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._restore()

    def _restore(self) -> None:
        if self._saved_term is None or self._fd is None:
            return
        try:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_term)
        except Exception:
            pass
        self._saved_term = None


# ─────────────────────────── image helpers ────────────────────────────

def resize_image_to_square_rgb(image_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    import cv2
    h, w = image_bgr.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    tx = (target_size - new_w) / 2
    ty = (target_size - new_h) / 2
    transform = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
    resized = cv2.warpAffine(image_bgr, transform, (target_size, target_size))
    return np.ascontiguousarray(resized[..., ::-1])  # BGR → RGB


def crop_xywh(image: np.ndarray, roi_xywh: Iterable[int] | None) -> np.ndarray:
    """Crop image using (x, y, width, height) top-left convention."""
    if roi_xywh is None:
        return image
    x, y, roi_w, roi_h = [int(v) for v in roi_xywh]
    if roi_w <= 0 or roi_h <= 0:
        raise ValueError(f"ROI width/height must be positive, got {list(roi_xywh)}")
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + roi_w)
    y2 = min(h, y + roi_h)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise RuntimeError(f"Empty crop from roi_xywh={list(roi_xywh)} image_shape={image.shape}")
    return cropped


# ─────────────────────────── robot runtime ────────────────────────────

CLOSED_GRIPPER_POSITION = 0.0


def home_array(values: Iterable[float], default_gripper: float = 1.0) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.shape == (6,):
        arr = np.concatenate([arr, np.array([default_gripper], dtype=np.float32)])
    if arr.shape != (7,):
        raise ValueError(f"Home position must have 6 joints + optional gripper, got shape {arr.shape}")
    arr[6] = clamp01(arr[6])
    return arr


def configured_home_positions(
    left_home_values: Iterable[float],
    right_home_values: Iterable[float],
    *,
    close: tuple[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build startup Home poses and apply explicit gripper Home overrides."""
    left_home = home_array(left_home_values)
    right_home = home_array(right_home_values)
    if close is None:
        return left_home, right_home
    if close != ("right", "gripper"):
        raise ValueError("Only '--close right gripper' is supported")
    right_home[6] = CLOSED_GRIPPER_POSITION
    return left_home, right_home


def configure_path(path: str | None) -> None:
    if path is None:
        return
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class PositioningControl:
    """Own manual-follow and model-control state inside the robot process.

    The Web process can enable/disable *follow*, but it has no command capable
    of changing master-arm torque.  Master torque changes only after a physical
    Key1 release observed in the Telop data stream.
    """

    def __init__(
        self,
        env,
        args: argparse.Namespace,
        *,
        emit=log,
        light_states: Dict[str, Any] | None = None,
    ):
        if light_states is None:
            from dobot_arm import DobotArmLightState

            light_states = {
                "off": DobotArmLightState.Off,
                "yellow": DobotArmLightState.Yellow,
                "green": DobotArmLightState.Green,
            }
        self.env = env
        self.args = args
        self._emit = emit
        self._light_states = light_states
        self.state = "positioning_paused"
        self.control_owner = "positioning"
        self.left_follow = False
        self.right_follow = False
        self.left_adjustment = True
        self.right_adjustment = True
        self.left_telop_torque = True
        self.right_telop_torque = True
        self._origins = {"left": None, "right": None}
        self._previous_buttons = None
        self.light = "off"
        self.last_error = ""
        self._set_light("off", force=True)
        self._emit_state()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "control_owner": self.control_owner,
            "left_follow": self.left_follow,
            "right_follow": self.right_follow,
            "left_adjustment": self.left_adjustment,
            "right_adjustment": self.right_adjustment,
            "left_telop_torque": self.left_telop_torque,
            "right_telop_torque": self.right_telop_torque,
            "light": self.light,
            "last_error": self.last_error,
        }

    def _emit_state(self) -> None:
        self._emit(
            CONTROL_STATE_PREFIX
            + json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True)
        )

    def _set_light(self, light: str, *, force: bool = False) -> None:
        if not force and light == self.light:
            return
        self.light = light
        self.env.set_light_state(self._light_states[light])

    @staticmethod
    def _validate_side(side: str) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")

    def _capture_origin(self, side: str, observation: Dict[str, Any] | None = None) -> None:
        observation = observation if observation is not None else self.env.get_observation()
        self._origins[side] = {
            "dobot": np.asarray(
                observation[f"{side}_dobot_arm_joints"][:6], dtype=np.float32
            ).copy(),
            "telop": np.asarray(
                observation[f"{side}_telop_arm_joints_and_buttons"][:6],
                dtype=np.float32,
            ).copy(),
        }

    def enter_positioning(self) -> None:
        self.state = "positioning_paused"
        self.control_owner = "positioning"
        self.left_follow = False
        self.right_follow = False
        self.left_adjustment = True
        self.right_adjustment = True
        self._origins = {"left": None, "right": None}
        self.last_error = ""
        self._set_light("off")
        self._emit_state()

    def set_adjustment(self, side: str, enabled: bool) -> None:
        self._validate_side(side)
        if self.state not in {"positioning_paused", "positioning_follow"}:
            raise RuntimeError("adjustment is only available while positioning")
        enabled = bool(enabled)
        follow = not enabled
        if follow and not getattr(self, f"{side}_follow"):
            self._capture_origin(side)
        if not follow:
            self._origins[side] = None
        setattr(self, f"{side}_adjustment", enabled)
        setattr(self, f"{side}_follow", follow)
        self.state = (
            "positioning_follow"
            if self.left_follow or self.right_follow
            else "positioning_paused"
        )
        self.last_error = ""
        self._set_light("yellow" if self.state == "positioning_follow" else "off")
        self._emit_state()

    def handoff_to_model(self) -> None:
        if self.left_follow or self.right_follow:
            raise RuntimeError("manual follow must be disabled before model handoff")
        if self.state not in {"positioning_paused", "model_waiting"}:
            raise RuntimeError("model handoff requires paused positioning")
        self.state = "model_waiting"
        self.control_owner = "model"
        self._set_light("off")
        self.last_error = ""
        self._emit_state()

    def model_connected(self) -> None:
        if self.state != "model_waiting":
            raise RuntimeError("model client connected before positioning handoff")
        self.state = "model_control"
        self.control_owner = "model"
        self._set_light("green")
        self.last_error = ""
        self._emit_state()

    def model_disconnected(self) -> None:
        self.enter_positioning()

    def begin_home(self) -> None:
        self.left_follow = False
        self.right_follow = False
        self.left_adjustment = True
        self.right_adjustment = True
        self._origins = {"left": None, "right": None}
        self.state = "homing"
        self.control_owner = "homing"
        self._set_light("off")
        self._emit_state()

    def complete_home(self) -> None:
        self.enter_positioning()

    def wait_for_home_resume(self) -> None:
        if self.state != "homing":
            raise RuntimeError("Home completion requires homing state")
        self.state = "home_waiting"
        self.control_owner = "homing"
        self._set_light("off")
        self._emit_state()

    def resume_model_after_home(self) -> None:
        if self.state != "home_waiting":
            raise RuntimeError("model resume requires completed Home state")
        self.state = "model_control"
        self.control_owner = "model"
        self.last_error = ""
        self._set_light("green")
        self._emit_state()

    def report_error(self, exc: BaseException) -> None:
        self.last_error = str(exc)
        self._emit_state()

    def _buttons(self, observation: Dict[str, Any]) -> Dict[str, tuple[int, int]]:
        return {
            side: tuple(
                int(value)
                for value in observation[
                    f"{side}_telop_arm_joints_and_buttons"
                ][-2:]
            )
            for side in ("left", "right")
        }

    def observe_physical_buttons(self, observation: Dict[str, Any]) -> None:
        current = self._buttons(observation)
        if self._previous_buttons is None:
            self._previous_buttons = current
            return
        changed = False
        for side in ("left", "right"):
            previous = self._previous_buttons[side]
            buttons = current[side]
            if previous[0] == 1 and buttons[0] == 0:
                enabled = not getattr(self, f"{side}_telop_torque")
                if self.args.execute:
                    setter = getattr(self.env, f"set_{side}_telop_torque_mode")
                    setter(enabled)
                setattr(self, f"{side}_telop_torque", enabled)
                changed = True
        self._previous_buttons = current
        if changed:
            self._emit_state()

    def _follow_action(
        self, side: str, observation: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        current = np.asarray(
            observation[f"{side}_dobot_arm_joints"], dtype=np.float32
        )
        origin = self._origins[side]
        if origin is None:
            self._capture_origin(side, observation)
            origin = self._origins[side]
        telop = np.asarray(
            observation[f"{side}_telop_arm_joints_and_buttons"], dtype=np.float32
        )
        joints = origin["dobot"] + (telop[:6] - origin["telop"]) * FOLLOW_SIGNS
        gripper = 1.0 - clamp01((float(telop[6]) - 3.0) * 2.0)
        action = np.concatenate(
            [joints, np.array([gripper], dtype=np.float32)]
        ).astype(np.float32, copy=False)
        delta = float(np.max(np.abs(action[:6] - current[:6])))
        limit = float(self.args.max_joint_delta)
        if limit > 0 and delta >= limit:
            message = (
                f"{side} follow delta {delta:.3f} rad exceeds limit {limit:.3f}"
            )
            if self.last_error != message:
                self.last_error = message
                self._emit_state()
            return None
        if self.last_error.startswith(f"{side} follow delta "):
            self.last_error = ""
            self._emit_state()
        return action, current[:6].copy()

    def process_tick(self) -> None:
        if self.state not in {"positioning_paused", "positioning_follow"}:
            return
        observation = self.env.get_observation()
        self.observe_physical_buttons(observation)
        if not (self.left_follow or self.right_follow):
            return
        payload = {}
        for side in ("left", "right"):
            if not getattr(self, f"{side}_follow"):
                continue
            result = self._follow_action(side, observation)
            if result is None:
                continue
            action, current = result
            payload[f"action_{side}_dobot_arm_joints"] = action
            payload[f"states_{side}_dobot_arm_joints"] = current
        if payload and self.args.execute:
            self.env.step(payload)

    def handle_command(self, command: str) -> bool:
        command = command.strip().lower()
        if command in {"position", "cancel"}:
            self.enter_positioning()
            return True
        if command == "model":
            self.handoff_to_model()
            return True
        if command.startswith("adjust-"):
            parts = command.split("-")
            if len(parts) != 3 or parts[1] not in {"left", "right"}:
                return False
            if parts[2] not in {"on", "off"}:
                return False
            self.set_adjustment(parts[1], parts[2] == "on")
            return True
        return False


class RobotRuntimeEnv:

    @property
    def communication_spec(self):
        """This SDK maps six joints and one gripper per arm; custom runtimes override it."""
        return {"state_dim": 14, "action_dim": 14,
                "state_names": list(STATE_NAMES), "action_names": list(STATE_NAMES)}

    def __init__(self, args: argparse.Namespace):
        configure_path(args.env_src)
        from dobot_environment import Configuration, DobotEnvironment

        self.args = args
        left_home, right_home = configured_home_positions(
            args.left_home,
            args.right_home,
            close=args.close,
        )
        self.config = Configuration(
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            top_camera_id=args.top_camera_id,
            left_camera_id=args.left_camera_id,
            right_camera_id=args.right_camera_id,
            left_gripper_port=args.left_gripper_port,
            right_gripper_port=args.right_gripper_port,
            left_dobot_arm_ip=args.left_dobot_arm_ip,
            right_dobot_arm_ip=args.right_dobot_arm_ip,
            left_telop_arm_port=args.left_telop_arm_port,
            right_telop_arm_port=args.right_telop_arm_port,
            left_dobot_arm_default_home_position=left_home,
            right_dobot_arm_default_home_position=right_home,
            control_frequancy=args.control_frequency,
        )
        self.env = DobotEnvironment(
            self.config,
            sampling_frequancy=args.sampling_frequency,
            max_action_delta_threshold=1e9,  # disable built-in limit; we do our own check
            data_collection_mode=True,
        )
        # One line reader owns stdin for lifecycle, positioning, and action
        # controls.  Multiple readers race and can silently swallow Web commands.
        self.hotkey = HomeHotkey(enabled=False)
        self.positioning = PositioningControl(self.env, args)
        self._run_next_chunk = False
        self._hold_actions = False
        self._cancel_model = threading.Event()
        self._control_channel_lost = threading.Event()
        self._connection_lock = threading.Lock()
        self._active_connection = None
        self.operator_control = OperatorControl(
            enabled=True,
            on_command=self._observe_operator_command,
            on_disconnect=self._handle_operator_disconnect,
        )
        if args.close is not None:
            log("[home] --close right gripper: right Home gripper target=0.000")
        log("Robot environment ready")

    def _observe_operator_command(self, command: str) -> None:
        if command == "cancel":
            self.request_model_cancel()

    def _handle_operator_disconnect(self) -> None:
        self._control_channel_lost.set()
        log("[control] Web control channel closed; cancelling model ownership")
        self.request_model_cancel()

    def request_model_cancel(self) -> None:
        self._cancel_model.set()
        self._hold_actions = False
        self._run_next_chunk = False
        connection_lock = getattr(self, "_connection_lock", None)
        if connection_lock is None:
            connection = None
        else:
            with connection_lock:
                connection = self._active_connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _raise_if_model_cancelled(self) -> None:
        cancel_event = getattr(self, "_cancel_model", None)
        if cancel_event is not None and cancel_event.is_set():
            raise ModelSessionCancelled("model session cancelled by Web control")

    def begin_model_connection(self, connection) -> None:
        self._cancel_model.clear()
        with self._connection_lock:
            self._active_connection = connection

    def end_model_connection(self, connection) -> None:
        with self._connection_lock:
            if self._active_connection is connection:
                self._active_connection = None

    def clear_model_cancel(self) -> None:
        self._cancel_model.clear()

    def close(self) -> None:
        try:
            self.env.terminate()
        except Exception as exc:
            log(f"Failed to terminate robot environment: {exc}")

    def _home_prompt_available(self) -> bool:
        """True when the per-step prompt is the operator's channel for asking to home."""
        return bool(self.args.execute and self.args.confirm_each_action and self.args.home_key)

    def _wait_for_operator(self, prompt: str, *, clear: bool = False) -> str:
        """Read one explicit operator line without competing stdin readers."""
        if self.operator_control.active:
            if clear:
                self.operator_control.clear()
            self._raise_if_model_cancelled()
            return self.operator_control.wait(
                prompt,
                poll=self._raise_if_model_cancelled,
            )
        if clear:
            _flush_stdin()
        self._raise_if_model_cancelled()
        return input(prompt).strip().lower()

    def _drain_running_controls(self) -> None:
        """Apply controls received while a hold or action chunk is in progress."""
        if not self.operator_control.active:
            self._raise_if_model_cancelled()
            return
        while True:
            reply = self.operator_control.get_nowait()
            if reply is None:
                break
            if reply == "s":
                if self._hold_actions:
                    log("[control] hold released; pausing before the next action")
                self._hold_actions = False
            elif reply == "r":
                if not self._hold_actions:
                    log("[control] hold started; actions continue until Enter is released")
                self._hold_actions = True
            elif reply == "cancel":
                self.request_model_cancel()
            elif self.args.home_key and reply == "h":
                log("[control] home requested; stopping hold at the next step boundary")
                self._hold_actions = False
                self.hotkey.requested.set()
            elif reply:
                log(f"[control] ignored {reply!r} while actions are already running")
        self._raise_if_model_cancelled()

    def clear_held_actions(self) -> None:
        """Do not carry a held-key session across a disconnected policy client."""
        self._hold_actions = False
        if not self.operator_control.active:
            return
        while self.operator_control.get_nowait() is not None:
            pass

    def process_positioning_commands(self) -> None:
        """Drain lifecycle commands while no model client owns the robot."""
        if not self.operator_control.active:
            return
        while True:
            command = self.operator_control.get_nowait()
            if command is None:
                return
            try:
                if self.positioning.handle_command(command):
                    continue
                if command == "h" and self.args.home_key:
                    self.positioning.begin_home()
                    try:
                        self.go_home()
                    finally:
                        self.positioning.complete_home()
                    continue
                if command:
                    log(f"[control] ignored {command!r} while waiting for positioning")
            except Exception as exc:
                log(f"[control] command {command!r} failed: {exc}")
                self.positioning.report_error(exc)

    def process_positioning_tick(self) -> None:
        try:
            self.positioning.process_tick()
        except Exception as exc:
            log(f"[positioning] tick failed: {exc}")
            self.positioning.report_error(exc)

    def request_action_chunk(self) -> None:
        """Run the next received action chunk without per-action prompts."""
        self._run_next_chunk = True

    def _start_chunk_hotkey(self) -> bool:
        """Temporarily enable `h` while a confirmed chunk is running."""
        # The line-control reader already handles h at every action boundary;
        # starting HomeHotkey too would create a second stdin reader.
        if self.operator_control.active:
            return False
        if not (self.args.confirm_each_action and self.args.home_key):
            return False
        self.hotkey.stop()
        self.hotkey = HomeHotkey(enabled=True)
        self.hotkey.start()
        return True

    def _restore_step_hotkey(self) -> None:
        if not self.args.confirm_each_action:
            return
        self.hotkey.stop()
        self.hotkey = HomeHotkey(enabled=False)

    def go_home(self) -> None:
        """Send both arms to the configured startup Home positions.

        move_to_position_smoothly() commands both grippers to their home value in one
        un-interpolated step before it starts interpolating the joints. ``--close right
        gripper`` explicitly makes the right-hand Home target closed instead of its
        configured open value.
        """
        if not self.args.execute:
            log("[home] dry-run: not sending robot commands")
            return
        left_home = self.config.left_dobot_arm_default_home_position
        right_home = self.config.right_dobot_arm_default_home_position
        log(f"[home] returning to home over {self.args.home_steps} steps")
        log(f"[home]   grippers snap NOW to left={left_home[6]:.3f} right={right_home[6]:.3f} "
            f"before the arms retract")
        self.env.move_to_position_smoothly(left_home, right_home, steps=self.args.home_steps)
        log("[home] arms are at home")

    def handle_home_request(self) -> None:
        """Home the arms, then hold until the operator decides to carry on.

        Everything after the flag is cleared runs under try/finally: a fault inside
        go_home() must still drop the tty out of cbreak, still stop the operator to
        say so, and still leave the hotkey armed for another try.
        """
        self.hotkey.requested.clear()
        self._hold_actions = False
        self.positioning.begin_home()
        failure: Exception | None = None
        resume_model = False
        try:
            self.go_home()
        except Exception as exc:  # noqa: BLE001 - operator must be told, not the traceback
            failure = exc
            log(f"[home] *** GO-HOME FAILED: {exc} — arms may be stopped mid-motion ***")
        finally:
            # Drop out of cbreak so the prompt below behaves like a normal line read.
            self.hotkey.stop()
            try:
                # Clear input before announcing that Home has completed.  Otherwise
                # a queued Web resume could be consumed while the arms are moving.
                if self.operator_control.active:
                    self.operator_control.clear()
                else:
                    _flush_stdin()
                self._raise_if_model_cancelled()
                self.positioning.wait_for_home_resume()
                prompt = (
                    "[home] GO-HOME FAILED. Press Enter to resume anyway, Ctrl+C to stop ..."
                    if failure is not None
                    else "[home] at home. Press Enter to resume the policy, Ctrl+C to stop ..."
                )
                while True:
                    reply = self._wait_for_operator(prompt)
                    if reply == "cancel":
                        self.request_model_cancel()
                        self._raise_if_model_cancelled()
                    if reply in {"", "resume-home"}:
                        resume_model = True
                        break
                    log(
                        f"[home] ignored {reply!r}; "
                        "waiting for Enter or explicit resume-home"
                    )
            finally:
                self.hotkey = HomeHotkey(
                    enabled=self.args.home_key and not self.args.confirm_each_action
                )
                self.hotkey.start()
                if resume_model and not self._cancel_model.is_set():
                    self.positioning.resume_model_after_home()

    def move_home_if_requested(self) -> None:
        if not self.args.home_on_start:
            return
        if not self.args.execute:
            log("home-on-start requested, but --execute is off; not moving arms")
            return
        self._wait_for_operator(
            "Press Enter to move robot to home position (Ctrl+C to stop)...", clear=True
        )
        self.env.move_to_position_smoothly(
            self.config.left_dobot_arm_default_home_position,
            self.config.right_dobot_arm_default_home_position,
            steps=self.args.home_steps,
        )

    def capture_observation(self, cursor: int) -> Dict[str, Any]:
        obs = self.env.get_observation()
        return self.observation_from_raw(obs, cursor)

    def observation_from_raw(self, obs, cursor: int) -> Dict[str, Any]:
        """Format one SDK snapshot; optional transports can reuse its camera geometry."""
        self.positioning.observe_physical_buttons(obs)
        left_state  = np.asarray(obs["left_dobot_arm_joints"],  dtype=np.float32)
        right_state = np.asarray(obs["right_dobot_arm_joints"], dtype=np.float32)
        if left_state.shape[0] < 7 or right_state.shape[0] < 7:
            raise RuntimeError(
                f"Expected ≥7-D arm states, got left={left_state.shape}, right={right_state.shape}"
            )

        top_crop = crop_xywh(obs["top_camera_frame"], self.args.top_roi_xywh)
        images = {
            "top_camera":   resize_image_to_square_rgb(top_crop,                    self.args.image_size).astype(np.uint8, copy=False),
            "left_camera":  resize_image_to_square_rgb(obs["left_camera_frame"],  self.args.image_size).astype(np.uint8, copy=False),
            "right_camera": resize_image_to_square_rgb(obs["right_camera_frame"], self.args.image_size).astype(np.uint8, copy=False),
        }
        return {
            "type": "observation",
            "protocol": PROTOCOL_VERSION,
            "cursor": cursor,
            "task": self.args.task,
            "state": np.concatenate([left_state[:7], right_state[:7]]).astype(np.float32),
            "images": images,
            "timestamps": {"sent": time.time()},
        }

    def accept_action_reply(self, reply, observation) -> None:
        """Hook for transports that acknowledge observations before executing actions."""

    def exchange_observation(self, connection, observation):
        """Send one observation and receive a reply; variants may upload data first."""
        cursor = observation["cursor"]
        payload_bytes = sum(img.nbytes for img in observation["images"].values()) + observation["state"].nbytes
        log(f"[robot] sending observation cursor={cursor} raw_bytes={payload_bytes}")
        started = time.perf_counter()
        send_message(connection, observation)
        sent = time.perf_counter()
        log(f"[robot] waiting actions cursor={cursor} send_ms={(sent - started) * 1000:.1f}")
        reply = recv_message(connection)
        log(f"[robot] received actions cursor={cursor} wait_ms={(time.perf_counter() - sent) * 1000:.1f}")
        return reply

    def execute_action_chunk(self, actions: np.ndarray) -> int:
        if self.positioning.state != "model_control":
            raise RuntimeError(
                f"model actions rejected while control state is {self.positioning.state}"
            )
        self._raise_if_model_cancelled()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected (N,14) actions, got {actions.shape}")
        steps = min(actions.shape[0], self.args.actions_per_response)

        # read actual position ONCE before the chunk — step 0 delta is vs. real position,
        # steps 1+ delta is vs. the previous commanded action (consecutive-step difference).
        obs = self.env.get_observation()
        prev_left  = np.asarray(obs["left_dobot_arm_joints"],  dtype=np.float32)[:6]
        prev_right = np.asarray(obs["right_dobot_arm_joints"], dtype=np.float32)[:6]
        run_chunk = self._run_next_chunk
        self._run_next_chunk = False
        chunk_hotkey_active = self._start_chunk_hotkey() if run_chunk else False

        try:
            for step_idx, action in enumerate(actions[:steps]):
                if self.operator_control.active:
                    self._drain_running_controls()
                self._raise_if_model_cancelled()
                if self.hotkey.requested.is_set():
                    log(f"[home] hotkey honoured: abandoning chunk at step {step_idx}/{steps}")
                    self.handle_home_request()
                    return step_idx
                requested_before = self.hotkey.requested.is_set()
                prev_left, prev_right, run_chunk = self._execute_one_action(
                    action, step_idx, prev_left, prev_right, run_chunk
                )
                if run_chunk and not chunk_hotkey_active:
                    chunk_hotkey_active = self._start_chunk_hotkey()
                if self.hotkey.requested.is_set():
                    # If the flag was raised by THIS step's prompt, the action was skipped,
                    # so it must not be counted as executed.
                    executed = step_idx if not requested_before else step_idx + 1
                    log(f"[home] hotkey honoured: abandoning chunk after step {step_idx}")
                    self.handle_home_request()
                    return executed
                time.sleep(self.args.step_dt)
            if self.args.settle_dt > 0:
                log(f"[robot] waiting {self.args.settle_dt}s for arm to settle before next capture")
                time.sleep(self.args.settle_dt)
            return steps
        finally:
            if chunk_hotkey_active:
                self._restore_step_hotkey()

    def _execute_one_action(
        self,
        action: np.ndarray,
        step_idx: int,
        prev_left: np.ndarray,
        prev_right: np.ndarray,
        run_chunk: bool,
    ):
        left_action  = np.asarray(action[:7],  dtype=np.float32).copy()
        right_action = np.asarray(action[7:14], dtype=np.float32).copy()
        left_action[6]  = clamp01(left_action[6])
        right_action[6] = clamp01(right_action[6])

        left_delta  = np.abs(left_action[:6]  - prev_left)
        right_delta = np.abs(right_action[:6] - prev_right)
        max_delta   = max(float(left_delta.max()), float(right_delta.max()))

        log(f"── step {step_idx:02d} ──────────────────────────────────")
        log(f"  left  delta : {np.array2string(left_delta,  precision=4, suppress_small=True)}  max={left_delta.max():.4f}")
        log(f"  right delta : {np.array2string(right_delta, precision=4, suppress_small=True)}  max={right_delta.max():.4f}")
        log(f"  left  target: {np.array2string(left_action[:6],  precision=4, suppress_small=True)}  gripper={left_action[6]:.3f}")
        log(f"  right target: {np.array2string(right_action[:6], precision=4, suppress_small=True)}  gripper={right_action[6]:.3f}")
        log(f"  joint max_delta={max_delta:.4f}  limit={self.args.max_joint_delta}")

        if self.args.max_joint_delta > 0 and max_delta > self.args.max_joint_delta:
            log(f"  *** BLOCKED: joint delta {max_delta:.4f} > limit {self.args.max_joint_delta} (grippers excluded) ***")
            # A blocked chunk used to return with no prompt at all, which left the
            # operator no way to ask for home during a run of blocked steps — exactly
            # the situation where they most want out. Offer the choice here too.
            if self._home_prompt_available() and not run_chunk and not self._hold_actions:
                reply = self._wait_for_operator(
                    "  BLOCKED. Press Enter to skip, 'c'+Enter to run the remaining chunk, "
                    "'h'+Enter to go home ..."
                )
                if reply == "h":
                    self.hotkey.requested.set()
                elif reply == "c":
                    run_chunk = True
                elif reply == "r":
                    self._hold_actions = True
            # return previous reference unchanged so the next step still compares correctly
            return prev_left, prev_right, run_chunk

        if self.args.execute and self.args.confirm_each_action and not run_chunk:
            if self._hold_actions:
                self._drain_running_controls()
            if self.hotkey.requested.is_set():
                return prev_left, prev_right, run_chunk
            hint = ", 'h'+Enter to go home" if self.args.home_key else ""
            while not self._hold_actions:
                reply = self._wait_for_operator(
                    f"  Press Enter to execute one action, 'c'+Enter to run the remaining chunk{hint}, "
                    "Ctrl+C to abort ..."
                )
                if reply == "cancel":
                    self.request_model_cancel()
                    self._raise_if_model_cancelled()
                if self.args.home_key and reply == "h":
                    self.hotkey.requested.set()
                    # Skip this action; execute_action_chunk sees the flag and homes.
                    return prev_left, prev_right, run_chunk
                if reply == "c":
                    run_chunk = True
                    break
                if reply == "r":
                    self._hold_actions = True
                    log("[control] hold started; actions continue until Enter is released")
                    break
                if reply == "s":
                    log("[control] hold is already stopped; waiting for an action command")
                    continue
                if reply:
                    log(f"[control] unknown action command {reply!r}; waiting again")
                    continue
                break

        if not self.args.execute:
            log("dry-run: not sending robot commands")
            return left_action[:6], right_action[:6], run_chunk

        self.env.step({
            "action_left_dobot_arm_joints":   left_action,
            "states_left_dobot_arm_joints":   prev_left,
            "action_right_dobot_arm_joints":  right_action,
            "states_right_dobot_arm_joints":  prev_right,
        })
        return left_action[:6], right_action[:6], run_chunk


def _run_robot_server_locked(args: argparse.Namespace, *, runtime_factory=None) -> None:
    configured_key()  # Reject missing authentication before opening hardware.
    runtime = (runtime_factory or RobotRuntimeEnv)(args)
    try:
        communication_spec = getattr(runtime, "communication_spec", {"state_dim": 14, "action_dim": 14})
        dimensions = CommunicationSpec.from_metadata(communication_spec)
        runtime.operator_control.start()
        runtime.move_home_if_requested()
        runtime.hotkey.start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((args.host, args.port))
            server_sock.listen(1)
            server_sock.setblocking(False)
            log(
                f"Robot server listening on {args.host}:{args.port}. "
                f"execute={args.execute}, max_joint_delta={args.max_joint_delta}, "
                f"top_roi_xywh={args.top_roi_xywh}"
            )
            log(MODEL_HANDOFF_HINT)
            while True:
                if not runtime.operator_control.active:
                    raise RuntimeError("Web control channel closed")
                runtime.process_positioning_commands()
                runtime.process_positioning_tick()
                try:
                    conn, addr = server_sock.accept()
                except BlockingIOError:
                    time.sleep(1.0 / max(1, args.control_frequency))
                    continue
                if runtime.positioning.state != "model_waiting":
                    log(
                        f"Rejected client {addr}: model handoff not ready "
                        f"(state={runtime.positioning.state})"
                    )
                    log(MODEL_HANDOFF_HINT)
                    conn.close()
                    continue
                log(f"Client connected from {addr}")
                try:
                    conn = authenticate(conn, server=True)
                    runtime.begin_model_connection(conn)
                    runtime.positioning.model_connected()
                    with conn:
                        send_message(conn, {
                            "type": "hello",
                            "protocol": PROTOCOL_VERSION,
                            "image_size": args.image_size,
                            "task": args.task,
                            **communication_spec,
                        })
                        cursor = 0
                        while True:
                            if args.max_steps > 0 and cursor >= args.max_steps:
                                log(f"Reached max_steps={args.max_steps}; closing session")
                                break
                            log(f"[robot] capturing observation cursor={cursor}")
                            obs = runtime.capture_observation(cursor)
                            dimensions.validate_state(obs["state"])
                            reply = runtime.exchange_observation(conn, obs)
                            if reply.get("type") != "actions":
                                raise RuntimeError(f"Unexpected reply keys: {reply.keys()}")
                            dimensions.validate_actions(reply["actions"])
                            runtime.accept_action_reply(reply, obs)
                            inference_ms = reply.get("inference_ms")
                            if inference_ms is not None:
                                log(f"inference_ms={float(inference_ms):.1f}")

                            executed = runtime.execute_action_chunk(reply["actions"])
                            cursor += executed
                except ModelSessionCancelled as exc:
                    log(f"Session cancelled (client {addr}): {exc}")
                except Exception as exc:
                    log(f"Session error (client {addr}): {exc}")
                finally:
                    conn.close()
                    runtime.end_model_connection(conn)
                    runtime.clear_held_actions()
                    runtime.positioning.model_disconnected()
                    runtime.clear_model_cancel()
                    log(MODEL_HANDOFF_HINT)
    except KeyboardInterrupt:
        log("Interrupted")
    finally:
        runtime.operator_control.stop()
        runtime.hotkey.stop()  # restore the terminal before anything else
        runtime.positioning.enter_positioning()
        runtime.close()


def run_robot_server(args: argparse.Namespace, *, runtime_factory=None) -> None:
    lease = RobotHardwareLease(args.hardware_lock_path)
    lease.acquire()
    try:
        _run_robot_server_locked(args, runtime_factory=runtime_factory)
    finally:
        lease.release()


# ─────────────────────────── openpi client ────────────────────────────

def _hwc_to_chw(img: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB → (3,H,W) uint8. CHW passthrough."""
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        return np.ascontiguousarray(img.transpose(2, 0, 1))
    return np.ascontiguousarray(img)


def build_openpi_obs(message: Dict[str, Any], task: str, *, state_dim: int | None = None) -> Dict[str, Any]:
    """Convert robot-server TCP message → openpi policy.infer() input dict.

    Images arrive as (H,W,3) uint8 RGB (resize_image_to_square_rgb flips BGR→RGB).
    BoxDualArmInputs expects CHW uint8.
    """
    state = np.asarray(message["state"], dtype=np.float32)
    if (state.ndim != 1 or not state.size or not np.isfinite(state).all()
            or (state_dim is not None and state.shape != (state_dim,))):
        raise ValueError(f"Expected finite state shape ({state_dim or 'D'},), got {state.shape}")
    images = message["images"]
    return {
        "state": state,
        "images": {
            "top_camera":   _hwc_to_chw(images["top_camera"]),
            "left_camera":  _hwc_to_chw(images["left_camera"]),
            "right_camera": _hwc_to_chw(images["right_camera"]),
        },
        "prompt": task,
    }


def warmup_openpi_policy(policy, task: str, *, use_history: bool = False, state_dim: int = 14) -> None:
    """Compile using the bridge wire format, then discard any warmup history."""
    log("[openpi-client] warming up policy server...")
    policy.infer({
        "state": np.zeros(positive_dimension(state_dim, "state_dim"), dtype=np.float32),
        "images": {
            name: np.zeros((3, 224, 224), dtype=np.uint8)
            for name in IMAGE_NAMES
        },
        "prompt": task,
    })
    if use_history:
        reply = policy.infer({"reset": True})
        if reply.get("reset_finished") is not True:
            raise RuntimeError("Memory server did not acknowledge the warmup history reset.")
        log("[openpi-client] warmup history cleared; task history starts with the first observation")
        log("[openpi-client] history contains inference observations only; this bridge does not upload intermediate frames")
    log("[openpi-client] warmup done")


def run_openpi_client(
    args: argparse.Namespace, *, observation_handler=None, warmup_handler=None, message_handler=None
) -> None:
    health_server = ClientHealthServer(
        args.health_host,
        args.health_port,
        args.health_token,
        request_timeout=args.health_timeout,
    )
    policy = None
    try:
        health_server.start()
        from dobot_nexus.inference.transport.client import WebsocketPolicyClient

        log(f"Connecting to openpi policy server ws://{args.policy_host}:{args.policy_port}")
        policy = WebsocketPolicyClient(host=args.policy_host, port=args.policy_port,
                                       connect_timeout=args.connect_timeout)
        metadata = policy.get_server_metadata()
        log(f"Policy server metadata: {metadata}")
        dimensions = CommunicationSpec.from_metadata(metadata)
        for name in ("state_dim", "action_dim"):
            expected = getattr(args, name, None)
            if expected is not None and positive_dimension(expected, name) != getattr(dimensions, name):
                raise ValueError(f"Configured {name}={expected} does not match policy {getattr(dimensions, name)}.")
        if metadata.get("history_requires_dense_frames") and observation_handler is None:
            raise RuntimeError(
                "This Memory policy requires dense history. Set BRIDGE_MODE=memory "
                "on both start_dobot_dual_arm_14d_server.sh and start_robot_client.sh."
            )

        (warmup_handler or warmup_openpi_policy)(
            policy,
            args.task or "Put the cube into box.",
            use_history=bool(metadata.get("use_history")),
            state_dim=dimensions.state_dim,
        )

        with socket.create_connection(
            (args.robot_host, args.robot_port),
            timeout=args.connect_timeout,
        ) as sock:
            sock = authenticate(sock, server=False)
            log(f"Connected to robot server {args.robot_host}:{args.robot_port}")
            hello = recv_message(sock)
            if hello.get("protocol") != PROTOCOL_VERSION:
                raise RuntimeError(f"Protocol mismatch: {hello}")
            dimensions.validate_robot(hello)
            health_server.set_state("ready")
            task = (
                args.task
                if args.task is not None
                else hello.get("task", "Put the cube into box.")
            )
            log(f"Robot task={task!r}, image_size={hello.get('image_size')}")

            while True:
                log("[openpi-client] waiting observation")
                try:
                    message = recv_message(sock, wait_for_message=True)
                except ConnectionError as exc:
                    log(f"[openpi-client] robot server closed: {exc}")
                    break

                msg_type = message.get("type")
                if msg_type == "error":
                    raise RuntimeError(message.get("message", "robot server error"))
                if msg_type != "observation" and message_handler is not None:
                    reply = message_handler(policy, message, task, metadata)
                    send_message(sock, reply)
                    continue
                if msg_type != "observation":
                    raise RuntimeError(f"Unexpected message type: {msg_type}")

                cursor = int(message["cursor"])
                observation = build_openpi_obs(message, task, state_dim=dimensions.state_dim)
                log(f"[openpi-client] received observation cursor={cursor}")

                log(f"[openpi-client] calling policy.infer cursor={cursor}")
                t0 = time.perf_counter()
                result = (
                    policy.infer(observation)
                    if observation_handler is None
                    else observation_handler(policy, message, task, metadata)
                )
                inference_ms = (time.perf_counter() - t0) * 1000

                actions = dimensions.validate_actions(result["actions"])
                actions = actions[: args.actions_per_response]
                log(f"cursor={cursor} actions={actions.shape} inference_ms={inference_ms:.1f}")

                send_message(sock, {
                    "type": "actions",
                    "protocol": PROTOCOL_VERSION,
                    "cursor": cursor,
                    "actions": actions,
                    "inference_ms": inference_ms,
                    **({"history_ack": result["history_ack"]} if "history_ack" in result else {}),
                })
    finally:
        try:
            if policy is not None:
                policy.close()
        finally:
            health_server.close()


# ─────────────────────────── arg parsing ──────────────────────────────

def _nonempty_health_token(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("health token must not be empty")
    return value


def add_robot_server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=12006)
    p.add_argument(
        "--hardware-lock-path",
        default="/tmp/yechen-robot-hardware.lock",
    )
    p.add_argument("--env-src", default="/data/yechen",
                   help="path prepended to sys.path to find dobot_environment")
    p.add_argument("--execute", action="store_true")

    confirm_grp = p.add_mutually_exclusive_group()
    confirm_grp.add_argument("--confirm-each-action",    dest="confirm_each_action", action="store_true",  default=True)
    confirm_grp.add_argument("--no-confirm-each-action", dest="confirm_each_action", action="store_false")

    arm_grp = p.add_mutually_exclusive_group()
    arm_grp.add_argument("--arm-execution-confirm",    dest="arm_execution_confirm", action="store_true",  default=True)
    arm_grp.add_argument("--no-arm-execution-confirm", dest="arm_execution_confirm", action="store_false")

    p.add_argument("--home-on-start",  action="store_true")
    p.add_argument("--home-steps",     type=int,   default=50)

    homekey_grp = p.add_mutually_exclusive_group()
    homekey_grp.add_argument("--home-key",    dest="home_key", action="store_true",  default=True,
                             help="press 'h' during a run to return to the startup home position")
    homekey_grp.add_argument("--no-home-key", dest="home_key", action="store_false")
    p.add_argument("--step-dt",        type=float, default=0.05)
    p.add_argument("--settle-dt",      type=float, default=0.0,
                   help="extra wait after last action in chunk before capturing next observation")
    p.add_argument("--max-joint-delta",      type=float, default=0.30)
    p.add_argument("--actions-per-response", type=int,   default=20)
    p.add_argument("--max-steps",      type=int,   default=0, help="0 = run until interrupted")
    p.add_argument("--sampling-frequency",  type=float, default=200)
    p.add_argument("--control-frequency",   type=int,   default=20)
    p.add_argument("--camera-width",   type=int, default=640)
    p.add_argument("--camera-height",  type=int, default=480)
    p.add_argument("--camera-fps",     type=int, default=30)
    p.add_argument("--top-camera-id",    default="130322272591")
    p.add_argument("--left-camera-id",   default="130322272429")
    p.add_argument("--right-camera-id",  default="218622275674")
    p.add_argument("--left-gripper-port",   default="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A97AN322-if00-port0")
    p.add_argument("--right-gripper-port",  default="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG031XGI-if00-port0")
    p.add_argument("--left-dobot-arm-ip",   default="192.168.5.1")
    p.add_argument("--right-dobot-arm-ip",  default="192.168.5.2")
    p.add_argument("--left-telop-arm-port",  default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B42009619-if00")
    p.add_argument("--right-telop-arm-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B42009414-if00")
    p.add_argument("--left-home",  type=float, nargs="+",
                   default=[-1.5707963, 0, -1.5707963, 0,  1.5707963,  1.5707963, 1.0])
    p.add_argument("--right-home", type=float, nargs="+",
                   default=[ 1.5707963, 0,  1.5707963, 0, -1.5707963, -1.5707963, 1.0])
    p.add_argument(
        "--close",
        nargs=2,
        metavar=("SIDE", "DEVICE"),
        default=None,
        help="override a Home gripper target; currently only '--close right gripper' is supported",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--top-roi-xywh", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   default=[145, 30, 400, 400],
                   help="top camera crop: top-left (x, y) + width + height")
    p.add_argument("--no-top-crop", dest="top_roi_xywh", action="store_const", const=None,
                   help="use the full top-camera frame before resize")
    p.add_argument("--task", default="Put the cube into box.")


def add_openpi_client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--state-dim", type=int, help="expected policy state width; defaults to model metadata")
    p.add_argument("--action-dim", type=int, help="expected policy action width; defaults to model metadata")
    p.add_argument("--robot-host",  default="10.40.1.198")
    p.add_argument("--robot-port",  type=int, default=12006)
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--actions-per-response", type=int, default=20,
                   help="how many steps from the action chunk to send back per obs")
    p.add_argument("--connect-timeout", type=float, default=20.0)
    p.add_argument("--health-host", default="127.0.0.1")
    p.add_argument("--health-port", type=int, default=12007)
    p.add_argument("--health-token", required=False,default="task", type=_nonempty_health_token)
    p.add_argument("--health-timeout", type=float, default=0.2)
    p.add_argument("--task", default=None,
                   help="override task prompt; None uses whatever robot server sends")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    add_robot_server_args(sub.add_parser("robot-server"))
    add_openpi_client_args(sub.add_parser("openpi-client"))
    args = parser.parse_args(list(argv))
    if args.mode == "robot-server" and args.close is not None:
        close = tuple(part.strip().lower() for part in args.close)
        if close != ("right", "gripper"):
            parser.error("--close currently only supports: --close right gripper")
        args.close = close
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    np.set_printoptions(3, suppress=True, linewidth=150)
    if args.mode == "robot-server":
        run_robot_server(args)
    elif args.mode == "openpi-client":
        run_openpi_client(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
