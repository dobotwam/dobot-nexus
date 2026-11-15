#!/usr/bin/env python3
"""Dense-history extension of the standard robot bridge.

Run both robot-server and openpi-client with this entry point. It shares the
ordinary script's SDK setup, action execution, operator controls and transports.
Only unacknowledged top-camera frames live on the robot; the existing policy
server owns feature encoding and long-term memory. RGB transport is lossless.
History uses temporal-delta compression and per-chunk policy acknowledgements;
the robot releases each accepted chunk before action inference starts.

Home starts a new trial by default (--memory-home-mode continue preserves it).
Copy the complete pi05_real_robot directory to the robot machine.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np

if __package__:
    from . import dobot_dual_arm_14d as base
    from .history import (
        HISTORY_CODEC, HISTORY_PROTOCOL, MAX_HISTORY_CHUNK, ObservationHistory,
        SerializedEnvironment, decode_history_frames, encode_history_frames,
    )
else:
    import dobot_dual_arm_14d as base
    from history import (
        HISTORY_CODEC, HISTORY_PROTOCOL, MAX_HISTORY_CHUNK, ObservationHistory,
        SerializedEnvironment, decode_history_frames, encode_history_frames,
    )


class MemoryRobotRuntime(base.RobotRuntimeEnv):
    # Protocol 2 converts arbitrary bytes through Unicode and can hold the GIL
    # long enough to lose camera periods for a dense RGB batch. Both peers are
    # Python 3; protocol 4 keeps bytes binary with the same TCP framing/reader.

    def __init__(self, args):
        super().__init__(args)
        self.env = SerializedEnvironment(self.env)
        self.positioning.env = self.env
        self.history = None

    def _history_image(self, observation):
        crop = base.crop_xywh(observation["top_camera_frame"], self.args.top_roi_xywh)
        return base.resize_image_to_square_rgb(crop, self.args.image_size)

    def _start_history(self):
        self.history = ObservationHistory(
            self.env.get_observation, self._history_image,
            fps=self.args.history_fps, max_pending=self.args.history_max_pending,
            max_gap_s=self.args.history_max_gap,
            on_error=self.request_model_cancel,
        )
        self.history.start()
        base.log(f"[memory] new episode={self.history.episode_id} fps={self.history.fps:g}")

    def _stop_history(self):
        if self.history is not None:
            self.history.close()
            self.history = None

    def begin_model_connection(self, connection):
        super().begin_model_connection(connection)
        self._stop_history()

    def end_model_connection(self, connection):
        try:
            self._stop_history()
        finally:
            super().end_model_connection(connection)

    def _raise_if_model_cancelled(self):
        if getattr(self, "history", None) is not None:
            self.history.check()
        super()._raise_if_model_cancelled()

    def capture_observation(self, cursor):
        if self.history is None:
            self._start_history()
        raw, packet = self.history.snapshot()
        message = self.observation_from_raw(raw, cursor)
        # Both current image and last history entry come from the same SDK read.
        message["images"]["top_camera"] = packet["frames"][-1]
        message["history"] = packet
        capture = self.history.capture_status()
        base.log(f"[memory] prepared frames={packet['step_indices'][0]}..{packet['frame_index']} "
                 f"count={len(packet['frames'])} action_cursor={cursor}")
        base.log(f"[memory] capture reads={capture['reads']} grid_frames={capture['slots']} "
                 f"observed_fps={capture['observed_fps']:.2f} target_fps={self.history.fps:g} "
                 f"repeated_slots={capture['repeated_slots']} max_source_gap_s={capture['max_source_gap_s']:.4f}")
        return message

    def exchange_observation(self, connection, observation):
        """Upload bounded chunks with policy ACKs before requesting actions."""
        packet = observation["history"]
        batch_size = self.args.history_batch_frames
        upload_started = time.perf_counter()
        total_wire_bytes = 0
        for start in range(0, len(packet["frames"]), batch_size):
            self._raise_if_model_cancelled()
            frames = packet["frames"][start:start + batch_size]
            indices = packet["step_indices"][start:start + batch_size]
            started = time.perf_counter()
            blobs = encode_history_frames(frames)
            encoded = time.perf_counter()
            chunk = {**packet, "frame_index": indices[-1], "step_indices": indices,
                     "timestamps": packet["timestamps"][start:start + batch_size],
                     "codec": HISTORY_CODEC, "compressed_frames": blobs}
            if "capture_timestamps" in packet:
                chunk["capture_timestamps"] = packet["capture_timestamps"][start:start + batch_size]
            chunk.pop("frames")
            state = self.history.status()
            base.log(f"[memory] sending history frames={indices[0]}..{indices[-1]} "
                     f"raw_bytes={sum(f.nbytes for f in frames)} compressed_bytes={sum(map(len, blobs))} "
                     f"pending={state['pending']}/{state['capacity']}")
            wire_bytes = base.send_message(connection, {
                "type": "history", "protocol": base.PROTOCOL_VERSION,
                "cursor": observation["cursor"], "history": chunk,
            })
            total_wire_bytes += wire_bytes
            sent = time.perf_counter()
            reply = base.recv_message(connection)
            expected = {"episode_id": packet["episode_id"], "frame_index": indices[-1]}
            if (reply.get("type") != "history_ack" or reply.get("cursor") != observation["cursor"]
                    or reply.get("history_ack") != expected):
                raise ValueError("Missing/mismatched chunk history acknowledgement; update both Memory scripts.")
            self.history.acknowledge(**expected)
            state = self.history.status()
            base.log(f"[memory] acknowledged frame={indices[-1]} wire_bytes={wire_bytes} "
                     f"compress_ms={(encoded - started) * 1000:.1f} send_ms={(sent - encoded) * 1000:.1f} "
                     f"ack_wait_ms={(time.perf_counter() - sent) * 1000:.1f} "
                     f"pending={state['pending']}/{state['capacity']}")
        base.log(f"[memory] history ready frame={packet['frame_index']} frames={len(packet['frames'])} "
                 f"wire_bytes={total_wire_bytes} total_ms={(time.perf_counter() - upload_started) * 1000:.1f}")
        # Images and history cutoff are the original snapshot, even if capture
        # continued during upload. Never include future frames in this action.
        observation["history"] = {
            "protocol": HISTORY_PROTOCOL, "episode_id": packet["episode_id"],
            "frame_index": packet["frame_index"], "fps": packet["fps"], "uploaded": True,
        }
        return super().exchange_observation(connection, observation)

    def accept_action_reply(self, reply, observation):
        packet = observation["history"]
        expected = {"episode_id": packet["episode_id"], "frame_index": packet["frame_index"]}
        if reply.get("cursor") != observation["cursor"] or reply.get("history_ack") != expected:
            raise ValueError("Missing/mismatched history acknowledgement; use the _memory client.")
        if packet.get("uploaded"):
            if self.history.status()["acked"] != packet["frame_index"]:
                raise ValueError("Actions arrived before the full history was acknowledged.")
        else:
            self.history.acknowledge(**expected)

    def handle_home_request(self):
        if self.args.memory_home_mode == "reset":
            # Stop the old producer before clearing; the next capture starts frame 0.
            self._stop_history()
            base.log("[memory] Home ends this trial; next observation starts a new episode")
        super().handle_home_request()

    def close(self):
        self._stop_history()
        super().close()


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return int(value)


def validate_base_script():
    """Reject mixed script versions before parsing arguments or opening hardware."""
    missing = []
    for name, parameters in (
        ("run_robot_server", ("runtime_factory",)),
        ("run_openpi_client", ("observation_handler", "warmup_handler", "message_handler")),
        ("send_message", ()),
        ("warmup_openpi_policy", ("use_history",)),
    ):
        function = getattr(base, name, None)
        if not callable(function):
            missing.append(name)
            continue
        signature = inspect.signature(function)
        missing.extend(f"{name}({parameter}=...)" for parameter in parameters
                       if parameter not in signature.parameters)
    for name in (
        "observation_from_raw", "accept_action_reply", "begin_model_connection",
        "end_model_connection", "request_model_cancel", "_raise_if_model_cancelled",
        "exchange_observation",
    ):
        if not callable(getattr(base.RobotRuntimeEnv, name, None)):
            missing.append(f"RobotRuntimeEnv.{name}")
    if missing:
        loaded = Path(base.__file__).resolve()
        raise RuntimeError(
            f"Incompatible base script loaded from {loaded}. Missing: {', '.join(missing)}. "
            "Copy the complete pi05_real_robot directory from the same Nexus checkout; "
            "_runtime/dobot_dual_arm_14d.py, memory.py and history.py must match."
        )


class MemoryBridge:
    """Translate dense robot observations into existing policy control messages."""

    def __init__(self, *, history_fps=30.0, max_gap_s=0.1):
        self.history_fps = history_fps
        self.max_gap_s = max_gap_s
        self.episode_id = None
        self.last_frame = -1
        self.last_timestamp = None
        self.prompt = None
        self.failed = False
        self.last_image = None
        self.last_cursor = None
        self.awaiting_observation = False
        self.last_capture_timestamp = None
        self.grid_origin = None
        self.sampling = None

    @staticmethod
    def warmup(policy, task, *, use_history=False, state_dim=14):
        # Reuse the wire image/state conversion without the old sparse-history notice.
        base.warmup_openpi_policy(policy, task, use_history=False, state_dim=state_dim)
        if use_history:
            if policy.infer({"reset": True}).get("reset_finished") is not True:
                raise RuntimeError("Policy did not acknowledge warmup history reset.")
            base.log("[memory] warmup cleared; dense history begins with the robot episode")

    def __call__(self, policy, message, task, metadata):
        if self.failed:
            raise RuntimeError("History request failed; reconnect to start a new trial.")
        try:
            return self._infer(policy, message, task, metadata)
        except Exception:
            # No automatic replay after an uncertain server write/failed inference.
            self.failed = True
            raise

    def _infer(self, policy, message, task, metadata):
        use_history = bool(metadata.get("use_history"))
        packet = message.get("history")
        dimensions = base.CommunicationSpec.from_metadata(metadata)
        observation = base.build_openpi_obs(message, task, state_dim=dimensions.state_dim)
        if packet is None and not use_history:
            result = policy.infer(observation)
            dimensions.validate_actions(result["actions"])
            return result
        if not isinstance(packet, dict) or packet.get("protocol") != HISTORY_PROTOCOL:
            raise ValueError("Dense history is missing; run robot-server with the _memory script.")
        current_image = observation["images"]["top_camera"].transpose(1, 2, 0)
        if packet.get("uploaded"):
            if (packet["episode_id"] != self.episode_id or packet["frame_index"] != self.last_frame
                    or task != self.prompt or message["cursor"] != self.last_cursor
                    or not self.awaiting_observation or packet["fps"] != self.history_fps):
                raise ValueError("Observation does not match acknowledged history.")
            if not np.array_equal(current_image, self.last_image):
                raise ValueError("Current top image must match the history cutoff frame.")
        else:
            # Compatibility for callers supplying a single, uncompressed packet.
            self._upload(policy, packet, task, metadata, message["cursor"], current_image=current_image)
        started = time.perf_counter()
        result = policy.infer({**observation, "step_index": self.last_frame} if use_history else observation)
        dimensions.validate_actions(result["actions"])
        self.awaiting_observation = False
        base.log(f"[memory] infer frame={self.last_frame} sample_actions_ms={(time.perf_counter() - started) * 1000:.1f}")
        return {**result, "history_ack": {"episode_id": self.episode_id, "frame_index": self.last_frame}}

    def handle_message(self, policy, message, task, metadata):
        if self.failed:
            raise RuntimeError("History request failed; reconnect to start a new trial.")
        try:
            if message.get("type") != "history" or message.get("protocol") != base.PROTOCOL_VERSION:
                raise ValueError("Unsupported Memory transport message; update both Memory scripts.")
            packet = message["history"]
            frames = decode_history_frames(packet.get("codec"), packet["compressed_frames"])
            self._upload(policy, {**packet, "frames": frames}, task, metadata, message["cursor"])
            return {"type": "history_ack", "cursor": message["cursor"],
                    "history_ack": {"episode_id": self.episode_id, "frame_index": self.last_frame}}
        except Exception:
            self.failed = True
            raise

    def _upload(self, policy, packet, task, metadata, cursor, *, current_image=None):
        use_history = bool(metadata.get("use_history"))
        if use_history and metadata.get("history_camera", "base") != "base":
            raise ValueError("This robot transport supplies top_camera/base history only.")
        if not isinstance(packet, dict) or packet.get("protocol") != HISTORY_PROTOCOL:
            raise ValueError("Dense history is missing; run robot-server with the _memory script.")
        episode = packet.get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise ValueError("Missing history episode_id.")
        if not np.isclose(float(packet["fps"]), self.history_fps, rtol=0, atol=1e-6):
            raise ValueError("Robot history FPS does not match the bridge/training FPS.")
        new_episode = episode != self.episode_id
        if not new_episode and task != self.prompt:
            raise ValueError("A prompt change requires a new robot history episode.")
        if not new_episode and self.awaiting_observation and cursor != self.last_cursor:
            raise ValueError("History chunk cursor changed before its observation arrived.")
        last = -1 if new_episode else self.last_frame
        indices = [_integer(s, "History frame index") for s in packet["step_indices"]]
        frames = packet["frames"]
        timestamps = np.asarray(packet["timestamps"], dtype=np.float64)
        current = _integer(packet["frame_index"], "Current frame index")
        if not indices or indices != list(range(last + 1, last + 1 + len(indices))):
            raise ValueError("History frames must be dense, ordered and never replayed.")
        if len(frames) != len(indices) or current != indices[-1]:
            raise ValueError("History length/current frame mismatch.")
        if timestamps.shape != (len(indices),) or not np.isfinite(timestamps).all():
            raise ValueError("History timestamps must align with frames.")
        sequence = timestamps if new_episode else np.r_[self.last_timestamp, timestamps]
        gaps = np.diff(sequence)
        if np.any(gaps <= 0) or np.any(gaps > self.max_gap_s):
            raise ValueError("History capture timestamps contain an invalid gap.")
        sampling = packet.get("sampling")
        captures = None
        if not new_episode and sampling != self.sampling:
            raise ValueError("History sampling mode changed within an episode.")
        if sampling is not None:
            if sampling != "nearest_timestamp_v1":
                raise ValueError("Unsupported history sampling mode.")
            captures = np.asarray(packet.get("capture_timestamps"), dtype=np.float64)
            if captures.shape != timestamps.shape or not np.isfinite(captures).all():
                raise ValueError("History capture timestamps must align with resampled slots.")
            origin = timestamps[0] if new_episode else self.grid_origin
            if not np.allclose(timestamps, origin + np.asarray(indices) / self.history_fps, rtol=0, atol=1e-6):
                raise ValueError("Resampled history does not follow the configured time grid.")
            source_sequence = captures if new_episode else np.r_[self.last_capture_timestamp, captures]
            if np.any(np.diff(source_sequence) < 0) or np.any(np.abs(captures - timestamps) > self.max_gap_s + 1e-6):
                raise ValueError("Invalid source timestamps for resampled history.")
        for frame in frames:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (224, 224, 3):
                raise ValueError("History frames must be uint8 RGB [224,224,3].")
        if current_image is not None and not np.array_equal(frames[-1], current_image):
            raise ValueError("Current top image must match the history cutoff frame.")
        started = time.perf_counter()
        if use_history and new_episode:
            if policy.infer({"reset": True}).get("reset_finished") is not True:
                raise RuntimeError("Policy did not acknowledge episode reset.")
        if use_history:
            reply = policy.infer({
                "add_buffer": True, "history_frames": frames,
                "step_indices": indices, "prompt": task,
            })
            if reply.get("add_buffer_finished") is not True:
                raise RuntimeError("Policy did not acknowledge history upload.")
        self.episode_id, self.last_frame = episode, current
        self.last_timestamp, self.prompt = float(timestamps[-1]), task
        self.last_capture_timestamp = None if captures is None else float(captures[-1])
        self.grid_origin = float(timestamps[0]) if new_episode else self.grid_origin
        self.sampling = sampling
        self.last_image, self.last_cursor = frames[-1].copy(), cursor
        self.awaiting_observation = True
        base.log(f"[memory] episode={episode} uploaded={len(indices)} "
                 f"frames={indices[0]}..{current} span_s={timestamps[-1] - timestamps[0]:.3f} "
                 f"upload_encode_ms={(time.perf_counter() - started) * 1000:.1f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    try:
        validate_base_script()
    except RuntimeError as exc:
        parser.error(str(exc))
    sub = parser.add_subparsers(dest="mode", required=True)
    robot = sub.add_parser("robot-server")
    base.add_robot_server_args(robot)
    client = sub.add_parser("openpi-client")
    base.add_openpi_client_args(client)
    client.set_defaults(policy_port=12023)
    for command in (robot, client):
        command.add_argument("--history-fps", type=float, default=30.0)
        command.add_argument("--history-max-gap", type=float, default=0.1,
                             help="maximum allowed host observation interval, seconds")
    robot.add_argument("--history-max-pending", type=int, default=512)
    robot.add_argument("--history-batch-frames", type=int, default=32,
                       help="frames per lossless upload; each batch is acknowledged before action inference")
    robot.add_argument("--memory-home-mode", choices=("reset", "continue"), default="reset",
                       help="reset: Home ends the trial; continue: preserve history through Home")
    args = parser.parse_args(argv)
    if (not np.isfinite(args.history_fps) or args.history_fps <= 0
            or not np.isfinite(args.history_max_gap) or args.history_max_gap < 1 / args.history_fps):
        parser.error("history FPS must be positive and max gap >= one frame period")
    if args.actions_per_response <= 0:
        parser.error("actions-per-response must be positive")
    if args.mode == "robot-server":
        if not 1 <= args.history_batch_frames <= MAX_HISTORY_CHUNK:
            parser.error(f"history-batch-frames must be between 1 and {MAX_HISTORY_CHUNK}")
        if args.image_size != 224 or args.camera_fps < args.history_fps or args.history_max_pending < 1:
            parser.error("Memory needs image-size=224, camera-fps >= history-fps and positive queue capacity")
        # --close is an optional base-script feature, separate from Memory's
        # required transport hooks checked above.
        args.close = getattr(args, "close", None)
        if args.close is not None:
            args.close = tuple(part.strip().lower() for part in args.close)
            if args.close != ("right", "gripper"):
                parser.error("--close currently only supports: --close right gripper")
    return args


def main(argv=None):
    args = parse_args(argv)
    np.set_printoptions(3, suppress=True, linewidth=150)
    if args.mode == "robot-server":
        base.run_robot_server(args, runtime_factory=MemoryRobotRuntime)
    else:
        bridge = MemoryBridge(history_fps=args.history_fps, max_gap_s=args.history_max_gap)
        base.run_openpi_client(args, observation_handler=bridge, warmup_handler=bridge.warmup,
                               message_handler=bridge.handle_message)


if __name__ == "__main__":
    main(sys.argv[1:])
