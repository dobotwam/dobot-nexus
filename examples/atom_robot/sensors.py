"""Persistent RGB streams with bounded buffers and monotonic freshness checks."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import RuntimeFault, require_fresh


class FrameTimeout(RuntimeFault):
    """A camera's bounded wait did not receive a frame."""


@dataclass(frozen=True)
class CameraSpec:
    name: str
    serial: str
    width: int
    height: int
    fps: int = 15
    warmup_frames: int = 60


ATOM_CAMERAS = (
    CameraSpec("top_camera", "254122304640", 1280, 720),
    CameraSpec("left_camera", "260522272633", 640, 480),
    CameraSpec("right_camera", "260322273894", 640, 480),
)

DEFAULT_CAMERA_CONFIG = Path(__file__).parent / "config/cameras_0759.json"


def load_camera_specs(path):
    """Read deployment camera identities without importing the RealSense driver."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise RuntimeFault("Camera configuration requires schema_version=1")
    try:
        specs = tuple(CameraSpec(**row) for row in data["cameras"])
    except (KeyError, TypeError) as exc:
        raise RuntimeFault("Invalid camera configuration") from exc
    if len(specs) != 3 or {s.name for s in specs} != {s.name for s in ATOM_CAMERAS}:
        raise RuntimeFault("ATOM requires top_camera, left_camera and right_camera exactly once")
    if len({s.serial for s in specs}) != 3:
        raise RuntimeFault("Camera serial numbers must be distinct")
    for spec in specs:
        if not isinstance(spec.serial, str) or not spec.serial.strip():
            raise RuntimeFault("Camera serial must be a nonempty string")
        if any(
            type(v) is not int or v <= 0
            for v in (spec.width, spec.height, spec.fps, spec.warmup_frames)
        ):
            raise RuntimeFault("Camera dimensions, fps and warmup must be positive integers")
        if spec.warmup_frames < 60:
            raise RuntimeFault("Camera validation requires at least 60 warmup frames")
    return specs


def resize_with_pad(rgb, size=224):
    """Match Nexus Pi0.5's Pillow bilinear, centered black-padding transform."""
    from PIL import Image

    source = Image.fromarray(rgb)
    width, height = source.size
    ratio = max(width / size, height / size)
    resized = source.resize(
        (max(1, int(width / ratio)), max(1, int(height / ratio))),
        resample=getattr(Image, "Resampling", Image).BILINEAR,
    )
    padded = Image.new("RGB", (size, size), 0)
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(padded, dtype=np.uint8)


class RealSenseCamera:
    def __init__(self, spec):
        self.spec = spec
        self.pipeline = None
        self.saved_options = []

    def start(self):
        if self.pipeline is not None:
            raise RuntimeFault("Camera stream already started")
        import pyrealsense2 as rs

        self.rs = rs
        self.context = rs.context()
        devices = [
            d
            for d in self.context.query_devices()
            if d.get_info(rs.camera_info.serial_number) == self.spec.serial
        ]
        if len(devices) != 1:
            raise RuntimeFault("Required camera serial is absent or ambiguous: " + self.spec.serial)
        # Limit driver buffering, so receipt time cannot conceal a queue of old frames.
        for sensor in devices[0].query_sensors():
            if sensor.supports(rs.option.frames_queue_size):
                original = sensor.get_option(rs.option.frames_queue_size)
                sensor.set_option(rs.option.frames_queue_size, 1)
                self.saved_options.append((sensor, original))
        pipeline = rs.pipeline(self.context)
        cfg = rs.config()
        cfg.enable_device(self.spec.serial)
        cfg.enable_stream(
            rs.stream.color, self.spec.width, self.spec.height, rs.format.rgb8, self.spec.fps
        )
        try:
            pipeline.start(cfg)
        except BaseException:
            for sensor, original in self.saved_options:
                sensor.set_option(rs.option.frames_queue_size, original)
            self.saved_options = []
            raise
        self.pipeline = pipeline

    def read(self):
        try:
            frame = self.pipeline.wait_for_frames(1000).get_color_frame()
        except RuntimeError as exc:
            if "Frame didn't arrive" in str(exc):
                raise FrameTimeout(str(exc)) from exc
            raise
        if not frame:
            raise RuntimeFault("Missing RGB frame: " + self.spec.serial)
        return (
            np.asanyarray(frame.get_data()).copy(),
            int(frame.get_frame_number()),
            float(frame.get_timestamp()),
            time.monotonic(),
        )

    def close(self):
        errors = []
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as exc:
                errors.append(str(exc))
            self.pipeline = None
        for sensor, original in self.saved_options:
            try:
                sensor.set_option(self.rs.option.frames_queue_size, original)
            except Exception as exc:
                errors.append(str(exc))
        self.saved_options = []
        if errors:
            raise RuntimeFault("Camera cleanup failed: " + "; ".join(errors))


class CameraRig:
    def __init__(self, specs=ATOM_CAMERAS, *, factory=RealSenseCamera, heartbeat=None):
        self.specs = tuple(specs)
        self.factory = factory
        self.heartbeat = heartbeat
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frames = {}
        self.errors = []
        self.cameras = []
        self.threads = []
        self.started = False
        self.closed = False
        self.stats = {
            s.name: {
                "starts": 0,
                "frames": 0,
                "stops": 0,
                "duplicate_frames": 0,
                "device_clock_adjustments": 0,
                "startup_timeouts": 0,
            }
            for s in self.specs
        }

    def start(self):
        if self.started or self.closed:
            raise RuntimeFault("Use a fresh rig for each runtime lifecycle")
        self.started = True
        self.startup_deadline = time.monotonic() + 20.0
        try:
            for spec in self.specs:
                camera = self.factory(spec)
                self.cameras.append(camera)
                camera.start()
                self.stats[spec.name]["starts"] += 1
                thread = threading.Thread(target=self._capture, args=(spec, camera), daemon=True)
                self.threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def _capture(self, spec, camera):
        previous_number = None
        previous_sensor_time = None
        try:
            while not self.stop.is_set():
                try:
                    rgb, number, sensor_time, received = camera.read()
                except FrameTimeout:
                    with self.lock:
                        warmed = self.stats[spec.name]["frames"] >= spec.warmup_frames
                        if not warmed and time.monotonic() < self.startup_deadline:
                            self.stats[spec.name]["startup_timeouts"] += 1
                        else:
                            raise
                    time.sleep(0.02)
                    continue
                if rgb.shape != (spec.height, spec.width, 3) or rgb.dtype != np.uint8:
                    raise RuntimeFault("Camera stream geometry or RGB dtype changed: " + spec.name)
                if previous_number is not None:
                    if number < previous_number:
                        raise RuntimeFault(
                            f"Camera frame counter reset: {spec.name}, {previous_number}->{number}"
                        )
                    if number == previous_number:
                        # Duplicate delivery is not a fresh observation. Leaving
                        # the heartbeat unchanged lets the independent age guard
                        # stop a stuck camera.
                        with self.lock:
                            self.stats[spec.name]["duplicate_frames"] += 1
                        time.sleep(0.001)
                        continue
                    if sensor_time <= previous_sensor_time:
                        # RealSense device/global clock calibration may adjust
                        # timestamps. Frame identity + monotonic host receipt,
                        # with a one-frame driver buffer, remains authoritative.
                        with self.lock:
                            self.stats[spec.name]["device_clock_adjustments"] += 1
                previous_number = number
                previous_sensor_time = sensor_time
                image = resize_with_pad(rgb)
                with self.lock:
                    stat = self.stats[spec.name]
                    stat["frames"] += 1
                    if stat["frames"] >= spec.warmup_frames:
                        self.frames[spec.name] = (image, received, number, sensor_time)
                        if self.heartbeat is not None:
                            self.heartbeat(spec.name, received)
        except BaseException as exc:
            with self.lock:
                self.errors.append(type(exc).__name__ + ": " + str(exc))
            self.stop.set()

    def wait_ready(self, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.errors:
                    raise RuntimeFault("; ".join(self.errors))
                if len(self.frames) == len(self.specs):
                    return
            time.sleep(0.01)
        raise RuntimeFault("Camera startup did not produce all warmed streams")

    def snapshot(self, maximum_age=0.3, maximum_skew=0.1):
        with self.lock:
            if self.errors:
                raise RuntimeFault("; ".join(self.errors))
            if len(self.frames) != len(self.specs):
                raise RuntimeFault("Cameras are not ready")
            now = time.monotonic()
            timestamps = [self.frames[s.name][1] for s in self.specs]
            for spec in self.specs:
                require_fresh(self.frames[spec.name][1], now, maximum_age, spec.name)
            if max(timestamps) - min(timestamps) > maximum_skew:
                raise RuntimeFault("Camera snapshot time skew is too large")
            return (
                {s.name: self.frames[s.name][0].copy() for s in self.specs},
                {
                    s.name: {
                        "received_monotonic": self.frames[s.name][1],
                        "frame_number": self.frames[s.name][2],
                    }
                    for s in self.specs
                },
            )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        errors = []
        for thread in self.threads:
            thread.join(timeout=1.2)
        for index, camera in enumerate(self.cameras):
            spec = self.specs[index]
            try:
                camera.close()
                self.stats[spec.name]["stops"] += 1
            except Exception as exc:
                errors.append(str(exc))
        for thread in self.threads:
            thread.join(timeout=0.2)
        if any(t.is_alive() for t in self.threads):
            errors.append("Camera worker did not stop")
        if errors:
            raise RuntimeFault("; ".join(errors))

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
