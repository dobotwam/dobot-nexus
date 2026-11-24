"""Memory sessions over the shared OpenPI msgpack WebSocket transport."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from dobot_nexus.core.contracts import SamplingOptions
from dobot_nexus.inference.session import SessionOptions
from dobot_nexus.inference.transport.websocket import WebsocketPolicyServer
from dobot_nexus.modules.memory.runtime import MemorySession
from dobot_nexus.modules.memory.sampling import SingleFrame
from dobot_nexus.modules.memory.types import MemoryConfig

from .openpi import (
    OPENPI_PROTOCOL_NAME,
    OPENPI_TO_NEXUS_IMAGES,
    build_nexus_observation,
    policy_identity,
    validated_camera_mapping,
)


def _memory_camera_mapping(mapping, policy, config):
    cameras = policy.spec.camera_names
    result = validated_camera_mapping(mapping, camera_names=cameras)
    if config.camera not in cameras:
        raise ValueError(f"History camera {config.camera!r} is missing from Artifact cameras.")
    return result, next(wire for wire, public in result.items() if public == config.camera)


class MemoryConnection:
    _session_type = MemorySession

    def __init__(
        self,
        policy: Any,
        encoder: Callable[[list[Any]], np.ndarray],
        config: MemoryConfig,
        *,
        execute_steps: int,
        num_steps: int = 10,
        seed: int | None = None,
        vary_seed: bool = True,
        default_prompt: str | None = None,
        max_environments: int = 4,
        max_history_bytes: int = 256 * 1024 * 1024,
        camera_mapping: Mapping[str, str] = OPENPI_TO_NEXUS_IMAGES,
        single_frame: SingleFrame | None = None,
    ) -> None:
        self.camera_mapping, self.history_wire_camera = _memory_camera_mapping(
            camera_mapping, policy, config
        )
        self._legacy_history_wire_camera = next(
            (wire for wire, public in OPENPI_TO_NEXUS_IMAGES.items() if public == config.camera),
            None,
        )
        self.session = self._session_type(
            policy,
            encoder,
            config,
            SessionOptions(execute_steps, seed=seed, num_steps=num_steps),
            max_environments=max_environments,
            max_history_bytes=max_history_bytes,
            single_frame=single_frame,
        )
        self.num_steps = num_steps
        self.seed, self.vary_seed, self.default_prompt = seed, vary_seed, default_prompt
        self._request_index = 0

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        openpi_request = self._uses_openpi(request)
        if request.get("reset"):
            self.session.reset(request.get("env_id"), request.get("seed"))
            if request.get("seed") is not None:
                self.seed = int(request["seed"])
            self._request_index = 0
            return {"reset_finished": True}
        if not request.get("prompt") and self.default_prompt:
            request = {**request, "prompt": self.default_prompt}
        if request.get("add_buffer"):
            camera = request.get("history_wire_camera", self._legacy_history_wire_camera)
            if not isinstance(camera, str) or camera != self.history_wire_camera:
                raise ValueError(
                    f"add_buffer requires history_wire_camera={self.history_wire_camera!r}."
                )
            frames = request.get("history_frames")
            if frames is None:
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise ImportError("Decoding history_jpeg requires pip install Pillow.") from exc
                frames = []
                blobs = request.get("history_jpeg", [])
                if len(blobs) > self.session.max_batch_frames:
                    raise ValueError("History batch exceeds frame limit.")
                for blob in blobs:
                    with Image.open(io.BytesIO(blob)) as image:
                        if image.size != tuple(reversed(self.session.policy.spec.image_size)):
                            raise ValueError(
                                "History JPEG size does not match the policy image size."
                            )
                        frames.append(np.array(image.convert("RGB")))
            self.session.push_frames(
                frames,
                request["step_indices"],
                env_id=request.get("env_id", "default"),
                prompt=request["prompt"],
            )
            return {"add_buffer_finished": True}
        if openpi_request:
            # Reuse the BC wire codec. The model/session still receives the same
            # canonical HWC observation, including its history control fields.
            request = {
                **request,
                **build_nexus_observation(
                    request,
                    image_size=tuple(self.session.policy.spec.image_size),
                    state_dim=int(self.session.policy.spec.state_dim),
                    camera_mapping=self.camera_mapping,
                ),
            }
        steps = request.get("step_indices")
        if steps is None and "step_index" in request:
            steps = [request["step_index"]]
        if "env_id" in request and "env_ids" not in request:
            request = {**request, "env_ids": [request["env_id"]]}
        if request.get("act"):
            actions = self.session.act(request, step_indices=steps)
            return {"actions": actions[0] if openpi_request else actions}
        seed = request.get("seed")
        if seed is None and self.seed is not None:
            seed = self.seed + self._request_index if self.vary_seed else self.seed
        output = self.session.predict_chunk(
            request,
            step_indices=steps,
            options=SamplingOptions(seed=seed, num_steps=self.num_steps),
        )
        self._request_index += 1
        # Existing Nexus clients keep [B,H,A]; OpenPI robot bridges expect [H,A].
        return {
            "actions": output.actions[0] if openpi_request else output.actions,
            "policy_timing": output.timing,
            "seed": seed,
        }

    def _uses_openpi(self, request: Mapping[str, Any]) -> bool:
        images = request.get("images")
        wire = self.camera_mapping.keys()
        native = set(self.camera_mapping.values())
        control = request.get("reset") or request.get("add_buffer")
        if "protocol" in request:
            protocol = request["protocol"]
            if protocol not in ("nexus", "openpi", OPENPI_PROTOCOL_NAME):
                raise ValueError(f"Unknown memory protocol {protocol!r}.")
            expected = native if protocol == "nexus" else wire
            if (not control or "images" in request) and (
                not isinstance(images, Mapping) or images.keys() != expected
            ):
                raise ValueError(f"Protocol {protocol!r} requires camera names {sorted(expected)}.")
            return protocol != "nexus"
        if control or not isinstance(images, Mapping):
            return False
        keys = images.keys()
        has_wire = bool(keys & (wire - native))
        has_native = bool(keys & (native - wire))
        if has_wire and has_native:
            raise ValueError("Use either OpenPI or Nexus camera names, not both.")
        if not has_wire and not has_native and keys & wire & native:
            raise ValueError("Ambiguous camera names. Specify protocol='nexus' or 'openpi'.")
        return has_wire


class MemoryWebsocketServer:
    """Serve an independent memory session on each WebSocket connection."""

    _connection_type = MemoryConnection

    def __init__(
        self,
        policy: Any,
        encoder: Callable[[list[Any]], np.ndarray],
        config: MemoryConfig,
        *,
        execute_steps: int,
        num_steps: int,
        host: str,
        port: int,
        seed: int | None = None,
        vary_seed: bool = True,
        default_prompt: str | None = None,
        warmup: bool = False,
        max_environments: int = 4,
        max_history_bytes: int = 256 * 1024 * 1024,
        camera_mapping: Mapping[str, str] = OPENPI_TO_NEXUS_IMAGES,
        single_frame: SingleFrame | None = None,
    ) -> None:
        self.camera_mapping, self.history_wire_camera = _memory_camera_mapping(
            camera_mapping, policy, config
        )
        self.max_environments, self.max_history_bytes = max_environments, max_history_bytes
        self.policy = policy
        self.host, self.port = host, port
        self.warmup_enabled = warmup
        self.metadata = {
            **policy.metadata,
            "use_history": True,
            "history_camera": config.camera,
            "history_wire_camera": self.history_wire_camera,
            "history_requires_dense_frames": True,
            "openpi_compatible": True,
        }
        self.encoder, self.memory_config = encoder, config
        self.single_frame = single_frame
        self.execute_steps, self.num_steps = execute_steps, num_steps
        self.seed, self.vary_seed, self.default_prompt = seed, vary_seed, default_prompt

    @contextmanager
    def open_connection(self, context: Any) -> Iterator[Callable[[dict], dict]]:
        connection = self._connection_policy()
        try:
            yield connection.infer
        finally:
            # Only this connection's buffers; never clear the shared model/encoder caches.
            connection.session.reset()

    def prepare(self) -> dict[str, Any]:
        if self.warmup_enabled:
            self.warmup()
        self.metadata.update(
            policy_identity(self.policy),
            history_camera=self.memory_config.camera,
            history_wire_camera=self.history_wire_camera,
        )
        return self.metadata

    def serve_forever(self) -> None:
        WebsocketPolicyServer(
            host=self.host,
            port=self.port,
            metadata=self.metadata,
            open_connection=self.open_connection,
            prepare=self.prepare,
            max_connections=4,
        ).serve_forever()

    def _connection_policy(self) -> MemoryConnection:
        return self._connection_type(
            self.policy,
            self.encoder,
            self.memory_config,
            max_environments=self.max_environments,
            max_history_bytes=self.max_history_bytes,
            execute_steps=self.execute_steps,
            num_steps=self.num_steps,
            camera_mapping=self.camera_mapping,
            single_frame=self.single_frame,
            seed=self.seed,
            vary_seed=self.vary_seed,
            default_prompt=self.default_prompt,
        )

    def warmup(self) -> None:
        # Disposable connection: robot sessions inherit neither history nor RNG progress.
        height, width = self.policy.spec.image_size
        with self.open_connection(None) as infer:
            infer(
                {
                    "images": {
                        name: np.zeros((height, width, 3), np.uint8)
                        for name in self.policy.spec.camera_names
                    },
                    "state": np.zeros(self.policy.spec.state_dim, np.float32),
                    "prompt": "warm up",
                    "protocol": "nexus",
                    "seed": 0,
                }
            )


def configured_memory_limits(config: Any) -> dict[str, int]:
    """Resolve session quotas identically for both public deployment entrypoints."""
    server = getattr(config, "server", {})
    return {
        "max_environments": int(server.get("memory_max_environments", 4)),
        "max_history_bytes": int(server.get("memory_max_history_bytes", 256 * 1024 * 1024)),
    }
