"""Memory sessions over the shared OpenPI msgpack WebSocket transport."""

from __future__ import annotations

import argparse
import io
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np

from dobot_nexus.core.contracts import SamplingOptions
from dobot_nexus.inference.session import SessionOptions
from dobot_nexus.inference.transport.websocket import WebsocketPolicyServer
from dobot_nexus.models.pi05.deploy.server import (
    OPENPI_TO_NEXUS_IMAGES,
    _policy_identity,
    build_nexus_observation,
)
from dobot_nexus.models.pi05.memory.runtime import MemorySession


class MemoryConnection:
    def __init__(self, policy, encoder, config, *, execute_steps, num_steps=10,
                 seed=None, vary_seed=True, default_prompt=None,
                 max_environments=4, max_history_bytes=256*1024*1024):
        self.session = MemorySession(
            policy, encoder, config, SessionOptions(execute_steps, seed=seed, num_steps=num_steps),
            max_environments=max_environments, max_history_bytes=max_history_bytes
        )
        self.num_steps = num_steps
        self.seed, self.vary_seed, self.default_prompt = seed, vary_seed, default_prompt
        self._request_index = 0

    def infer(self, request):
        if request.get("reset"):
            self.session.reset(request.get("env_id"), request.get("seed"))
            if request.get("seed") is not None:
                self.seed = int(request["seed"])
            self._request_index = 0
            return {"reset_finished": True}
        if not request.get("prompt") and self.default_prompt:
            request = {**request, "prompt": self.default_prompt}
        if request.get("add_buffer"):
            from PIL import Image

            frames = request.get("history_frames")
            if frames is None:
                frames = []
                blobs = request.get("history_jpeg", [])
                if len(blobs) > self.session.max_batch_frames:
                    raise ValueError("History batch exceeds frame limit.")
                for blob in blobs:
                    with Image.open(io.BytesIO(blob)) as image:
                        if image.size != tuple(reversed(self.session.policy.spec.image_size)):
                            raise ValueError("History JPEG size does not match the policy image size.")
                        frames.append(np.array(image.convert("RGB")))
            self.session.push_frames(
                frames,
                request["step_indices"],
                env_id=request.get("env_id", "default"),
                prompt=request["prompt"],
            )
            return {"add_buffer_finished": True}
        images = request.get("images")
        openpi_request = isinstance(images, Mapping) and any(
            name in images for name in OPENPI_TO_NEXUS_IMAGES
        )
        if openpi_request:
            if set(images) & set(OPENPI_TO_NEXUS_IMAGES.values()):
                raise ValueError("Use either OpenPI or Nexus camera names, not both.")
            # Reuse the BC wire codec. The model/session still receives the same
            # canonical HWC observation, including its history control fields.
            request = {
                **request,
                **build_nexus_observation(
                    request,
                    image_size=tuple(self.session.policy.spec.image_size),
                    state_dim=int(self.session.policy.spec.state_dim),
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


class MemoryWebsocketServer:
    """Model-owned composition of Memory sessions and the generic transport."""

    def __init__(self, policy, encoder, config, *, execute_steps, num_steps, host, port,
                 seed=None, vary_seed=True, default_prompt=None, warmup=False,
                 max_environments=4, max_history_bytes=256*1024*1024):
        self.max_environments, self.max_history_bytes = max_environments, max_history_bytes
        self.policy = policy
        self.host, self.port = host, port
        self.warmup_enabled = warmup
        self.metadata = {
            **policy.metadata,
            "use_history": True,
            "history_camera": config.camera,
            "history_requires_dense_frames": True,
            "openpi_compatible": True,
        }
        self.encoder, self.memory_config = encoder, config
        self.execute_steps, self.num_steps = execute_steps, num_steps
        self.seed, self.vary_seed, self.default_prompt = seed, vary_seed, default_prompt

    @contextmanager
    def open_connection(self, context):
        connection = self._connection_policy()
        try:
            yield connection.infer
        finally:
            # Only this connection's buffers; never clear the shared model/encoder caches.
            connection.session.reset()

    def prepare(self):
        if self.warmup_enabled:
            self.warmup()
        self.metadata.update(_policy_identity(self.policy))
        return self.metadata

    def serve_forever(self):
        WebsocketPolicyServer(
            host=self.host, port=self.port, metadata=self.metadata,
            open_connection=self.open_connection, prepare=self.prepare, max_connections=4,
        ).serve_forever()

    def _connection_policy(self):
        return MemoryConnection(
            self.policy,
            self.encoder,
            self.memory_config,
            max_environments=self.max_environments, max_history_bytes=self.max_history_bytes,
            execute_steps=self.execute_steps,
            num_steps=self.num_steps,
            seed=self.seed, vary_seed=self.vary_seed, default_prompt=self.default_prompt,
        )

    def warmup(self):
        # Disposable connection: robot sessions inherit neither history nor RNG progress.
        height, width = self.policy.spec.image_size
        with self.open_connection(None) as infer:
            infer({
                "images": {
                    name: np.zeros((height, width, 3), np.uint8)
                    for name in self.policy.spec.camera_names
                },
                "state": np.zeros(self.policy.spec.state_dim, np.float32),
                "prompt": "warm up", "seed": 0,
            })


def configured_memory_limits(config):
    """Resolve session quotas identically for both public deployment entrypoints."""
    server = getattr(config, "server", {})
    return {
        "max_environments": int(server.get("memory_max_environments", 4)),
        "max_history_bytes": int(server.get("memory_max_history_bytes", 256 * 1024 * 1024)),
    }


def build_frame_encoder(policy):
    memory = policy.model.memory_config
    from dobot_nexus.models.pi05.memory.torch_vision import TorchFrameTokenEncoder

    return TorchFrameTokenEncoder(
        memory.vision_checkpoint, device=policy.device, expected_sha256=memory.vision_sha256,
        cuda_graph=policy.model.pi05_config.cuda_graph,
        max_graphs=policy.model.pi05_config.cuda_graph_max_entries,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12021)
    args = parser.parse_args()
    from dobot_nexus.core.config import load_config
    from dobot_nexus.inference.policy import load_policy

    config = load_config(args.config, expected_task="infer")
    policy = load_policy(config)
    memory = policy.model.memory_config
    encoder = build_frame_encoder(policy)
    server = MemoryWebsocketServer(
        policy,
        encoder,
        memory,
        execute_steps=int(config.inference.execute_steps),
        num_steps=int(config.inference.num_steps),
        host=args.host,
        port=args.port,
        seed=config.inference.get("seed"),
        warmup=config.inference.get("warmup", True),
        **configured_memory_limits(config),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
