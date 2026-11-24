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
from dobot_nexus.models.pi05.deploy.cameras import (
    artifact_camera_mapping as _artifact_camera_mapping,
)
from dobot_nexus.models.pi05.memory.runtime import MemorySession
from dobot_nexus.modules.serving.memory_ws import MemoryConnection as _MemoryConnection
from dobot_nexus.modules.serving.memory_ws import MemoryWebsocketServer as _MemoryWebsocketServer
from dobot_nexus.modules.serving.memory_ws import configured_memory_limits
from dobot_nexus.modules.serving.openpi import OPENPI_TO_NEXUS_IMAGES, build_nexus_observation
from dobot_nexus.modules.serving.openpi import policy_identity as _policy_identity

__all__ = [
    "annotations",
    "argparse",
    "OPENPI_TO_NEXUS_IMAGES",
    "Mapping",
    "MemoryConnection",
    "MemorySession",
    "MemoryWebsocketServer",
    "SamplingOptions",
    "SessionOptions",
    "WebsocketPolicyServer",
    "_policy_identity",
    "build_frame_encoder",
    "build_nexus_observation",
    "configured_memory_limits",
    "contextmanager",
    "io",
    "main",
    "np",
]


class MemoryConnection(_MemoryConnection):
    _session_type = MemorySession

    def __init__(self, policy, encoder, config, *, camera_mapping=None, **kwargs):
        cameras = _artifact_camera_mapping(policy, camera_mapping)
        super().__init__(policy, encoder, config, camera_mapping=cameras, **kwargs)


class MemoryWebsocketServer(_MemoryWebsocketServer):
    _connection_type = MemoryConnection

    def __init__(self, policy, encoder, config, *, camera_mapping=None, **kwargs):
        cameras = _artifact_camera_mapping(policy, camera_mapping)
        super().__init__(policy, encoder, config, camera_mapping=cameras, **kwargs)


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
