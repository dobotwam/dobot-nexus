#!/usr/bin/env python3
"""Serve a Nexus Artifact through the OpenPI WebSocket policy protocol."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np

from dobot_nexus.inference.handlers import PredictionHandler
from dobot_nexus.inference.transport.websocket import WebsocketPolicyServer
from dobot_nexus.models.pi05.deploy.cameras import artifact_camera_mapping
from dobot_nexus.modules.serving.openpi import (
    OPENPI_PROTOCOL_NAME,
    OPENPI_TO_NEXUS_IMAGES,
    _image_to_hwc_uint8,
    build_nexus_observation,
)
from dobot_nexus.modules.serving.openpi import NexusOpenPiPolicy as _NexusOpenPiPolicy
from dobot_nexus.modules.serving.openpi import build_openpi_metadata as _build_openpi_metadata
from dobot_nexus.modules.serving.openpi import policy_identity as _policy_identity

__all__ = [
    "Any",
    "Iterable",
    "Path",
    "WebsocketPolicyServer",
    "annotations",
    "argparse",
    "as_file",
    "files",
    "logging",
    "nullcontext",
    "os",
    "time",
    "OPENPI_PROTOCOL_NAME",
    "OPENPI_TO_NEXUS_IMAGES",
    "NexusOpenPiPolicy",
    "Mapping",
    "PredictionHandler",
    "_image_to_hwc_uint8",
    "_policy_identity",
    "build_nexus_observation",
    "build_openpi_metadata",
    "main",
    "np",
    "parse_args",
    "run_server",
]


class NexusOpenPiPolicy(_NexusOpenPiPolicy):
    def __init__(self, policy, *, camera_mapping=None, **kwargs):
        cameras = artifact_camera_mapping(policy, camera_mapping)
        inference = getattr(getattr(policy, "config", None), "inference", None)
        kwargs.setdefault("num_steps", getattr(inference, "num_steps", 10))
        super().__init__(policy, camera_mapping=cameras, **kwargs)


def build_openpi_metadata(policy, *, camera_mapping=None):
    cameras = artifact_camera_mapping(policy, camera_mapping)
    return _build_openpi_metadata(policy, camera_mapping=cameras)


def _load_policy(args: argparse.Namespace) -> Any:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from dobot_nexus.core.artifacts import load_manifest, resolve_artifact
    from dobot_nexus.core.config import load_config
    from dobot_nexus.inference import load_policy
    from dobot_nexus.models.pi05.configuration_recap import POLICY_TARGETS

    artifact = resolve_artifact(args.artifact)
    model = load_manifest(artifact)["model"]
    target = model.get("target", "")
    profiles = {
        "dobot_nexus.models.pi05.modeling_pi05:Pi05Model": "pi05.yaml",
        "dobot_nexus.algorithms.rltoken.policy:RLTokenPolicy": "pi05_rltoken.yaml",
        **dict.fromkeys(POLICY_TARGETS, "pi05_recap.yaml"),
        "dobot_nexus.models.pi05.memory.modeling_memory:Pi05MemoryModel": "pi05_memory.yaml",
    }
    if target not in profiles:
        raise ValueError(
            f"Unsupported Pi0.5 deployment target {target!r}. "
            "Legacy Classic Artifacts require metadata migration before deployment."
        )
    if args.recap_condition is not None and not target.endswith(":Pi05ReCapModel"):
        raise ValueError("--recap-condition requires a RECAP Artifact.")
    profile = files("dobot_nexus.models.pi05.deploy.profiles").joinpath(profiles[target])
    config_context = nullcontext(Path(args.config)) if args.config else as_file(profile)
    overrides = [
        f"model.artifact={artifact}",
        f"model.target={target}",
        f"runtime.device={args.device}",
        f"model.inference_backend={args.backend}",
        f"inference.seed={args.seed}",
    ]
    if args.num_steps is not None:
        overrides.append(f"inference.num_steps={args.num_steps}")
    if args.recap_condition is not None:
        overrides.append(f"inference.recap.condition={args.recap_condition}")
    for argument, key in (("torch_compile", "compile"), ("cuda_graph", "cuda_graph"),
                          ("compile_mode", "compile_mode")):
        value = getattr(args, argument, None)
        if value is not None:
            overrides.append(f"runtime.{key}={str(value).lower()}")
    with config_context as config_path:
        config = load_config(config_path, overrides, expected_task="infer")
    if profiles[target] == "pi05_rltoken.yaml":
        config.setdefault("inference", {}).setdefault("num_steps", None)
    logging.info("Inference profile: %s; RECAP condition: %s", args.config or profiles[target],
                 config.inference.get("recap", {}).get("condition"))
    return load_policy(config)


def run_server(args: argparse.Namespace) -> None:
    policy = _load_policy(args)
    inference = getattr(getattr(policy, "config", None), "inference", None)
    num_steps = args.num_steps if args.num_steps is not None else getattr(inference, "num_steps", 10)
    if getattr(policy.model, "memory_config", None) is not None:
        from dobot_nexus.models.pi05.deploy.memory_server import (
            MemoryWebsocketServer,
            build_frame_encoder,
            configured_memory_limits,
        )

        server = MemoryWebsocketServer(
            policy, build_frame_encoder(policy), policy.model.memory_config,
            execute_steps=int(policy.config.inference.execute_steps),
            num_steps=num_steps, host=args.host, port=args.port,
            seed=args.seed, vary_seed=args.vary_seed, default_prompt=args.default_prompt,
            warmup=args.warmup,
            **configured_memory_limits(policy.config),
        )
        server.metadata.update(_policy_identity(policy))
        logging.info("Loaded Memory policy metadata: %s", server.metadata)
        server.serve_forever()
        return
    cameras = artifact_camera_mapping(policy)
    adapter = NexusOpenPiPolicy(
        policy,
        seed=args.seed,
        num_steps=num_steps,
        vary_seed=args.vary_seed,
        default_prompt=args.default_prompt,
        camera_mapping=cameras,
    )
    metadata = build_openpi_metadata(policy, camera_mapping=cameras)
    logging.info("Loaded policy metadata: %s", metadata)

    def prepare():
        if args.warmup:
            started = time.monotonic()
            policy.warmup()
            logging.info("Policy warmup completed in %.1f ms", (time.monotonic() - started) * 1000)
        identity = _policy_identity(policy)
        logging.info("Policy acceleration: %s", identity)
        return identity

    WebsocketPolicyServer(
        # BC and RECAP intentionally share the legacy seed counter across connections.
        open_connection=lambda context: nullcontext(adapter.infer),
        host=args.host, port=args.port, metadata=metadata, prepare=prepare,
    ).serve_forever()


def _torch_backend(value: str) -> str:
    if value != "torch":
        raise argparse.ArgumentTypeError(
            "Production only supports 'torch'; use local_script/pi05/reference_jax/ "
            "for JAX comparisons."
        )
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--config",
        help="Inference YAML; defaults to the BC, RECAP, Memory or RLToken profile from the Artifact."
    )
    parser.add_argument("--recap-condition", choices=("base", "positive", "negative"),
                        help="Override RECAP conditioning; the shipped profile defaults to base.")
    parser.add_argument(
        "--backend", type=_torch_backend, choices=("torch",), default="torch",
        help="Production uses Torch; JAX comparisons live in local_script/pi05/reference_jax/.",
    )
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=None,
                        help="Compile Torch inference regions selected by --compile-mode.")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None,
                        help="Capture and replay the complete Torch sampler (CUDA only).")
    parser.add_argument(
        "--compile-mode", choices=("legacy", "openpi", "default", "max-autotune-no-cudagraphs")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, help="Override inference.num_steps from the profile.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-seed",
        dest="vary_seed",
        action="store_false",
        help="Reuse one diffusion noise seed for deterministic protocol checks.",
    )
    parser.add_argument("--default-prompt")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(vary_seed=True, warmup=True)
    args = parser.parse_args(argv)
    if args.port <= 0 or args.gpu < 0 or (args.num_steps is not None and args.num_steps <= 0):
        parser.error("port/num-steps must be positive and gpu must be non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_server(parse_args(argv))
    except KeyboardInterrupt:
        logging.info("OpenPI-compatible policy server stopped")


if __name__ == "__main__":
    main()
