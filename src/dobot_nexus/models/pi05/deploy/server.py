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


OPENPI_TO_NEXUS_IMAGES = {
    "top_camera": "base",
    "left_camera": "left_wrist",
    "right_camera": "right_wrist",
}
OPENPI_PROTOCOL_NAME = "openpi-websocket-msgpack_numpy"


def _image_to_hwc_uint8(value: Any, name: str, image_size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"OpenPI image {name!r} must be 3-D, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(
            f"OpenPI image {name!r} must be CHW or HWC RGB with 3 channels, got {image.shape}."
        )
    if image.dtype != np.uint8:
        raise TypeError(f"OpenPI image {name!r} must use uint8, got {image.dtype}.")
    if image.shape[:2] != image_size:
        raise ValueError(
            f"OpenPI image {name!r} must have size {image_size}, got {image.shape[:2]}."
        )
    return np.ascontiguousarray(image)


def build_nexus_observation(
    observation: Mapping[str, Any],
    *,
    image_size: tuple[int, int] = (224, 224),
    state_dim: int = 14,
    default_prompt: str | None = None,
) -> dict[str, Any]:
    """Map the observation emitted by the supplied OpenPI bridge to Nexus."""
    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (state_dim,) or not np.isfinite(state).all():
        raise ValueError(
            f"OpenPI state must be finite with shape ({state_dim},), got {state.shape}."
        )

    raw_images = observation.get("images")
    if not isinstance(raw_images, Mapping):
        raise TypeError("OpenPI observation.images must be an object.")
    missing = sorted(set(OPENPI_TO_NEXUS_IMAGES) - set(raw_images))
    if missing:
        raise KeyError(f"OpenPI observation is missing cameras: {missing}.")

    prompt = observation.get("prompt") or default_prompt
    if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
        prompt = prompt[0]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "OpenPI observation.prompt must be a non-empty string, or configure --default-prompt."
        )

    return {
        "images": {
            nexus_name: _image_to_hwc_uint8(raw_images[openpi_name], openpi_name, image_size)
            for openpi_name, nexus_name in OPENPI_TO_NEXUS_IMAGES.items()
        },
        "state": state,
        "prompt": [prompt],
    }


def _policy_identity(policy: Any) -> dict[str, Any]:
    manifest_model: dict[str, Any] = {}
    if getattr(policy, "manifest", None):
        manifest_model = dict(policy.manifest.get("model", {}))
    runtime_model = getattr(policy, "model", None)
    runtime_metadata = dict(getattr(runtime_model, "inference_metadata", {}))
    return {
        **runtime_metadata,
        "framework": "dobot-nexus",
        "model_name": policy.spec.model_name,
        "implementation": policy.spec.implementation,
        "artifact": str(policy.artifact_dir) if policy.artifact_dir else None,
        "backend": runtime_metadata.get("backend", "torch"),
        "operator_source": runtime_metadata.get(
            "operator_source", manifest_model.get("operator_source")
        ),
        "operator_revision": runtime_metadata.get(
            "operator_revision", manifest_model.get("operator_revision")
        ),
    }


class NexusOpenPiPolicy:
    """OpenPI ``BasePolicy.infer``-compatible wrapper around ``InferencePolicy``."""

    def __init__(
        self,
        policy: Any,
        *,
        seed: int = 42,
        num_steps: int = 10,
        vary_seed: bool = True,
        default_prompt: str | None = None,
    ) -> None:
        self.policy = policy
        self.seed = int(seed)
        self.num_steps = int(num_steps)
        self.vary_seed = bool(vary_seed)
        self.default_prompt = default_prompt
        self.prediction = PredictionHandler(
            policy, seed=seed, num_steps=num_steps, vary_seed=vary_seed
        )

        spec = policy.spec
        expected_cameras = set(OPENPI_TO_NEXUS_IMAGES.values())
        if set(spec.camera_names) != expected_cameras:
            raise ValueError(
                f"The Artifact cameras must be {sorted(expected_cameras)}, "
                f"got {sorted(spec.camera_names)}."
            )

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        canonical = build_nexus_observation(
            observation,
            image_size=tuple(self.policy.spec.image_size),
            state_dim=int(self.policy.spec.state_dim),
            default_prompt=self.default_prompt,
        )
        result = self.prediction(canonical)
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected = (
            1,
            int(self.policy.spec.action_horizon),
            int(self.policy.spec.action_dim),
        )
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(
                f"Nexus policy returned invalid actions: shape={actions.shape}, "
                f"expected={expected}, finite={np.isfinite(actions).all()}."
            )

        return {
            **result,
            "state": np.ascontiguousarray(canonical["state"], dtype=np.float32),
            "actions": np.ascontiguousarray(actions[0]),
        }


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
        f"inference.num_steps={args.num_steps}",
        f"inference.seed={args.seed}",
    ]
    if args.recap_condition is not None:
        overrides.append(f"inference.recap.condition={args.recap_condition}")
    for argument, key in (("torch_compile", "compile"), ("cuda_graph", "cuda_graph"),
                          ("compile_mode", "compile_mode")):
        value = getattr(args, argument, None)
        if value is not None:
            overrides.append(f"runtime.{key}={str(value).lower()}")
    with config_context as config_path:
        config = load_config(config_path, overrides, expected_task="infer")
    logging.info("Inference profile: %s; RECAP condition: %s", args.config or profiles[target],
                 config.inference.get("recap", {}).get("condition"))
    return load_policy(config)


def run_server(args: argparse.Namespace) -> None:
    policy = _load_policy(args)
    if getattr(policy.model, "memory_config", None) is not None:
        from dobot_nexus.models.pi05.deploy.memory_server import (
            MemoryWebsocketServer,
            build_frame_encoder,
            configured_memory_limits,
        )

        server = MemoryWebsocketServer(
            policy, build_frame_encoder(policy), policy.model.memory_config,
            execute_steps=int(policy.config.inference.execute_steps),
            num_steps=args.num_steps, host=args.host, port=args.port,
            seed=args.seed, vary_seed=args.vary_seed, default_prompt=args.default_prompt,
            warmup=args.warmup,
            **configured_memory_limits(policy.config),
        )
        server.metadata.update(_policy_identity(policy))
        logging.info("Loaded Memory policy metadata: %s", server.metadata)
        server.serve_forever()
        return
    identity = _policy_identity(policy)
    adapter = NexusOpenPiPolicy(
        policy,
        seed=args.seed,
        num_steps=args.num_steps,
        vary_seed=args.vary_seed,
        default_prompt=args.default_prompt,
    )
    metadata = {
        **policy.spec.to_dict(),
        **identity,
        "protocol": OPENPI_PROTOCOL_NAME,
        "policy_spec": policy.spec.to_dict(),
        "input_schema": {
            "state": [policy.spec.state_dim],
            "images": {name: [3, *policy.spec.image_size] for name in OPENPI_TO_NEXUS_IMAGES},
            "prompt": "string",
        },
        "output_schema": {"actions": [policy.spec.action_horizon, policy.spec.action_dim]},
    }
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
        "--config", help="Inference YAML; defaults to the BC, RECAP or Memory profile from the Artifact."
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
    parser.add_argument("--compile-mode", choices=("legacy", "openpi", "default", "max-autotune-no-cudagraphs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
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
    if args.port <= 0 or args.gpu < 0 or args.num_steps <= 0:
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
