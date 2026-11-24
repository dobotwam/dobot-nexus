"""Map OpenPI wire observations, action chunks and metadata to Nexus policy contracts."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

from dobot_nexus.inference.handlers import PredictionHandler

OPENPI_TO_NEXUS_IMAGES = {
    "top_camera": "base",
    "left_camera": "left_wrist",
    "right_camera": "right_wrist",
}
OPENPI_PROTOCOL_NAME = "openpi-websocket-msgpack_numpy"


def validated_camera_mapping(
    mapping: Mapping[str, str], *, camera_names: Collection[str] | None = None
) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("camera_mapping must be a non-empty mapping.")
    result = dict(mapping)
    if any(not isinstance(name, str) or not name for pair in result.items() for name in pair):
        raise ValueError("camera_mapping camera names must be non-empty strings.")
    if len(set(result.values())) != len(result):
        raise ValueError("camera_mapping must be one-to-one with unique destination camera names.")
    if camera_names is not None and set(result.values()) != set(camera_names):
        raise ValueError(
            f"Artifact cameras {sorted(camera_names)} must match camera_mapping destinations "
            f"{sorted(result.values())}."
        )
    return result


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
    camera_mapping: Mapping[str, str] = OPENPI_TO_NEXUS_IMAGES,
) -> dict[str, Any]:
    """Map the observation emitted by the supplied OpenPI bridge to Nexus."""
    camera_mapping = validated_camera_mapping(camera_mapping)
    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (state_dim,) or not np.isfinite(state).all():
        raise ValueError(
            f"OpenPI state must be finite with shape ({state_dim},), got {state.shape}."
        )

    raw_images = observation.get("images")
    if not isinstance(raw_images, Mapping):
        raise TypeError("OpenPI observation.images must be an object.")
    missing = sorted(set(camera_mapping) - set(raw_images))
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
            for openpi_name, nexus_name in camera_mapping.items()
        },
        "state": state,
        "prompt": [prompt],
    }


def policy_identity(policy: Any) -> dict[str, Any]:
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
        camera_mapping: Mapping[str, str] = OPENPI_TO_NEXUS_IMAGES,
    ) -> None:
        self.policy = policy
        self.seed = int(seed)
        self.num_steps = int(num_steps)
        self.vary_seed = bool(vary_seed)
        self.default_prompt = default_prompt
        self.camera_mapping = validated_camera_mapping(
            camera_mapping, camera_names=policy.spec.camera_names
        )
        self.prediction = PredictionHandler(
            policy, seed=seed, num_steps=num_steps, vary_seed=vary_seed
        )

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        canonical = build_nexus_observation(
            observation,
            image_size=tuple(self.policy.spec.image_size),
            state_dim=int(self.policy.spec.state_dim),
            default_prompt=self.default_prompt,
            camera_mapping=self.camera_mapping,
        )
        result = self.prediction(canonical, replay_seed=observation.get("replay_seed"))
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


def build_openpi_metadata(
    policy: Any, *, camera_mapping: Mapping[str, str] = OPENPI_TO_NEXUS_IMAGES
) -> dict[str, Any]:
    """Describe the adapter's wire cameras and the policy's native dimensions."""
    camera_mapping = validated_camera_mapping(camera_mapping, camera_names=policy.spec.camera_names)
    return {
        **policy.spec.to_dict(),
        **policy_identity(policy),
        "protocol": OPENPI_PROTOCOL_NAME,
        "supports_replay_seed": True,
        "policy_spec": policy.spec.to_dict(),
        "input_schema": {
            "state": [policy.spec.state_dim],
            "images": {name: [3, *policy.spec.image_size] for name in camera_mapping},
            "prompt": "string",
        },
        "output_schema": {"actions": [policy.spec.action_horizon, policy.spec.action_dim]},
    }
