from __future__ import annotations

from dataclasses import dataclass

import torch

_DEFAULT_CAMERAS = {
    "base": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}
_CAMERA_ALIASES = {
    "main_images": "base_0_rgb",
    "wrist_images": "left_wrist_0_rgb",
    "extra_view_images": "right_wrist_0_rgb",
}


def _torch_dtype(value: str, *, field: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "bf16-mixed"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "16", "16-mixed"}:
        return torch.float16
    if normalized in {"fp32", "float32", "32", "32-true"}:
        return torch.float32
    raise ValueError(f"Unsupported Pi0 {field} {value!r}.")


@dataclass(frozen=True)
class Pi0Config:
    """Public Pi0 settings plus model-owned operator constants."""

    checkpoint: str
    precision: str
    parameter_precision: str
    state_dim: int
    action_dim: int
    action_horizon: int
    num_steps: int
    camera_mapping: dict[str, str]
    image_size: tuple[int, int]
    torch_compile: bool = False
    model_action_dim: int = 32
    max_token_len: int = 48
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    discrete_state_input: bool = False

    @classmethod
    def from_config(cls, config, *, mode: str) -> Pi0Config:
        if mode not in {"train", "infer"}:
            raise ValueError(f"Pi0 mode must be 'train' or 'infer', got {mode!r}.")
        model = config.model
        checkpoint = str(model.get("checkpoint") or "")
        if not checkpoint:
            raise ValueError(f"model.checkpoint is required when building Pi0 in {mode!r} mode.")

        configured_cameras = dict(model.get("io", {}).get("cameras", _DEFAULT_CAMERAS))
        camera_mapping = {
            str(public): _CAMERA_ALIASES.get(str(target), str(target))
            for public, target in configured_cameras.items()
        }
        image_size = tuple(int(item) for item in model.get("image_size", [224, 224]))
        if len(image_size) != 2:
            raise ValueError(f"Pi0 model.image_size must contain two values, got {image_size}.")
        legacy_backend = model.get("backend", {})
        compute_precision = str(
            model.get("precision", config.get("runtime", {}).get("precision", "bf16"))
        )
        parameter_precision = str(
            model.get("parameter_precision", "fp32")
            if mode == "train"
            else model.get("inference_parameter_precision", compute_precision)
        )
        result = cls(
            checkpoint=checkpoint,
            precision=compute_precision,
            parameter_precision=parameter_precision,
            state_dim=int(model.state_dim),
            action_dim=int(model.action_dim),
            action_horizon=int(model.action_horizon),
            num_steps=int(
                config.get("inference", {}).get(
                    "num_steps", legacy_backend.get("num_inference_steps", 10)
                )
            ),
            camera_mapping=camera_mapping,
            image_size=(image_size[0], image_size[1]),
            torch_compile=bool(config.get("runtime", {}).get("compile", False)),
        )
        result.validate()
        if mode == "train":
            if result.parameter_torch_dtype != torch.float32:
                raise ValueError(
                    "The aligned Pi0 operator requires model.parameter_precision='fp32' "
                    "during training."
                )
            runtime_precision = str(config.get("runtime", {}).get("precision", compute_precision))
            runtime_dtype = _torch_dtype(runtime_precision, field="runtime precision")
            if runtime_dtype != result.torch_dtype:
                raise ValueError(
                    "Pi0 model.precision and runtime.precision must select the same compute "
                    f"dtype during training, got {compute_precision!r} and {runtime_precision!r}."
                )
        return result

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.camera_mapping)

    @property
    def torch_dtype(self) -> torch.dtype:
        return _torch_dtype(self.precision, field="compute precision")

    @property
    def parameter_torch_dtype(self) -> torch.dtype:
        return _torch_dtype(self.parameter_precision, field="parameter precision")

    @property
    def core_dtype(self) -> str:
        return {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }[self.torch_dtype]

    def core_kwargs(self) -> dict[str, object]:
        return {
            "pi05": False,
            "action_dim": self.model_action_dim,
            "action_horizon": self.action_horizon,
            "max_token_len": self.max_token_len,
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
            "dtype": self.core_dtype,
            "discrete_state_input": self.discrete_state_input,
            "pcd": False,
        }

    def validate(self) -> None:
        _ = self.torch_dtype
        _ = self.parameter_torch_dtype
        if self.state_dim <= 0 or self.action_dim <= 0 or self.action_horizon <= 0:
            raise ValueError("Pi0 state/action dimensions and horizon must be positive.")
        if self.model_action_dim < max(self.state_dim, self.action_dim):
            raise ValueError("Pi0 public dimensions cannot exceed the 32-D operator boundary.")
        if self.num_steps <= 0:
            raise ValueError("Pi0 num_steps must be positive.")
        if self.image_size != (224, 224):
            raise ValueError("The aligned Pi0 operator requires model.image_size=[224, 224].")
        if set(self.camera_mapping.values()) != set(_DEFAULT_CAMERAS.values()):
            raise ValueError(
                "Pi0 cameras must populate base_0_rgb, left_wrist_0_rgb, and "
                "right_wrist_0_rgb exactly once."
            )
