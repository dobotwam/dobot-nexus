from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from dobot_nexus.core.config import resolve_autocast

TRAINING_RECIPE = "openpi_augmax041_fp32_v1"


def validate_torch_backend(config, *, mode: str) -> None:
    """Reject reference execution while accepting historical Torch configurations."""
    backend = str(config.model.get("inference_backend", "torch")).lower()
    if backend != "torch":
        raise ValueError(
            "Pi0.5 model.inference_backend only supports 'torch' in production; "
            f"got {backend!r}. For JAX comparisons use "
            "local_script/pi05/reference_jax/ (python -m "
            "local_script.pi05.reference_jax --help)."
        )
    if mode == "infer":
        # Ignore paths saved by historical trainers. Keep raw training config
        # keys intact so existing checkpoint resume fingerprints stay valid.
        for key in ("openpi_root", "jax_checkpoint", "jax_export_base_checkpoint"):
            config.model.pop(key, None)


def resolve_core_settings(config, *, mode):
    """Restore the trained operator profile and shapes for every Pi0.5 variant."""
    model = config.model
    default = model.get("operator_numerics", "legacy")
    max_token_len = int(model.get("max_token_len", 200))
    if mode == "infer" and model.get("artifact"):
        from dobot_nexus.core.artifacts import load_manifest, resolve_artifact
        from dobot_nexus.core.config import load_config

        root = resolve_artifact(model.artifact)
        saved = load_config(root / "config_resolved.yaml", expected_task="train")
        # Legacy Memory artifacts already used OpenPI math; their model-owned
        # resolver supplies that default. Legacy BC artifacts used legacy math.
        trained_default = "openpi" if saved.model.target.endswith(":Pi05MemoryModel") else "legacy"
        trained = saved.model.get("operator_numerics", trained_default)
        metadata = load_manifest(root).get("model", {})
        recorded = metadata.get("operator_numerics", trained)
        if recorded != trained:
            raise ValueError("Artifact operator_numerics disagrees with its saved configuration.")
        if "operator_numerics" in model and default != trained:
            raise ValueError("Cannot override trained model.operator_numerics.")
        default = trained
        trained_tokens = int(saved.model.get(
            "max_token_len", 256 if saved.model.target.endswith(":Pi05MemoryModel") else 200
        ))
        if int(metadata.get("max_token_len", trained_tokens)) != trained_tokens:
            raise ValueError("Artifact max_token_len disagrees with its saved configuration.")
        if "max_token_len" in model and max_token_len != trained_tokens:
            raise ValueError("Cannot override trained model.max_token_len.")
        max_token_len = trained_tokens
        trained_dim = saved.model.get("model_action_dim", 32)
        recorded_dim = metadata.get("model_action_dim", trained_dim)
        if recorded_dim != trained_dim:
            raise ValueError("Artifact model_action_dim disagrees with its saved configuration.")
        if "model_action_dim" in model and model.model_action_dim != trained_dim:
            raise ValueError("Cannot override trained model.model_action_dim.")
        model["model_action_dim"] = trained_dim
    if default not in {"legacy", "openpi"}:
        raise ValueError("model.operator_numerics must be 'legacy' or 'openpi'.")
    model["operator_numerics"] = default
    model["max_token_len"] = max_token_len
    return default, max_token_len


def validate_training_recipe(config) -> None:
    """Record model-owned behavior and reject accidental effective-batch drift."""
    model = config["model"]
    recipe = model.setdefault("training_recipe", TRAINING_RECIPE)
    if recipe != TRAINING_RECIPE:
        raise ValueError(f"model.training_recipe must be {TRAINING_RECIPE!r}, got {recipe!r}.")
    data = config.get("data", {})
    expected = data.get("global_batch_size")
    if expected is None:
        return
    if type(expected) is not int or expected <= 0:
        raise ValueError("data.global_batch_size must be a positive integer.")
    world_size = (torch.distributed.get_world_size() if torch.distributed.is_initialized()
                  else int(os.environ.get("WORLD_SIZE", "1")))
    batch = int(data.get("batch_size", 1))
    accum = int(config.get("train", {}).get("grad_accum_steps", 1))
    actual = batch * world_size * accum
    if actual != expected:
        raise ValueError(
            f"data.global_batch_size={expected}, but batch_size={batch} * "
            f"world_size={world_size} * grad_accum_steps={accum} = {actual}. "
            "Adjust data.batch_size or train.grad_accum_steps for this launch."
        )


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
    raise ValueError(f"Unsupported Pi0.5 {field} {value!r}.")


@dataclass(frozen=True)
class Pi05Config:
    """Public robot dimensions and configurable internal action boundary."""

    checkpoint: str
    precision: str
    parameter_precision: str
    state_dim: int
    action_dim: int
    action_horizon: int
    num_steps: int
    camera_mapping: dict[str, str]
    image_size: tuple[int, int]
    inference_backend: str = "torch"
    torch_compile: bool = False
    torch_compile_mode: str = "max-autotune-no-cudagraphs"
    torch_compile_preserve_precision: bool = True
    cuda_graph: bool = False
    cuda_graph_max_entries: int = 2
    model_action_dim: int = 32
    max_token_len: int = 200
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    discrete_state_input: bool = True
    operator_numerics: str = "legacy"
    autocast: str = "auto"

    @classmethod
    def from_config(cls, config, *, mode: str) -> Pi05Config:
        validate_torch_backend(config, mode=mode)
        if config.get("dependencies", {}).get("rlinf_path") is not None:
            raise ValueError(
                "Pi0.5 no longer supports dependencies.rlinf_path. Remove this field "
                "to use the built-in torch_core. If the external checkout contains "
                "custom model changes, migrate them through model.target first."
            )
        # Older resolved configs may retain this obsolete export-only setting.
        config.model.pop("jax_export_base_checkpoint", None)
        if mode == "train":
            validate_training_recipe(config)
        model = config.model
        checkpoint = str(model.get("checkpoint") or "")
        if not checkpoint:
            raise ValueError(f"model.checkpoint is required when building Pi0.5 in {mode!r} mode.")

        # Accept v1 mappings, but always collapse them to the leaf core slots.
        configured_cameras = dict(model.get("io", {}).get("cameras", _DEFAULT_CAMERAS))
        camera_mapping = {
            str(public): _CAMERA_ALIASES.get(str(target), str(target))
            for public, target in configured_cameras.items()
        }
        legacy_backend = model.get("backend", {})
        image_size = tuple(int(item) for item in model.get("image_size", [224, 224]))
        compute_precision = str(
            model.get("precision", config.get("runtime", {}).get("precision", "bf16"))
        )
        parameter_precision = str(
            model.get("parameter_precision", "fp32")
            if mode == "train"
            else model.get("inference_parameter_precision", compute_precision)
        )
        operator_numerics, max_token_len = resolve_core_settings(config, mode=mode)
        result = cls(
            checkpoint=checkpoint,
            precision=compute_precision,
            parameter_precision=parameter_precision,
            autocast=resolve_autocast(config),
            state_dim=model.state_dim,
            action_dim=model.action_dim,
            action_horizon=model.action_horizon,
            num_steps=int(
                config.get("inference", {}).get("num_steps", legacy_backend.get("num_steps", 10))
            ),
            camera_mapping=camera_mapping,
            image_size=(image_size[0], image_size[1]),
            torch_compile=bool(config.get("runtime", {}).get("compile", False)),
            torch_compile_mode=str(config.get("runtime", {}).get(
                "compile_mode", "max-autotune-no-cudagraphs"
            )),
            torch_compile_preserve_precision=bool(config.get("runtime", {}).get(
                "compile_preserve_precision", True
            )),
            cuda_graph=bool(config.get("runtime", {}).get("cuda_graph", False)),
            cuda_graph_max_entries=int(config.get("runtime", {}).get("cuda_graph_max_entries", 2)),
            model_action_dim=model.get("model_action_dim", 32),
            max_token_len=max_token_len,
            operator_numerics=operator_numerics,
        )
        result.validate()
        if (mode == "infer" and result.torch_dtype == torch.float32
                and result.parameter_torch_dtype != torch.float32):
            raise ValueError(
                "Pi0.5 FP32 inference compute requires model.inference_parameter_precision='fp32' "
                "(or omit it to follow model.precision)."
            )
        if (mode == "infer"
                and result.torch_compile and result.torch_compile_mode == "openpi"
                and result.operator_numerics != "openpi"):
            raise ValueError("runtime.compile_mode='openpi' requires model.operator_numerics='openpi'.")
        if (mode == "infer"
                and result.torch_compile and result.torch_compile_mode == "legacy"
                and result.operator_numerics != "legacy"):
            raise ValueError(
                "runtime.compile_mode='legacy' requires model.operator_numerics='legacy'; "
                "Memory/OpenPI operators use compile_mode='openpi'. For native CUDA Graph "
                "inference set runtime.compile=false and runtime.cuda_graph=true. "
            )
        if mode == "train":
            if result.parameter_torch_dtype != torch.float32:
                raise ValueError(
                    "RLinf-aligned Pi0.5 training requires "
                    "model.parameter_precision='fp32'. BF16 is only the compute precision."
                )
            runtime_precision = str(
                config.get("runtime", {}).get("precision", compute_precision)
            )
            runtime_dtype = _torch_dtype(runtime_precision, field="runtime precision")
            if runtime_dtype != result.torch_dtype:
                raise ValueError(
                    "Pi0.5 model.precision and runtime.precision must select the same "
                    f"compute dtype during training, got {compute_precision!r} and "
                    f"{runtime_precision!r}."
                )
        return result

    @property
    def inference_autocast_enabled(self) -> bool:
        return (self.autocast == "auto" and self.parameter_torch_dtype != self.torch_dtype
                and self.torch_dtype != torch.float32)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.camera_mapping)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Operator compute dtype; retained as a compatibility alias."""
        return _torch_dtype(self.precision, field="compute precision")

    @property
    def parameter_torch_dtype(self) -> torch.dtype:
        """Persistent parameter and optimizer-master dtype."""
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
            "pi05": True,
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
        for name in ("state_dim", "action_dim", "action_horizon"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Pi0.5 model.{name} must be a positive integer.")
        if type(self.model_action_dim) is not int or self.model_action_dim <= 0:
            raise ValueError("Pi0.5 model.model_action_dim must be a positive integer.")
        if self.model_action_dim < max(self.state_dim, self.action_dim):
            raise ValueError(
                "Pi0.5 model_action_dim must cover both public dimensions, got "
                f"model_action_dim={self.model_action_dim}, state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}."
            )
        if self.num_steps <= 0:
            raise ValueError("Pi0.5 num_steps must be positive.")
        if self.cuda_graph_max_entries <= 0:
            raise ValueError("runtime.cuda_graph_max_entries must be positive.")
        if self.torch_compile_mode not in {"legacy", "openpi", "default", "max-autotune-no-cudagraphs"}:
            raise ValueError(
                "Pi0.5 runtime.compile_mode must be 'legacy', 'openpi', 'default', "
                "or 'max-autotune-no-cudagraphs'."
            )
        if self.max_token_len <= 0:
            raise ValueError("Pi0.5 max_token_len must be positive.")
        if self.inference_backend != "torch":
            raise ValueError("Pi0.5 model.inference_backend must be 'torch'.")
        if self.image_size != (224, 224):
            raise ValueError("The aligned Pi0.5 operator requires model.image_size=[224, 224].")
        if set(self.camera_mapping.values()) != set(_DEFAULT_CAMERAS.values()):
            raise ValueError(
                "Pi0.5 cameras must populate base_0_rgb, left_wrist_0_rgb, and "
                "right_wrist_0_rgb exactly once."
            )
