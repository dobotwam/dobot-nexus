from __future__ import annotations

import logging
import time

import torch

from dobot_nexus.core.contracts import (
    ObservationBatch,
    PolicyOutput,
    PolicySpec,
    SamplingOptions,
    TrainOutput,
)
from dobot_nexus.models.base import BaseVLAModule
from dobot_nexus.models.pi0.configuration_pi0 import Pi0Config, restore_token_limit
from dobot_nexus.models.pi0.data_adapter_pi0 import Pi0DataAdapter
from dobot_nexus.models.pi0.training import load_core_checkpoint, load_core_source

_LOGGER = logging.getLogger(__name__)


class Pi0Model(BaseVLAModule):
    """Self-contained PyTorch Pi0 implementation of the Nexus model contract."""

    def __init__(self, config, *, mode: str) -> None:
        super().__init__()
        if mode not in {"train", "infer"}:
            raise ValueError(f"Pi0Model mode must be 'train' or 'infer', got {mode!r}.")
        config = self._resolve_config(config, mode=mode)
        self.root_config = config
        self.mode = mode
        self.pi0_config = Pi0Config.from_config(config, mode=mode)
        resume_from = config.get("train", {}).get("resume_from")
        if mode == "train" and resume_from:
            from dobot_nexus.core.artifacts import load_manifest

            recorded = load_manifest(resume_from)["model"].get("max_token_len")
            if type(recorded) is not int or recorded != self.pi0_config.max_token_len:
                raise ValueError(
                    "Exact Pi0 resume requires matching model.max_token_len "
                    f"(checkpoint={recorded!r}, configured={self.pi0_config.max_token_len})."
                )
        if mode == "infer":
            _LOGGER.info("Pi0 inference parameters=%s compute=%s autocast=%s enabled=%s",
                         self.pi0_config.parameter_precision, self.pi0_config.precision,
                         self.pi0_config.autocast, self.pi0_config.inference_autocast_enabled)
        self.core_source = load_core_source(config, enable_compile=self.pi0_config.torch_compile)
        core_config = self.core_source.Pi0Config(**self.pi0_config.core_kwargs())
        with torch.device("meta"):
            self.core_model = core_config.create()
            self._configure_core()
            if self.pi0_config.autocast == "off":
                from dobot_nexus.models.pi0.numerics import configure_input_dtype_linears

                configure_input_dtype_linears(self.core_model)
        self.loaded_checkpoint = self._load_core_checkpoint()
        meta_parameters = [
            name for name, value in self.core_model.named_parameters() if value.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                "Pi0 checkpoint left uninitialized parameters: "
                + ", ".join(meta_parameters[:8])
            )
        self.core_model.to(dtype=self.pi0_config.parameter_torch_dtype)
        if mode == "train":
            parameter_dtypes = {value.dtype for value in self.core_model.parameters()}
            if parameter_dtypes != {torch.float32}:
                raise RuntimeError(
                    "The aligned Pi0 operator requires every trainable parameter to remain "
                    f"FP32, got {sorted(str(item) for item in parameter_dtypes)}."
                )
            if bool(config.get("train", {}).get("gradient_checkpointing", True)):
                self.core_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        self.adapter = self._build_adapter(self.core_source.Observation)

    def _resolve_config(self, config, *, mode):
        if mode == "infer" and config.model.get("artifact"):
            from dobot_nexus.core.artifacts import load_manifest, resolve_artifact
            from dobot_nexus.core.config import clone_config, load_config

            config = clone_config(config)
            root = resolve_artifact(config.model.artifact)
            trained = load_config(root / "config_resolved.yaml", expected_task="train")
            metadata = load_manifest(root)["model"]
            if metadata.get("target") != trained.model.target:
                raise ValueError("Artifact model.target disagrees with its saved configuration.")
            restore_token_limit(config.model, trained.model, metadata)
        return config

    def _configure_core(self):
        """Install optional Pi0 operators on the meta device before checkpoint loading."""

    def _load_core_checkpoint(self):
        return load_core_checkpoint(self.core_model, self.pi0_config.checkpoint)

    def _build_adapter(self, observation_type):
        return Pi0DataAdapter(self.pi0_config, self.root_config, observation_type)

    @classmethod
    def from_config(cls, config, *, mode: str) -> Pi0Model:
        return cls(config, mode=mode)

    def build_train_dataloader(self, config):
        if self.mode != "train":
            raise RuntimeError("A Pi0 inference module cannot build a training dataloader.")
        loader, self.resolved_data_config = self.adapter.pipeline.build_dataloader()
        token_limit = self.pi0_config.max_token_len
        # An omitted default is part of the original BC resume fingerprint.
        if "max_token_len" in config.model or token_limit != Pi0Config.max_token_len:
            config.model["max_token_len"] = token_limit
        return loader

    def training_step(self, batch) -> TrainOutput:
        if self.mode != "train":
            raise RuntimeError("training_step is only available in train mode.")
        observation, actions = batch["observation"], batch["actions"]
        model_kwargs = batch.get("model_kwargs")
        if model_kwargs:
            loss_per_step = self.core_model.compute_loss(
                observation, actions, train=True, **model_kwargs
            )
        else:
            loss_per_step = self.core_model.compute_loss(observation, actions, train=True)
        loss = loss_per_step.mean()
        return TrainOutput(
            loss=loss,
            metrics={"bc_loss": loss.detach(), "flow_loss": loss.detach()},
        )

    def _sampling_kwargs(self, encoded):
        return getattr(encoded, "model_kwargs", {})

    def predict_action_chunk(
        self,
        observation: ObservationBatch,
        options: SamplingOptions,
    ) -> PolicyOutput:
        if self.mode != "infer":
            raise RuntimeError("predict_action_chunk is only available in infer mode.")
        parameter = next(self.core_model.parameters())
        encoded = self.adapter.move_encoded(
            self.adapter.encode_observation(observation), parameter.device
        )
        generator = None
        if options.seed is not None:
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(int(options.seed))
        noise = options.noise
        if noise is not None:
            # Match the core's generated FP32 noise. Weight storage precision
            # must not quantize supplied noise or the Euler integration state.
            noise = torch.as_tensor(noise, device=parameter.device, dtype=torch.float32)
            if noise.ndim == 2:
                noise = noise[None]
            expected_noise = (
                observation.batch_size,
                self.pi0_config.action_horizon,
                self.pi0_config.model_action_dim,
            )
            if tuple(noise.shape) != expected_noise:
                raise ValueError(
                    f"Pi0 noise must have shape {expected_noise} or {expected_noise[1:]}, "
                    f"got {tuple(noise.shape)}."
                )
        num_steps = (
            self.pi0_config.num_steps if options.num_steps is None else int(options.num_steps)
        )
        if num_steps <= 0:
            raise ValueError("SamplingOptions.num_steps must be positive.")

        started = time.perf_counter()
        # An FP32 Artifact may deliberately retain FP32 parameters at inference
        # while the operator embeds inputs in BF16/FP16, as it does in training.
        # Leave the default same-dtype path unchanged.
        compute_dtype = self.pi0_config.torch_dtype
        with torch.inference_mode(), torch.autocast(
            device_type=parameter.device.type,
            dtype=compute_dtype,
            enabled=(self.pi0_config.autocast == "auto"
                     and parameter.dtype != compute_dtype and compute_dtype != torch.float32),
        ):
            sample_kwargs = {"num_steps": num_steps, "noise": noise, "rng": generator}
            model_kwargs = self._sampling_kwargs(encoded)
            if model_kwargs:
                actions = self.core_model.sample_actions(
                    encoded.observation, **sample_kwargs, **model_kwargs
                )
            else:
                actions = self.core_model.sample_actions(encoded.observation, **sample_kwargs)
        total_ms = (time.perf_counter() - started) * 1000.0
        return self.adapter.decode_actions(
            actions,
            aux={"raw_state": encoded.raw_state, "timing": {"total_ms": total_ms}},
        )

    @property
    def policy_spec(self) -> PolicySpec:
        return PolicySpec(
            model_name="pi0",
            implementation="pi0-pytorch",
            camera_names=self.pi0_config.camera_names,
            state_dim=self.pi0_config.state_dim,
            action_dim=self.pi0_config.action_dim,
            action_horizon=self.pi0_config.action_horizon,
            image_size=self.pi0_config.image_size,
        )

    @property
    def inference_metadata(self) -> dict:
        return {
            "backend": "torch",
            "autocast": self.pi0_config.autocast,
            "autocast_enabled": (
                self.pi0_config.autocast == "auto"
                and next(self.core_model.parameters()).dtype != self.pi0_config.torch_dtype
                and self.pi0_config.torch_dtype != torch.float32
            ),
            "operator_source": str(self.core_source.core_dir),
            "operator_revision": self.core_source.revision,
            "operator_embedded": self.core_source.embedded,
        }

    def artifact_assets(self) -> dict[str, str]:
        return {"data.tokenizer_path": str(self.adapter.pipeline.tokenizer_path)}

    def artifact_metadata(self) -> dict:
        metadata = super().artifact_metadata()
        metadata.update(
            {
                "model_action_dim": self.pi0_config.model_action_dim,
                "max_token_len": self.pi0_config.max_token_len,
                "compute_precision": self.pi0_config.precision,
                "parameter_precision": self.pi0_config.parameter_precision,
                "operator_source": str(self.core_source.core_dir),
                "operator_revision": self.core_source.revision,
                "operator_embedded": self.core_source.embedded,
            }
        )
        return metadata

    def export_backend_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        core_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized = key
            while normalized.startswith(("module.", "_orig_mod.", "_fsdp_wrapped_module.")):
                normalized = normalized.split(".", 1)[1]
            if normalized.startswith("core_model."):
                core_state[normalized.removeprefix("core_model.")] = value
        if not core_state:
            raise RuntimeError("No core_model.* tensors were found in the Pi0 state dict.")
        floating_dtypes = {
            value.dtype for value in core_state.values() if value.is_floating_point()
        }
        if floating_dtypes != {torch.float32}:
            raise RuntimeError(
                "Pi0 artifacts require FP32 training weights, got "
                f"{sorted(str(item) for item in floating_dtypes)}."
            )
        return core_state

    def save_nexus_weights(self) -> bool:
        return False
