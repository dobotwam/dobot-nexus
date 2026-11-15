from __future__ import annotations

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
from dobot_nexus.models.pi05.configuration_pi05 import Pi05Config
from dobot_nexus.models.pi05.data_adapter_pi05 import Pi05DataAdapter
from dobot_nexus.models.pi05.torch_core.source import load_core_source
from dobot_nexus.models.pi05.training import load_core_checkpoint


class Pi05Model(BaseVLAModule):
    """Reference Pi0.5 implementation of the public Nexus model contract."""

    def __init__(self, config, *, mode: str) -> None:
        super().__init__()
        if mode not in {"train", "infer"}:
            raise ValueError(f"Pi05Model mode must be 'train' or 'infer', got {mode!r}.")
        self.root_config = config
        self.mode = mode
        self.pi05_config = Pi05Config.from_config(config, mode=mode)
        self.inference_backend = self.pi05_config.inference_backend
        self.core_source = None
        self.core_model = None
        self.loaded_checkpoint = None
        self.torch_engine = None

        self.core_source = load_core_source()
        core_config = self.core_source.Pi0Config(**self.pi05_config.core_kwargs())
        with torch.device("meta"):
            self.core_model = core_config.create()
            if mode == "train":
                from dobot_nexus.models.pi05.training.compile import configure_training_compile

                configure_training_compile(
                    self.core_model, enabled=self.pi05_config.torch_compile
                )
            self._configure_core()
        self.loaded_checkpoint = self._load_core_checkpoint()
        meta_parameters = [
            name for name, value in self.core_model.named_parameters() if value.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                "Pi0.5 checkpoint left uninitialized parameters: "
                + ", ".join(meta_parameters[:8])
            )
        # RLinf keeps FP32 optimizer-master parameters and lets mixed
        # precision cast them to BF16 only while executing operators.
        self.core_model.to(dtype=self.pi05_config.parameter_torch_dtype)
        if mode == "train":
            parameter_dtypes = {value.dtype for value in self.core_model.parameters()}
            if parameter_dtypes != {torch.float32}:
                raise RuntimeError(
                    "RLinf-aligned Pi0.5 training requires every trainable parameter "
                    f"to remain FP32, got {sorted(str(item) for item in parameter_dtypes)}."
                )
        if mode == "train" and bool(
            config.get("train", {}).get("gradient_checkpointing", True)
        ):
            self.core_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.adapter = self._build_adapter(self.core_source.Observation)

    def reset_inference_acceleration(self) -> None:
        """Release captured storage before changing the core's device/weights."""
        if self.torch_engine is not None:
            self.torch_engine.clear()
            self.torch_engine = None

    def _apply(self, fn, recurse=True):
        self.reset_inference_acceleration()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode=True):
        if mode:
            self.reset_inference_acceleration()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_inference_acceleration()
        return super()._load_from_state_dict(*args, **kwargs)

    def _configure_core(self) -> None:
        """Pi0.5-private construction point, before materialization/loading."""
        if self.pi05_config.operator_numerics == "openpi":
            from dobot_nexus.models.pi05.torch_core.numerics import apply_openpi_numerics

            apply_openpi_numerics(self.core_model, self.core_source)

    def _load_core_checkpoint(self):
        return load_core_checkpoint(
            self.core_model, self.pi05_config.checkpoint,
            allow_action_expansion=self.mode == "train",
        )

    def _build_adapter(self, observation_type):
        return Pi05DataAdapter(self.pi05_config, self.root_config, observation_type)

    @classmethod
    def from_config(cls, config, *, mode: str) -> Pi05Model:
        return cls(config, mode=mode)

    def build_train_dataloader(self, config):
        if self.mode != "train":
            raise RuntimeError("A Pi0.5 inference module cannot build a training dataloader.")
        loader, self.resolved_data_config = self.adapter.pipeline.build_dataloader()
        # Extensions construct from a cloned config. Persist shared defaults to
        # the Trainer's config as well, so Artifact config and metadata agree.
        config.model.pop("jax_export_base_checkpoint", None)
        config.model["operator_numerics"] = self.pi05_config.operator_numerics
        config.model["max_token_len"] = self.pi05_config.max_token_len
        config.model["training_recipe"] = self.root_config.model.training_recipe
        return loader

    def training_step(self, batch) -> TrainOutput:
        """Compute the reference flow-matching BC objective."""
        if self.mode != "train":
            raise RuntimeError("training_step is only available in train mode.")
        assert self.core_model is not None
        device = batch["actions"].device
        # Trainer seeds each rank and saves/restores these CPU/CUDA RNG states.
        # Do not re-seed per step or maintain an uncheckpointed model RNG.
        rng = (torch.cuda.default_generators[device.index]
               if device.type == "cuda" else torch.default_generator)
        loss_per_step = self.core_model.compute_loss(
            batch["observation"], batch["actions"], train=True, rng=rng,
            **batch.get("model_kwargs", {}),
        )
        loss = loss_per_step.mean()
        return TrainOutput(
            loss=loss,
            metrics={
                "bc_loss": loss.detach(),
                "flow_loss": loss.detach(),
            },
        )

    def predict_action_chunk(
        self,
        observation: ObservationBatch,
        options: SamplingOptions,
    ) -> PolicyOutput:
        if self.mode != "infer":
            raise RuntimeError("predict_action_chunk is only available in infer mode.")
        assert self.core_model is not None
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
            noise = torch.as_tensor(
                noise, device=parameter.device,
                # Match sample_actions' generated noise and the JAX boundary.
                # Casting supplied noise to the weight dtype also quantizes
                # the Euler state and changes the fixed-noise parity test.
                dtype=getattr(self.core_model, "sampling_noise_dtype", torch.float32),
            )
            if noise.ndim == 2:
                noise = noise[None]
            expected_noise = (
                observation.batch_size,
                self.pi05_config.action_horizon,
                self.pi05_config.model_action_dim,
            )
            if tuple(noise.shape) != expected_noise:
                raise ValueError(
                    f"Pi0.5 Torch noise must have shape {expected_noise} or "
                    f"{expected_noise[1:]}, got {tuple(noise.shape)}."
                )
        num_steps = int(options.num_steps or self.pi05_config.num_steps)
        if num_steps <= 0:
            raise ValueError("SamplingOptions.num_steps must be positive.")

        started = time.perf_counter()
        if self.torch_engine is None:
            from dobot_nexus.models.pi05.openpi_torch import TorchPi05Engine

            self.torch_engine = TorchPi05Engine(
                self.core_model,
                compile=self.pi05_config.torch_compile,
                cuda_graph=self.pi05_config.cuda_graph,
                compile_mode=self.pi05_config.torch_compile_mode,
                preserve_precision=self.pi05_config.torch_compile_preserve_precision,
                max_graphs=self.pi05_config.cuda_graph_max_entries,
            )
        compute_dtype = self.pi05_config.torch_dtype
        with torch.inference_mode(), torch.autocast(
            device_type=parameter.device.type,
            dtype=compute_dtype,
            enabled=parameter.dtype != compute_dtype and compute_dtype != torch.float32,
        ):
            actions = self.torch_engine.sample_actions(
                encoded.observation,
                num_steps=num_steps,
                noise=noise,
                rng=generator,
                **encoded.model_kwargs,
            )
        # Copying to the host is the completion boundary, just as in the JAX
        # engine. Graph replay otherwise reports only CPU dispatch latency.
        host_actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        model_ms = (time.perf_counter() - started) * 1000.0
        output = self.adapter.decode_actions(
            host_actions, aux={"raw_state": encoded.raw_state},
        )
        output.timing.update({
            "model_ms": model_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        })
        return output

    @property
    def policy_spec(self) -> PolicySpec:
        return PolicySpec(
            model_name="pi05",
            implementation="pi05-pytorch",
            camera_names=self.pi05_config.camera_names,
            state_dim=self.pi05_config.state_dim,
            action_dim=self.pi05_config.action_dim,
            action_horizon=self.pi05_config.action_horizon,
            image_size=self.pi05_config.image_size,
        )

    @property
    def inference_metadata(self) -> dict:
        return {
            "backend": "torch",
            **(self.torch_engine.metadata if self.torch_engine is not None else {
                "torch_compile": self.pi05_config.torch_compile,
                "cuda_graph": self.pi05_config.cuda_graph,
            }),
            "operator_source": (
                str(self.core_source.core_dir) if self.core_source is not None else None
            ),
            "operator_revision": (
                self.core_source.revision if self.core_source is not None else None
            ),
            "operator_embedded": (
                self.core_source.embedded if self.core_source is not None else None
            ),
        }

    def artifact_assets(self) -> dict[str, str]:
        return {"data.tokenizer_path": str(self.adapter.pipeline.tokenizer_path)}

    def artifact_metadata(self) -> dict:
        metadata = super().artifact_metadata()
        metadata.update(
            {
                "model_action_dim": self.pi05_config.model_action_dim,
                "max_token_len": self.pi05_config.max_token_len,
                "compute_precision": self.pi05_config.precision,
                "parameter_precision": self.pi05_config.parameter_precision,
                "operator_numerics": self.pi05_config.operator_numerics,
                "normalization": self.adapter.pipeline.normalization,
            }
        )
        if self.core_source is not None:
            metadata.update(
                {
                    "operator_source": str(self.core_source.core_dir),
                    "operator_revision": self.core_source.revision,
                    "operator_embedded": self.core_source.embedded,
                }
            )
        if self.root_config.get("model", {}).get("training_recipe"):
            metadata["training_recipe"] = self.root_config["model"]["training_recipe"]
        return metadata

    def export_backend_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Save bare Pi0 keys so artifacts do not depend on a wrapper class."""
        core_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized = key
            while normalized.startswith(("module.", "_orig_mod.", "_fsdp_wrapped_module.")):
                normalized = normalized.split(".", 1)[1]
            if normalized.startswith("core_model."):
                core_state[normalized.removeprefix("core_model.")] = value
        if not core_state:
            raise RuntimeError("No core_model.* tensors were found in the Pi0.5 state dict.")
        floating_dtypes = {
            value.dtype for value in core_state.values() if value.is_floating_point()
        }
        if floating_dtypes != {torch.float32}:
            raise RuntimeError(
                "Pi0.5 artifacts require FP32 training weights; "
                f"got {sorted(str(item) for item in floating_dtypes)}."
            )
        return core_state

    def save_nexus_weights(self) -> bool:
        # Torch inference restores deployment/pytorch/full_weights.pt directly.
        return False
