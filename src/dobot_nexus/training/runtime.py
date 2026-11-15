from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

from dobot_nexus.core.config import resolve_training_runtime
from dobot_nexus.training.resume import (
    SEED_POLICY,
    TRAINING_STATE_VERSION,
    capture_rng_state,
    validate_training_state,
)
from dobot_nexus.training.seed import set_data_epoch

_LOGGER = logging.getLogger(__name__)
_DEFAULT_FSDP_VERSION = 2

_TORCH_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "32": torch.float32,
    "32-true": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "16": torch.float16,
    "16-mixed": torch.float16,
}


def _torch_dtype(value: Any, *, field: str) -> torch.dtype:
    normalized = str(value).lower()
    try:
        return _TORCH_DTYPES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported {field}={value!r}.") from exc


def _accelerate_precision(value: Any) -> str:
    dtype = _torch_dtype(value, field="runtime.precision")
    return {
        torch.float32: "no",
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
    }[dtype]


def _build_fsdp_plugin(fsdp_cfg):
    """Build a version-specific policy without enabling global Accelerate AMP."""
    try:
        from accelerate.utils import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            FullOptimStateDictConfig,
            FullStateDictConfig,
            MixedPrecision,
            ShardingStrategy,
            StateDictType,
        )
    except ImportError as exc:
        raise ImportError(
            "runtime.fsdp.enabled=true requires an Accelerate build with FSDP support."
        ) from exc

    version = int(fsdp_cfg.get("version", _DEFAULT_FSDP_VERSION))
    if version not in (1, 2):
        raise ValueError(
            f"Unsupported runtime.fsdp.version={version}; expected 1 or 2."
        )
    launch_version = os.environ.get("FSDP_VERSION")
    if launch_version is not None and int(launch_version) != version:
        raise ValueError(
            f"FSDP_VERSION={launch_version} conflicts with runtime.fsdp.version={version}; "
            "use the same version in the launcher and training configuration."
        )
    default_strategy = "no_shard" if version == 1 else "full_shard"
    strategy_name = str(fsdp_cfg.get("sharding_strategy", default_strategy)).lower()
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    if strategy_name not in strategies:
        raise ValueError(
            f"Unsupported runtime.fsdp.sharding_strategy={strategy_name!r}; "
            f"expected one of {sorted(strategies)}."
        )

    auto_wrap_name = str(fsdp_cfg.get("auto_wrap_policy", "no_wrap")).lower()
    auto_wrap_policies = {
        "no_wrap": "NO_WRAP",
        "size_based_wrap": "SIZE_BASED_WRAP",
        "transformer_based_wrap": "TRANSFORMER_BASED_WRAP",
    }
    if auto_wrap_name not in auto_wrap_policies:
        raise ValueError(
            f"Unsupported runtime.fsdp.auto_wrap_policy={auto_wrap_name!r}; "
            f"expected one of {sorted(auto_wrap_policies)}."
        )
    transformer_classes = fsdp_cfg.get(
        "transformer_layer_cls_to_wrap",
        fsdp_cfg.get("transformer_cls_names_to_wrap"),
    )
    if transformer_classes is not None:
        if not isinstance(transformer_classes, (list, tuple)) or not all(
            isinstance(item, str) and item for item in transformer_classes
        ):
            raise ValueError(
                "runtime.fsdp.transformer_layer_cls_to_wrap must be a list of class names."
            )
        transformer_classes = list(transformer_classes)
    if auto_wrap_name == "transformer_based_wrap" and not transformer_classes:
        raise ValueError(
            "runtime.fsdp.transformer_based_wrap requires transformer_layer_cls_to_wrap."
        )

    backward_prefetch_name = str(fsdp_cfg.get("backward_prefetch", "no_prefetch")).lower()
    backward_prefetch_policies = {
        "no_prefetch": None,
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    if backward_prefetch_name not in backward_prefetch_policies:
        raise ValueError(
            f"Unsupported runtime.fsdp.backward_prefetch={backward_prefetch_name!r}; "
            f"expected one of {sorted(backward_prefetch_policies)}."
        )

    common_kwargs = {
        "auto_wrap_policy": auto_wrap_policies[auto_wrap_name],
        "transformer_cls_names_to_wrap": transformer_classes,
        "min_num_params": (
            int(fsdp_cfg.get("min_num_params", 0)) if auto_wrap_name == "size_based_wrap" else None
        ),
        "cpu_offload": bool(fsdp_cfg.get("cpu_offload", False)),
        "activation_checkpointing": False,
    }
    if version == 2:
        return _build_fsdp2_plugin(fsdp_cfg, FullyShardedDataParallelPlugin, common_kwargs)

    policy = MixedPrecision(
        param_dtype=_torch_dtype(fsdp_cfg.get("param_dtype", "bf16"), field="fsdp.param_dtype"),
        reduce_dtype=_torch_dtype(fsdp_cfg.get("reduce_dtype", "fp32"), field="fsdp.reduce_dtype"),
        buffer_dtype=_torch_dtype(fsdp_cfg.get("buffer_dtype", "fp32"), field="fsdp.buffer_dtype"),
    )
    return FullyShardedDataParallelPlugin(
        # Keep the FSDP1 argument supported by the repository's declared
        # accelerate>=0.30 dependency. Accelerate 1.14 only deprecates it.
        sharding_strategy=strategies[strategy_name],
        mixed_precision_policy=policy,
        backward_prefetch=backward_prefetch_policies[backward_prefetch_name],
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        limit_all_gathers=bool(fsdp_cfg.get("limit_all_gathers", False)),
        use_orig_params=bool(fsdp_cfg.get("use_orig_params", False)),
        sync_module_states=bool(fsdp_cfg.get("sync_module_states", True)),
        forward_prefetch=bool(fsdp_cfg.get("forward_prefetch", False)),
        **common_kwargs,
    )


def _build_fsdp2_plugin(fsdp_cfg, plugin_class, common_kwargs):
    try:
        from accelerate.utils.fsdp_utils import (
            fsdp2_prepare_model,
            fsdp2_switch_optimizer_parameters,
        )
        from torch.distributed.fsdp import MixedPrecisionPolicy

        #check that the Accelerate plugin supports the fsdp_version argument and that the FSDP2 prepare APIs are available
        if "fsdp_version" not in inspect.signature(plugin_class).parameters:
            raise ImportError("Accelerate's FSDP plugin has no fsdp_version argument")
        if not callable(fsdp2_prepare_model) or not callable(fsdp2_switch_optimizer_parameters):
            raise ImportError("Accelerate's FSDP2 prepare APIs are unavailable")
    except ImportError as exc:
        raise ImportError(
            "runtime.fsdp.version=2 requires PyTorch's public FSDP2 API and Accelerate's "
            "joint model/optimizer preparation. Use torch==2.10.0 and accelerate==1.14.0 "
            "for the migration checks, or keep runtime.fsdp.version=1."
        ) from exc

    strategy = str(fsdp_cfg.get("sharding_strategy", "full_shard")).lower()
    if strategy not in ("full_shard", "shard_grad_op"):
        raise ValueError(
            f"runtime.fsdp.version=2 does not support sharding_strategy={strategy!r}; "
            "use full_shard or shard_grad_op."
        )
    reshard = fsdp_cfg.get("reshard_after_forward", strategy == "full_shard")
    if not isinstance(reshard, bool):
        raise ValueError("runtime.fsdp.reshard_after_forward must be a boolean for FSDP2.")
    if "sharding_strategy" in fsdp_cfg and reshard != (strategy == "full_shard"):
        raise ValueError("runtime.fsdp.reshard_after_forward conflicts with sharding_strategy.")
    if fsdp_cfg.get("forward_prefetch", False):
        raise ValueError("runtime.fsdp.forward_prefetch=true is not supported with FSDP2.")
    if str(fsdp_cfg.get("backward_prefetch", "no_prefetch")).lower() == "backward_post":
        raise ValueError(
            "runtime.fsdp.backward_prefetch=backward_post is not supported with FSDP2."
        )

    # These FSDP1 controls cannot be passed through to the v2 plugin. In particular,
    # even forward_prefetch=False raises in Accelerate. FSDP2 owns prefetch/rate
    # limiting and always uses per-parameter shards. BCTrainer seeds and loads the
    # same initial model on every rank; it does not rely on a v1 state broadcast.
    legacy_fields = [
        name for name in (
            "use_orig_params", "sync_module_states", "limit_all_gathers", "backward_prefetch"
        ) if name in fsdp_cfg
    ]
    if legacy_fields:
        _LOGGER.warning(
            "FSDP2 does not apply FSDP1 controls %s: it uses per-parameter shards and "
            "implicit prefetch/rate limiting; initial weights must match on every rank.",
            ", ".join(legacy_fields),
        )
    # FSDP2 leaves buffers in their model dtype. prepare() checks the requested
    # buffer_dtype instead of pretending MixedPrecisionPolicy casts them.
    _torch_dtype(fsdp_cfg.get("buffer_dtype", "fp32"), field="fsdp.buffer_dtype")
    policy = MixedPrecisionPolicy(
        param_dtype=_torch_dtype(fsdp_cfg.get("param_dtype", "bf16"), field="fsdp.param_dtype"),
        reduce_dtype=_torch_dtype(fsdp_cfg.get("reduce_dtype", "fp32"), field="fsdp.reduce_dtype"),
        output_dtype=None,
    )
    return plugin_class(
        fsdp_version=2,
        reshard_after_forward=reshard,
        mixed_precision_policy=policy,
        state_dict_type="SHARDED_STATE_DICT",
        cpu_ram_efficient_loading=False,
        **common_kwargs,
    )


def _register_train_output_pytree() -> None:
    """Let FSDP2 discover loss tensors, metrics, and aux outputs in TrainOutput for checkpointing."""
    from torch.utils import _pytree

    from dobot_nexus.core.contracts import TrainOutput

    if TrainOutput not in _pytree.SUPPORTED_NODES:
        names = ("loss", "metrics", "aux")
        _pytree.register_pytree_node(
            TrainOutput,
            lambda output: ([getattr(output, name) for name in names], None),
            lambda children, _context: TrainOutput(*children),
            flatten_with_keys_fn=lambda output: (
                [(_pytree.GetAttrKey(name), getattr(output, name)) for name in names], None
            ),
        )


class LocalRuntime:
    """Small single-process runtime used for development and smoke tests."""

    def __init__(self, config) -> None:
        runtime_cfg = config.get("runtime", {})
        requested = str(runtime_cfg.get("device", "auto"))
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        self.precision = str(runtime_cfg.get("precision", "fp32")).lower()
        self.grad_accum_steps = int(config.train.get("grad_accum_steps", 1))
        if self.grad_accum_steps <= 0:
            raise ValueError("train.grad_accum_steps must be positive.")
        self._micro_step = 0
        self._sync_gradients = True

    @property
    def is_main_process(self) -> bool:
        return True

    @property
    def sync_gradients(self) -> bool:
        return self._sync_gradients

    @property
    def optimizer_step_was_skipped(self) -> bool:
        return False

    @property
    def process_index(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def prepare(self, model, optimizer, data_loader, scheduler):
        model.to(self.device)
        return model, optimizer, data_loader, scheduler

    @contextmanager
    def accumulate(self, model):
        del model
        self._micro_step += 1
        self._sync_gradients = self._micro_step % self.grad_accum_steps == 0
        yield

    def autocast(self):
        dtype = _torch_dtype(self.precision, field="runtime.precision")
        if self.device.type != "cuda" or dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def backward(self, loss) -> None:
        (loss / self.grad_accum_steps).backward()

    def require_finite(self, value, *, label: str) -> None:
        """All ranks fail together before backward or an optimizer update."""
        detached = value.detach()
        # FSDP2 may return a DTensor norm; gather its scalar before reducing flags.
        if hasattr(detached, "full_tensor"):
            detached = detached.full_tensor()
        flag = torch.isfinite(detached).all().to(device=self.device, dtype=torch.int32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        if not bool(flag.item()):
            raise FloatingPointError(f"Non-finite {label}; training stopped before optimizer update.")

    def clip_grad_norm_(self, parameters, max_norm: float):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model):
        return model

    def model_state_dict(self, model) -> dict[str, Any]:
        return self.unwrap_model(model).state_dict()

    def optimizer_state_dict(self, model, optimizer) -> dict[str, Any]:
        del model
        return optimizer.state_dict()

    def iter_dataloader(self, data_loader, *, skip_batches: int = 0, epoch: int = 0):
        set_data_epoch(data_loader, epoch)
        iterator = iter(data_loader)
        for _ in range(skip_batches):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Cannot skip {skip_batches} batches in a DataLoader with fewer batches."
                ) from exc
        return iterator

    def save_training_state(
        self,
        *,
        model,
        optimizer,
        scheduler,
        resume_dir: str | Path,
        trainer_state: dict[str, Any],
    ) -> None:
        _save_dcp_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            resume_dir=resume_dir,
            trainer_state=trainer_state,
            process_index=self.process_index,
            world_size=self.world_size,
            is_main_process=self.is_main_process,
        )

    def load_training_state(
        self,
        *,
        model,
        optimizer,
        scheduler,
        checkpoint_dir: str | Path,
        expected_fingerprint: str | None = None,
        trusted_checkpoint: bool = False,
    ) -> dict[str, Any]:
        return _load_dcp_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            checkpoint_dir=checkpoint_dir,
            process_index=self.process_index,
            world_size=self.world_size,
            expected_fingerprint=expected_fingerprint,
            trusted_checkpoint=trusted_checkpoint,
        )

    def wait_for_everyone(self) -> None:
        return None

    def broadcast_object(self, value):
        return value

    def end_training(self) -> None:
        return None


class AccelerateRuntime:
    """Optional Accelerate runtime; its launch configuration owns DDP/FSDP details."""

    def __init__(self, config) -> None:
        try:
            from accelerate import Accelerator
            from accelerate.utils import (
                DistributedType, GradientAccumulationPlugin, InitProcessGroupKwargs,
            )
        except ImportError as exc:
            raise ImportError(
                "runtime.launcher='accelerate' requires `pip install -e '.[accelerate]'`."
            ) from exc
        runtime_cfg = config.runtime
        precision = str(runtime_cfg.get("precision", "bf16")).lower()
        fsdp_cfg = runtime_cfg.get("fsdp", {})
        self.fsdp_enabled = bool(fsdp_cfg.get("enabled", False))
        fsdp_plugin = _build_fsdp_plugin(fsdp_cfg) if self.fsdp_enabled else None
        self.fsdp_version = (
            int(fsdp_cfg.get("version", _DEFAULT_FSDP_VERSION)) if self.fsdp_enabled else None
        )
        self.fsdp_use_orig_params = bool(
            fsdp_plugin.use_orig_params if fsdp_plugin is not None else False
        )
        if self.fsdp_version == 2:
            self.fsdp_buffer_dtype = _torch_dtype(
                fsdp_cfg.get("buffer_dtype", "fp32"), field="fsdp.buffer_dtype"
            )
            # FSDP2 requires that the pytree discover loss tensors, metrics, and aux outputs in TrainOutput for checkpointing.
            _register_train_output_pytree()
        
        # Explicit FSDP MixedPrecision owns the PI operator casts. Keeping
        # Accelerate AMP disabled preserves the FP32 islands inside the model.
        mixed_precision = "no" if self.fsdp_enabled else _accelerate_precision(precision)
        accelerator_kwargs = {}
        timeout = runtime_cfg.get("distributed_timeout_seconds")
        if timeout is not None:
            if type(timeout) is not int or timeout <= 0:
                raise ValueError("runtime.distributed_timeout_seconds must be a positive integer.")
            accelerator_kwargs["kwargs_handlers"] = [
                InitProcessGroupKwargs(timeout=timedelta(seconds=timeout))
            ]
        self.accelerator = Accelerator(
            # BCTrainer counts full optimizer updates and carries partial windows
            # across epochs, matching LocalRuntime's loss scaling.
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=int(config.train.get("grad_accum_steps", 1)),
                sync_with_dataloader=False,
            ),
            mixed_precision=mixed_precision,
            fsdp_plugin=fsdp_plugin,
            step_scheduler_with_optimizer=False,
            rng_types=["generator"],
            **accelerator_kwargs,
        )
        if self.fsdp_enabled and self.accelerator.distributed_type != DistributedType.FSDP:
            raise RuntimeError(
                "runtime.fsdp.enabled=true requires an FSDP launch. Use `torchrun "
                "--standalone --nproc_per_node=<gpu_count> -m dobot_nexus.cli.main train ...`."
            )
        if self.fsdp_enabled:
            actual = getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1)
            if actual != self.fsdp_version:
                raise RuntimeError(
                    f"Accelerate selected FSDP{actual}, but the configuration requested "
                    f"FSDP{self.fsdp_version}. Start this run in fresh launcher processes."
                )
            # Persist an omitted default in artifacts and the resume fingerprint;
            # an older unversioned FSDP1 run must not silently resume with FSDP2.
            fsdp_cfg.setdefault("version", self.fsdp_version)
        self.device = self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        return bool(self.accelerator.is_main_process)

    @property
    def sync_gradients(self) -> bool:
        return bool(self.accelerator.sync_gradients)

    @property
    def optimizer_step_was_skipped(self) -> bool:
        return bool(self.accelerator.optimizer_step_was_skipped)

    @property
    def process_index(self) -> int:
        return int(self.accelerator.process_index)

    @property
    def world_size(self) -> int:
        return int(self.accelerator.num_processes)

    def prepare(self, model, optimizer, data_loader, scheduler):
        if self.fsdp_enabled:
            if optimizer.state:
                raise RuntimeError("FSDP optimizer must be prepared before its first update.")
            if self.fsdp_version == 2:
                for name, buffer in model.named_buffers():
                    if buffer.is_floating_point() and buffer.dtype != self.fsdp_buffer_dtype:
                        raise ValueError(
                            f"FSDP2 keeps model buffer {name!r} at {buffer.dtype}, but "
                            f"runtime.fsdp.buffer_dtype requests {self.fsdp_buffer_dtype}. "
                            "Set the buffer dtype in the model or match the configuration."
                        )
                # FSDP2 replaces parameters with DTensors. Accelerate must see the
                # model and optimizer together to remap every original group.
                return self.accelerator.prepare(model, optimizer, data_loader, scheduler)
            # This is Accelerate's supported FSDP1 path when the optimizer was
            # built before wrapping. Original parameters preserve arbitrary
            # optimizer groups while FSDP owns their sharded storage.
            if self.fsdp_use_orig_params:
                return self.accelerator.prepare(model, optimizer, data_loader, scheduler)
            if len(optimizer.param_groups) != 1:
                raise ValueError(
                    "The RLinf-aligned FSDP1 path currently requires one optimizer parameter group."
                )
            model = self.accelerator.prepare_model(model)
            # FSDP1 with use_orig_params=False replaces the original model
            # parameters with FlatParameter objects. Accelerate 1.14 does not
            # remap an optimizer created before prepare_model(), so point the
            # existing group at the wrapped parameters before preparing it.
            optimizer.param_groups[0]["params"] = list(model.parameters())
            optimizer = self.accelerator.prepare_optimizer(optimizer)
            data_loader = self.accelerator.prepare_data_loader(data_loader)
            scheduler = self.accelerator.prepare_scheduler(scheduler)
            return model, optimizer, data_loader, scheduler
        return self.accelerator.prepare(model, optimizer, data_loader, scheduler)

    def accumulate(self, model):
        return self.accelerator.accumulate(model)

    def autocast(self):
        return self.accelerator.autocast()

    def backward(self, loss) -> None:
        self.accelerator.backward(loss)

    def require_finite(self, value, *, label: str) -> None:
        """All ranks fail together before backward or an optimizer update."""
        detached = value.detach()
        # FSDP2 may return a DTensor norm; gather its scalar before reducing flags.
        if hasattr(detached, "full_tensor"):
            detached = detached.full_tensor()
        flag = torch.isfinite(detached).all().to(device=self.device, dtype=torch.int32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        if not bool(flag.item()):
            raise FloatingPointError(f"Non-finite {label}; training stopped before optimizer update.")

    def clip_grad_norm_(self, parameters, max_norm: float):
        return self.accelerator.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model)

    def model_state_dict(self, model) -> dict[str, Any]:
        return self.accelerator.get_state_dict(model)

    def optimizer_state_dict(self, model, optimizer) -> dict[str, Any]:
        if not self.fsdp_enabled:
            return optimizer.state_dict()
        if self.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_optimizer_state_dict,
            )

            return get_optimizer_state_dict(
                model,
                _unwrap_optimizer(optimizer),
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        plugin = self.accelerator.state.fsdp_plugin
        with FSDP.state_dict_type(
            model,
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        ):
            return FSDP.optim_state_dict(model, optimizer)

    def iter_dataloader(self, data_loader, *, skip_batches: int = 0, epoch: int = 0):
        set_data_epoch(data_loader, epoch)
        if skip_batches:
            data_loader = self.accelerator.skip_first_batches(data_loader, skip_batches)
            # skip_first_batches creates a fresh shard with iteration=0.
            set_data_epoch(data_loader, epoch)
        return iter(data_loader)

    def save_training_state(
        self,
        *,
        model,
        optimizer,
        scheduler,
        resume_dir: str | Path,
        trainer_state: dict[str, Any],
    ) -> None:
        _save_dcp_training_state(
            model=model,
            optimizer=_unwrap_optimizer(optimizer),
            scheduler=scheduler,
            scaler=getattr(self.accelerator, "scaler", None),
            resume_dir=resume_dir,
            trainer_state=trainer_state,
            process_index=self.process_index,
            world_size=self.world_size,
            is_main_process=self.is_main_process,
        )

    def load_training_state(
        self,
        *,
        model,
        optimizer,
        scheduler,
        checkpoint_dir: str | Path,
        expected_fingerprint: str | None = None,
        trusted_checkpoint: bool = False,
    ) -> dict[str, Any]:
        return _load_dcp_training_state(
            model=model,
            optimizer=_unwrap_optimizer(optimizer),
            scheduler=scheduler,
            scaler=getattr(self.accelerator, "scaler", None),
            checkpoint_dir=checkpoint_dir,
            process_index=self.process_index,
            world_size=self.world_size,
            expected_fingerprint=expected_fingerprint,
            trusted_checkpoint=trusted_checkpoint,
        )

    def wait_for_everyone(self) -> None:
        self.accelerator.wait_for_everyone()

    def broadcast_object(self, value):
        if self.world_size == 1:
            return value
        import torch.distributed as dist

        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def end_training(self) -> None:
        self.accelerator.end_training()


def build_runtime(config):
    config.runtime = resolve_training_runtime(config)
    launcher = str(config.get("runtime", {}).get("launcher", "local")).lower()
    if launcher == "local":
        return LocalRuntime(config)
    if launcher == "accelerate":
        return AccelerateRuntime(config)
    raise ValueError(f"Unsupported runtime.launcher={launcher!r}.")


def _unwrap_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def _state_dict_options(*, strict: bool = True):
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions
    except ImportError as exc:
        raise ImportError(
            "Training resume requires torch.distributed.checkpoint.state_dict."
        ) from exc
    return StateDictOptions(full_state_dict=False, cpu_offload=True, strict=strict)


def _save_dcp_training_state(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    resume_dir: str | Path,
    trainer_state: dict[str, Any],
    process_index: int,
    world_size: int,
    is_main_process: bool,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    rng_state = capture_rng_state()
    root = Path(resume_dir)
    dcp_dir = root / "dcp"
    rng_dir = root / "rng"
    dcp_dir.mkdir(parents=True, exist_ok=True)
    rng_dir.mkdir(parents=True, exist_ok=True)
    options = _state_dict_options()
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=dcp_dir,
        no_dist=world_size == 1,
    )
    torch.save(
        {
            "rng_state": rng_state,
            "data_state": trainer_state.get("data_state", {}),
            "process_index": process_index,
        },
        rng_dir / f"rank_{process_index:05d}.pt",
    )
    if is_main_process:
        payload = dict(trainer_state)
        payload.pop("data_state", None)
        payload["training_state_version"] = TRAINING_STATE_VERSION
        payload["scheduler_step_unit"] = "optimizer_update"
        payload["seed_policy"] = SEED_POLICY
        payload["world_size"] = world_size
        payload["scheduler"] = scheduler.state_dict() if scheduler is not None else None
        payload["scaler"] = scaler.state_dict() if scaler is not None else None
        torch.save(payload, root / "trainer_state.pt")


def _load_dcp_training_state(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_dir: str | Path,
    process_index: int,
    world_size: int,
    expected_fingerprint: str | None = None,
    trusted_checkpoint: bool = False,
) -> dict[str, Any]:
    if trusted_checkpoint is not True:
        raise ValueError("DCP resume contains executable metadata; only load trusted checkpoints "
                         "with train.trust_resume_checkpoint=true. Use deployment weights otherwise.")
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    root = Path(checkpoint_dir) / "resume"
    payload = torch.load(root / "trainer_state.pt", map_location="cpu", weights_only=True)
    validate_training_state(
        payload, world_size=world_size, expected_fingerprint=expected_fingerprint
    )
    # Every rank checks the same file set before entering DCP collectives.
    for rank in range(world_size):
        rng_path = root / "rng" / f"rank_{rank:05d}.pt"
        if not rng_path.is_file():
            raise FileNotFoundError(f"Per-rank RNG state not found: {rng_path}")
    rank_state = torch.load(
        root / "rng" / f"rank_{process_index:05d}.pt", map_location="cpu", weights_only=True
    )
    options = _state_dict_options()
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    reader = dcp.FileSystemReader(root / "dcp")
    metadata = reader.read_metadata()
    if not isinstance(metadata.planner_data, dict):
        raise ValueError("Resume checkpoint is missing its optimizer state layout.")
    # get_state_dict initializes a fresh Adam optimizer for every parameter.
    # A trained optimizer only has moments for parameters that received a
    # gradient. Preserve that lazy state (including a missing step counter)
    # instead of requesting nonexistent moments or inventing an extra update.
    saved_optimizer_parameters = {
        path[2]
        for path in metadata.planner_data.values()
        if isinstance(path, (tuple, list))
        and len(path) >= 4
        and tuple(path[:2]) == ("optimizer", "state")
    }
    optimizer_state["state"] = {
        name: state for name, state in optimizer_state["state"].items()
        if name in saved_optimizer_parameters
    }
    dcp.load(
        {"model": model_state, "optimizer": optimizer_state},
        storage_reader=reader,
        no_dist=world_size == 1,
    )
    incompatible = set_model_state_dict(model, model_state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "DCP model state does not match the current model: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}."
        )
    # DCP above still strictly requires every model tensor and every tensor of
    # each saved optimizer state. Only the assignment of absent, never-used
    # optimizer entries is permissive; it must not fabricate Adam moments.
    set_optimizer_state_dict(
        model, optimizer, optimizer_state, options=_state_dict_options(strict=False)
    )
    scheduler_state = payload.get("scheduler")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = payload.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    payload["rng_state"] = rank_state["rng_state"]
    payload["data_state"] = rank_state["data_state"]
    return payload
