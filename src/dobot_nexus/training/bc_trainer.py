from __future__ import annotations

import math
import time
from itertools import chain
from typing import Any

import torch

from dobot_nexus.core.artifacts import ArtifactManager, resume_fingerprint
from dobot_nexus.core.config import Config
from dobot_nexus.core.contracts import TrainOutput, TrainResult
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.core.import_utils import build_model
from dobot_nexus.training.ema import ParameterEMA
from dobot_nexus.training.resume import (
    capture_rng_state,
    data_rng_state,
    preserve_rng_state,
    resolve_resume_checkpoint,
    restore_data_rng_state,
    restore_rng_state,
    restore_worker_generator,
)
from dobot_nexus.training.runtime import build_runtime
from dobot_nexus.training.seed import seed_all, seed_dataloader, training_seed
from dobot_nexus.training.tracking import NullTracker, build_tracker, device_metrics


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def _loader_length(data_loader) -> int | None:
    try:
        return len(data_loader)
    except TypeError:
        return None


def _build_optimizer(config, model):
    """build an optimizer from config.
            Only AdamW is supported Now. TODO: add support for other optimizers."""
    optim_cfg = config.get("optim", {})
    name = str(optim_cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Only optim.name='adamw' is supported in v1, got {name!r}.")
    return torch.optim.AdamW(
        model.get_param_groups(optim_cfg),
        lr=float(optim_cfg.get("lr", 1e-5)),
        betas=tuple(float(item) for item in optim_cfg.get("betas", [0.9, 0.95])),
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )


def _build_scheduler(config, optimizer):
    """build a learning rate scheduler from config.
            Only warmup_cosine and constant are supported Now.
            TODO: add support for other schedulers."""
    scheduler_cfg = config.get("scheduler", {})
    name = str(scheduler_cfg.get("name", "warmup_cosine")).lower()
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    if name != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler.name={name!r}.")
    warmup = int(scheduler_cfg.get("warmup_steps", 0))
    total = int(scheduler_cfg.get("total_steps", config.train.max_steps))
    min_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

    initial = scheduler_cfg.get("warmup_initial_ratio")
    if initial is not None and (not math.isfinite(float(initial)) or not 0 <= initial <= 1):
        raise ValueError("scheduler.warmup_initial_ratio must be in [0,1].")

    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            if initial is not None:
                return float(initial) + (1 - float(initial)) * step / warmup
            return max(step + 1, 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class BCTrainer:
    """Backend-neutral BC loop that calls the algorithm module's training_step.
    The BCTrainer is responsible for:
    - building the model, optimizer, scheduler, and dataloader from config
    - running the training loop, including logging and checkpointing"""

    def __init__(self, config, experiment=None, *, runtime=None) -> None:
        self.config = config
        self.runtime = runtime if runtime is not None else build_runtime(config)
        try:
            self.experiment = (
                experiment if experiment is not None
                else Experiment.from_config(config, runtime=self.runtime)
            )
            try:
                self.artifacts = ArtifactManager(self.experiment.run_dir)
            except BaseException:
                if (close := getattr(self.experiment, "close", None)) is not None:
                    close()
                raise
        except BaseException:
            self.runtime.end_training()
            raise
        self.tracker = NullTracker()

    def run(self) -> TrainResult:
        try:
            with preserve_rng_state():
                self.tracker = build_tracker(
                    self.config, self.experiment,
                    is_main_process=self.runtime.is_main_process,
                )
            result = self._run()
        except BaseException:
            try:
                self.tracker.finish(exit_code=1)
            except Exception:
                self.experiment.logger.exception("Failed to finish the training tracker.")
            raise
        else:
            self.tracker.finish(exit_code=0)
            self.experiment.logger.info(
                "Training complete: step=%d checkpoint=%s", result.global_step, result.checkpoint
            )
            return result
        finally:
            try:
                if (close := getattr(self.experiment, "close", None)) is not None:
                    close()
            finally:
                self.runtime.end_training()

    def _run(self) -> TrainResult:
        config = self.config
        base_seed = config.get("exp", {}).get("seed", 0)
        
        rank_seed = training_seed(base_seed, self.runtime.process_index)
        # Set the global RNG state to the base seed before building the model and dataloader.
        seed_all(base_seed)
        resume_from = config.train.get("resume_from")
        last_checkpoint = None
        if resume_from:
            # Model validation and DCP restoration must see the same checkpoint if "auto" moves.
            last_checkpoint = resolve_resume_checkpoint(self.experiment.run_dir, resume_from)
            config = Config(config, train=Config(config.train, resume_from=str(last_checkpoint)))
        model = build_model(config, mode="train")
        data_loader = model.build_train_dataloader(config)
        seed_dataloader(data_loader, base_seed)
        optimizer = _build_optimizer(config, model)
        scheduler = _build_scheduler(config, optimizer)
        model, optimizer, data_loader, scheduler = self.runtime.prepare(
            model, optimizer, data_loader, scheduler
        )
        decay = config.train.get("ema_decay")
        ema = ParameterEMA(self.runtime.unwrap_model(model), decay) if decay is not None else None
        # Set the global RNG state to the rank-specific seed before starting training.
        seed_all(rank_seed)
        
        if self.runtime.is_main_process:
            # All ranks share base_seed; each training RNG uses base_seed + global_rank.
            self.experiment.logger.info(
                "Training RNG: base_seed=%d training_seed=base_seed+global_rank world_size=%d; "
                "scheduler unit=successful optimizer update",
                base_seed,
                self.runtime.world_size,
            )

        max_steps = int(config.train.max_steps)
        log_every = int(config.train.get("log_every", 10))
        save_every = int(config.train.get("save_every", 1000))
        grad_clip = float(config.train.get("grad_clip_norm", 1.0))
        if max_steps <= 0 or log_every <= 0 or save_every <= 0:
            raise ValueError("train.max_steps, log_every, and save_every must be positive.")

        global_step = 0
        data_epoch = 0
        batches_in_epoch = 0
        # Capture initial dataloader and global RNG states for checkpointing.
        epoch_generator_states = data_rng_state(data_loader)
        epoch_rng_state = capture_rng_state()
        last_metrics: dict[str, float] = {}
        if last_checkpoint is not None:
            restored = self.runtime.load_training_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=last_checkpoint,
                expected_fingerprint=resume_fingerprint(config),
                trusted_checkpoint=config.train.get("trust_resume_checkpoint", False),
            )
            if ema is not None:
                ema.load(last_checkpoint, self.runtime.process_index)
            global_step = int(restored["step"])
            if ema is not None and ema.updates != global_step:
                raise ValueError("EMA update count does not match the resumed optimizer step.")
            if global_step > max_steps:
                raise ValueError(f"Resume step {global_step} exceeds train.max_steps={max_steps}.")
            last_metrics = {
                str(key): float(value) for key, value in restored.get("metrics", {}).items()
            }
            restored_data = restored.get("data_state", {})
            data_epoch = int(restored_data.get("epoch", 0))
            batches_in_epoch = int(restored_data.get("batches_in_epoch", 0))
            saved_loader_length = restored_data.get("loader_length")
            current_loader_length = _loader_length(data_loader)
            if (
                saved_loader_length is not None
                and current_loader_length is not None
                and int(saved_loader_length) != current_loader_length
            ):
                raise ValueError(
                    "Resume DataLoader length changed: "
                    f"checkpoint={saved_loader_length}, current={current_loader_length}."
                )
            epoch_generator_states = restored_data["epoch_generator_states"]
            epoch_rng_state = restored_data["epoch_rng_state"]
            restore_rng_state(epoch_rng_state)
            restore_data_rng_state(data_loader, epoch_generator_states)
            iterator = self.runtime.iter_dataloader(
                data_loader,
                skip_batches=batches_in_epoch,
                epoch=data_epoch,
            )
            # Accelerate initializes its iterator lazily. Prime it before restoring
            # training RNG; otherwise worker/sampler setup can consume that state.
            sentinel = object()
            first_batch = next(iterator, sentinel)
            iterator = chain(() if first_batch is sentinel else (first_batch,), iterator)
            restore_worker_generator(data_loader, restored_data["generator_states"])
            restore_rng_state(restored["rng_state"])
            if (write_config := getattr(self.experiment, "write_resolved_config", None)) is not None:
                write_config(config)
            if self.runtime.is_main_process:
                self.experiment.logger.info(
                    "Resumed checkpoint=%s step=%d data_epoch=%d batches_in_epoch=%d",
                    last_checkpoint,
                    global_step,
                    data_epoch,
                    batches_in_epoch,
                )
        else:
            iterator = self.runtime.iter_dataloader(data_loader, epoch=data_epoch)
        if resume_from:
            with preserve_rng_state():
                self.tracker.resume(global_step)
        window_started = time.perf_counter()
        window_step = global_step
        optimizer.zero_grad(set_to_none=True)

        while global_step < max_steps:
            try:
                raw_batch = next(iterator)
            except StopIteration:
                data_epoch += 1
                batches_in_epoch = 0
                epoch_generator_states = data_rng_state(data_loader)
                epoch_rng_state = capture_rng_state()
                iterator = self.runtime.iter_dataloader(data_loader, epoch=data_epoch)
                raw_batch = next(iterator)
            batches_in_epoch += 1

            with self.runtime.accumulate(model):
                unwrapped = self.runtime.unwrap_model(model)
                batch = unwrapped.adapter.encode_train_batch(raw_batch, self.runtime.device)
                with self.runtime.autocast():
                    output = model(batch)
                if not isinstance(output, TrainOutput):
                    raise TypeError(f"training_step must return TrainOutput, got {type(output)!r}.")
                self.runtime.require_finite(output.loss, label="training loss")
                self.runtime.backward(output.loss)

                if not self.runtime.sync_gradients:
                    continue
                grad_norm = self.runtime.clip_grad_norm_(model.parameters(), grad_clip)
                self.runtime.require_finite(grad_norm, label="gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if self.runtime.optimizer_step_was_skipped:
                    continue
                if ema is not None:
                    ema.update()
                scheduler.step()
                global_step += 1

            last_metrics = {key: _to_float(value) for key, value in output.metrics.items()}
            last_metrics.setdefault("loss", _to_float(output.loss))
            last_metrics["grad_norm"] = _to_float(grad_norm)
            last_metrics["lr"] = float(optimizer.param_groups[0]["lr"])

            if global_step == 1 or global_step % log_every == 0 or global_step == max_steps:
                elapsed = max(time.perf_counter() - window_started, 1e-9)
                last_metrics["steps_per_sec"] = (
                    (global_step - window_step) / elapsed
                )
                window_started = time.perf_counter()
                window_step = global_step
                last_metrics["eta_seconds"] = (
                    (max_steps - global_step) / last_metrics["steps_per_sec"]
                )
                if self.runtime.is_main_process:
                    last_metrics.update(device_metrics(self.runtime.device))
                    formatted = " ".join(
                        f"{key}={value:.6g}" for key, value in last_metrics.items()
                    )
                    self.experiment.logger.info("step=%d %s", global_step, formatted)
                    with preserve_rng_state():
                        self.tracker.log(last_metrics, step=global_step)

            if global_step % save_every == 0 or global_step == max_steps:
                with preserve_rng_state():
                    last_checkpoint = self.artifacts.save(
                        model=model,
                        runtime=self.runtime,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        step=global_step,
                        metrics=last_metrics,
                        data_state={
                            "epoch": data_epoch,
                            "batches_in_epoch": batches_in_epoch,
                            "loader_length": _loader_length(data_loader),
                            "epoch_generator_states": epoch_generator_states,
                            "epoch_rng_state": epoch_rng_state,
                            "generator_states": data_rng_state(data_loader),
                        },
                        final=global_step == max_steps,
                        **({"ema": ema} if ema is not None else {}),
                    )
                if self.runtime.is_main_process:
                    self.experiment.logger.info("Saved checkpoint: %s", last_checkpoint)

        if last_checkpoint is None:
            raise RuntimeError("Training completed without producing or restoring a checkpoint.")
        result = TrainResult(
            run_dir=str(self.experiment.run_dir),
            checkpoint=str(last_checkpoint),
            global_step=global_step,
            metrics=last_metrics,
        )
        return result
