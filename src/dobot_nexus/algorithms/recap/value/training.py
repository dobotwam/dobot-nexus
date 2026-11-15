"""Dobot Fusion training with episode validation and per-rank resumable state."""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dobot_nexus.core.config import to_plain_dict
from dobot_nexus.modules.value.fusion import build_value_model
from dobot_nexus.training.resume import preserve_rng_state
from dobot_nexus.training.seed import seed_all
from dobot_nexus.training.tracking import device_metrics, tracking_session

from ..annotations import check_return_job, read_returns
from ..data import build_frame_index
from .checkpoint import (
    capture_value_rng_state,
    load_value_checkpoint,
    restore_value_rng_state,
    save_value_checkpoint,
)
from .dataset import ReCapValueDataset
from .distributed import value_runtime
from .processing import ReCapValueProcessor, forward_value, load_state_stats

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValueTrainResult:
    checkpoint: Path
    global_step: int
    metrics: dict[str, float]
    best_checkpoint: Path | None = None


def split_episodes(index, train):
    episodes = [e.episode_index for e in index.episodes]
    count = train.val_episodes
    if count is None:
        count = max(1, round(len(episodes) * train.val_ratio)) if train.val_ratio > 0 else 0
    if not 0 <= count < len(episodes):
        raise ValueError("Validation split must leave at least one training episode.")
    permutation = np.random.default_rng(train.seed).permutation(episodes).tolist()
    return sorted(permutation[count:]), sorted(permutation[:count])


def learning_rate_multiplier(config):
    warmup, decay, total = config.warmup_steps, config.decay_steps, config.max_steps
    stable = config.stable_steps
    requested = warmup + decay + (stable or 0)
    if requested > total:
        scale = total / requested
        warmup = int(warmup * scale)
        decay = int(decay * scale) if stable is not None else total - warmup
    stable = max(0, total - warmup - decay) if stable is None or requested > total else stable
    minimum = config.decay_lr / config.learning_rate

    def multiplier(step):
        if warmup and step < warmup:
            return minimum + (1 - minimum) * step / warmup
        if step < warmup + stable:
            return 1.0
        if decay and step < warmup + stable + decay:
            progress = (step - warmup - stable) / decay
            return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
        return minimum

    return multiplier


def _signature(config):
    train = asdict(config.train)
    for key in (
        "resume_from",
        "output_dir",
        "num_workers",
        "save_every",
        "log_every",
        "stop_steps",
        "cache_files",
    ):
        train.pop(key)
    return to_plain_dict({"train": train, "value": asdict(config.value)})


@torch.inference_mode()
def validate(model, loader, processor, device, runtime=None):
    model.eval()
    total, loss, error = 0, 0.0, 0.0
    for batch in loader:
        batch = processor.to_device(batch, device)
        out = forward_value(model, batch, with_targets=True)
        size = len(batch.keys)
        total += size
        loss += float(out.loss) * size
        error += float((out.values - batch.targets).abs().sum())
    model.train()
    if runtime is not None:
        total, loss, error = runtime.sum([total, loss, error])
    return {"val_loss": loss / total, "val_mae": error / total}


def train_value(config) -> ValueTrainResult:
    with value_runtime(config.train.device) as runtime:
        return _train_value(config, runtime)


def _train_value(config, runtime) -> ValueTrainResult:
    with ExitStack() as resources:
        return _train_value_impl(config, runtime, resources)


def _train_value_impl(config, runtime, resources) -> ValueTrainResult:
    device = runtime.device
    stop_steps = config.train.stop_steps or config.train.max_steps
    index = build_frame_index(config.dataset)
    returned = read_returns(config.annotation_dir, expected_index=index)
    check_return_job(returned, config)
    output_dir = config.train.output_dir
    if not config.train.resume_from and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Value output is not empty; use resume_from or a fresh directory.")
    seed_all(config.train.seed)
    train_eps, val_eps = split_episodes(index, config.train)
    saved = None
    if config.train.resume_from:
        loaded = load_value_checkpoint(config.train.resume_from, device=device, config=config)
        if (
            loaded.dataset_signature != index.dataset_signature
            or loaded.scale != returned.scale
            or loaded.returns_id != returned.manifest["files"]["returns.parquet"]
        ):
            raise ValueError("Cannot resume value training on different data or returns.")
        saved = torch.load(config.train.resume_from, map_location="cpu", weights_only=True)
        state = saved.get("training_state")
        if not state or state.get("signature") != _signature(config):
            raise ValueError("Checkpoint is not resumable with this training configuration.")
        if state.get("world_size", 1) != runtime.world_size:
            raise ValueError("Value resume requires the original world size.")
        if state["train_episodes"] != train_eps or state["val_episodes"] != val_eps:
            raise ValueError("Resume episode split differs from the saved split.")
        model, processor = loaded.model, loaded.processor
    else:
        model = build_value_model(config.value).to(device=device)
        stats = (
            load_state_stats(config.dataset.path, config.dataset.state_key)
            if config.value.include_state
            else None
        )
        processor = ReCapValueProcessor(config.value, state_stats=stats)
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    dataset = ReCapValueDataset(config, index, returned, cache_files=config.train.cache_files)

    def subset(episodes):
        selected = set(episodes)
        return torch.utils.data.Subset(
            dataset, [i for i, k in enumerate(index.keys) if k.episode_index in selected]
        )

    train_subset = subset(train_eps)
    sampler = (
        torch.utils.data.DistributedSampler(
            train_subset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=config.train.seed,
            drop_last=True,
        )
        if runtime.distributed
        else None
    )
    generator = torch.Generator().manual_seed(config.train.seed + runtime.rank)
    loader_options = dict(
        num_workers=config.train.num_workers,
        persistent_workers=config.train.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        collate_fn=processor.collate,
        generator=generator,
        **loader_options,
    )
    if not len(loader):
        raise ValueError("Training split must contain at least one full value batch.")
    val_subset = subset(val_eps)
    if runtime.distributed:
        val_subset = torch.utils.data.Subset(
            val_subset, range(runtime.rank, len(val_subset), runtime.world_size)
        )
    val_loader = (
        torch.utils.data.DataLoader(
            val_subset,
            batch_size=config.train.batch_size,
            shuffle=False,
            collate_fn=processor.collate,
            **loader_options,
        )
        if val_eps
        else None
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier(config.train))
    training_model = model
    if runtime.distributed:
        training_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        seed_all(config.train.seed + runtime.rank)
        if runtime.rank == 0:
            LOGGER.info(
                "value distributed ranks=%s batch_per_rank=%s accum=%s global_batch=%s "
                "stop_steps=%s schedule_steps=%s",
                runtime.world_size,
                config.train.batch_size,
                config.train.grad_accum_steps,
                runtime.world_size * config.train.batch_size * config.train.grad_accum_steps,
                stop_steps,
                config.train.max_steps,
            )
    start, epoch, consumed, best = 0, 0, 0, float("inf")
    if saved:
        state = saved["training_state"]
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, epoch, consumed, best = (
            saved["step"],
            state["epoch"],
            state["consumed"],
            state["best"],
        )
        rank_state = state.get("rank_states", [state])[runtime.rank]
        generator.set_state(rank_state["epoch_generator"])
    if sampler is not None:
        sampler.set_epoch(epoch)
    epoch_generator = generator.get_state()
    iterator = iter(loader)
    for _ in range(consumed):
        next(iterator)
    if saved:
        restore_value_rng_state(rank_state["rng"])
    tracker = resources.enter_context(tracking_session(
        {"tracking": config.tracking, "train": asdict(config.train), "value": asdict(config.value)},
        SimpleNamespace(run_dir=output_dir, logger=LOGGER), is_main_process=runtime.rank == 0,
        resume_step=start if saved else None,
    ))
    metrics, checkpoint = {}, None
    for step in range(start + 1, stop_steps + 1):
        started = time.monotonic()
        data_seconds = 0.0
        optimizer.zero_grad(set_to_none=True)
        sums = {"value_loss": 0.0, "expected_value_mae": 0.0}
        epoch_finished = False
        for _ in range(config.train.grad_accum_steps):
            if consumed == len(loader):
                epoch += 1
                consumed = 0
                epoch_generator = generator.get_state()
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
            waiting = time.monotonic()
            batch = next(iterator)
            data_seconds += time.monotonic() - waiting
            batch = processor.to_device(batch, device)
            consumed += 1
            epoch_finished |= consumed == len(loader)
            out = forward_value(training_model, batch, with_targets=True)
            (out.loss / config.train.grad_accum_steps).backward()
            sums["value_loss"] += float(out.loss.detach())
            sums["expected_value_mae"] += float((out.values.detach() - batch.targets).abs().mean())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        metrics = {k: v / config.train.grad_accum_steps for k, v in sums.items()}
        if runtime.distributed:
            metrics = dict(
                zip(
                    metrics,
                    [v / runtime.world_size for v in runtime.sum(metrics.values())],
                    strict=True,
                )
            )
        metrics.update(grad_norm=float(norm), lr=optimizer.param_groups[0]["lr"])
        metrics.update(step_seconds=time.monotonic() - started, data_seconds=data_seconds)
        improved = False
        if val_loader is not None and (epoch_finished or step == stop_steps):
            with preserve_rng_state():
                metrics.update(validate(model, val_loader, processor, device, runtime))
            improved = metrics["val_loss"] < best
            best = min(best, metrics["val_loss"])
        log_step = step == 1 or step % config.train.log_every == 0 or step == stop_steps
        if runtime.rank == 0 and log_step:
            metrics.update(device_metrics(device))
            LOGGER.info("value step=%s metrics=%s", step, metrics)
            with preserve_rng_state():
                tracker.log(metrics, step=step)
        elif runtime.rank == 0 and "val_loss" in metrics:
            with preserve_rng_state():
                tracker.log({k: v for k, v in metrics.items() if k.startswith("val_")}, step=step)
        if step % config.train.save_every == 0 or step == stop_steps or improved:
            rank_states = runtime.gather(
                {
                    "epoch_generator": epoch_generator,
                    "rng": capture_value_rng_state(),
                }
            )
            state = {
                "signature": _signature(config),
                "epoch": epoch,
                "consumed": consumed,
                "epoch_generator": epoch_generator,
                "rng": rank_states[0]["rng"],
                "world_size": runtime.world_size,
                "rank_states": rank_states,
                "scheduler": scheduler.state_dict(),
                "best": best,
                "train_episodes": train_eps,
                "val_episodes": val_eps,
            }
            kwargs = dict(
                model=model,
                config=config,
                scale=returned.scale,
                step=step,
                optimizer_state=optimizer.state_dict(),
                state_stats=processor.state_stats,
                training_state=state,
            )
            checkpoint = output_dir / f"step_{step:08d}.pt"
            if runtime.rank == 0:
                save_value_checkpoint(checkpoint, **kwargs)
                # Atomically replace aliases without copying large checkpoints.
                output_dir.mkdir(parents=True, exist_ok=True)
                for alias in ["last.pt", "best.pt"] if improved else ["last.pt"]:
                    temporary = output_dir / ("." + alias)
                    temporary.unlink(missing_ok=True)
                    os.link(checkpoint, temporary)
                    os.replace(temporary, output_dir / alias)
            runtime.barrier()
    if checkpoint is None:
        if start == stop_steps and config.train.resume_from:
            checkpoint = config.train.resume_from
        else:
            raise ValueError("Checkpoint step exceeds the requested training stop step.")
    best_path = output_dir / "best.pt"
    return ValueTrainResult(
        checkpoint, stop_steps, metrics, best_path if best_path.is_file() else None
    )
