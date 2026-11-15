from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dobot_nexus.core.config import to_plain_dict


class NullTracker:
    """No-op tracker used by non-main ranks and runs without tracking."""

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        del metrics, step

    def resume(self, step: int) -> None:
        del step

    def finish(self, *, exit_code: int = 0) -> None:
        del exit_code


class WandbTracker(NullTracker):
    """Small rank-zero-only wrapper around a Weights & Biases run."""

    def __init__(
        self, config: Any, experiment: Any, settings: Mapping[str, Any],
        *, sync_tensorboard: bool = False,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B tracking is enabled but `wandb` is not installed. "
                "Install the project tracking extra with `pip install -e '.[tracking]'`."
            ) from exc

        run_dir = Path(experiment.run_dir)
        wandb_dir = Path(str(settings.get("dir") or run_dir / "wandb")).expanduser()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        exp_config = config.get("exp", {})
        tags = settings.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        init_kwargs = {
            "project": str(settings.get("project") or "dobot-nexus"),
            "name": str(settings.get("name") or exp_config.get("name") or run_dir.name),
            "config": to_plain_dict(config),
            "dir": str(wandb_dir),
            "mode": str(settings.get("mode") or "online"),
            "job_type": "train",
            "tags": [str(tag) for tag in tags],
            "sync_tensorboard": sync_tensorboard,
        }
        for key in ("entity", "group", "notes"):
            value = settings.get(key)
            if value not in (None, ""):
                init_kwargs[key] = str(value)
        self._run = wandb.init(**init_kwargs)
        if self._run is None:
            raise RuntimeError("wandb.init() did not return a run.")
        self._sync_tensorboard = sync_tensorboard
        if sync_tensorboard:
            self._run.define_metric("*", step_metric="global_step")

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        # TensorBoard owns the metrics and global_step when synchronization is on.
        if not self._sync_tensorboard:
            self._run.log(dict(metrics), step=int(step))

    def finish(self, *, exit_code: int = 0) -> None:
        self._run.finish(exit_code=int(exit_code))


def _summary_writer_type():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard tracking requires `pip install -e '.[tensorboard]'`. "
            "For W&B synchronization install '.[tracking,tensorboard]'."
        ) from exc
    return SummaryWriter


class TensorBoardTracker:
    """Local scalar events, flushed at each log point and purged on exact resume."""

    def __init__(self, experiment: Any, settings: Mapping[str, Any]) -> None:
        directory = Path(settings.get("dir") or Path(experiment.run_dir) / "tensorboard")
        flush_secs = settings.get("flush_secs", 10)
        max_queue = settings.get("max_queue", 10)
        if (isinstance(flush_secs, bool) or not isinstance(flush_secs, (int, float))
                or not math.isfinite(flush_secs) or flush_secs <= 0):
            raise ValueError("tracking.tensorboard.flush_secs must be finite and positive.")
        if type(max_queue) is not int or max_queue <= 0:
            raise ValueError("tracking.tensorboard.max_queue must be a positive integer.")
        self.directory = directory.expanduser()
        self._writer_options = dict(log_dir=str(self.directory), flush_secs=flush_secs,
                                    max_queue=max_queue)
        self._writer = _summary_writer_type()(**self._writer_options)
        logger = getattr(experiment, "logger", logging.getLogger(__name__))
        logger.info("TensorBoard events: %s", self.directory)

    def resume(self, step: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("TensorBoard resume step must be a nonnegative integer.")
        self._writer.close()
        # TensorBoard reads files lexicographically. PyTorch's unpadded writer UID
        # sorts 10 before 9 when writers start in the same second. Open the purge
        # event in a later second so it follows all prior files (including stale steps).
        timestamps = []
        for path in self.directory.glob("events.out.tfevents.*"):
            try:
                timestamps.append(int(path.name.split(".")[3]))
            except (ValueError, IndexError):
                continue
        delay = max(timestamps, default=0) + 1 - time.time()
        if delay > 2:
            raise ValueError("TensorBoard event timestamps are in the future; correct the clock "
                             "or set tracking.tensorboard.dir to a fresh directory.")
        if delay > 0:
            time.sleep(delay + 0.001)
        # Hide events from updates newer than the restored optimizer checkpoint.
        self._writer = _summary_writer_type()(**self._writer_options, purge_step=step + 1)

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, float(value), global_step=int(step))
        self._writer.flush()

    def finish(self, *, exit_code: int = 0) -> None:
        del exit_code
        self._writer.close()


class CompositeTracker:
    def __init__(self, *trackers) -> None:
        self._trackers = trackers

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        for tracker in self._trackers:
            tracker.log(metrics, step=step)

    def resume(self, step: int) -> None:
        for tracker in self._trackers:
            tracker.resume(step)

    def finish(self, *, exit_code: int = 0) -> None:
        failure = None
        for tracker in self._trackers:
            try:
                tracker.finish(exit_code=exit_code)
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure


def build_tracker(config: Any, experiment: Any, *, is_main_process: bool):
    """Build local/remote recording only on rank zero; sync avoids duplicate logs."""

    if not is_main_process:
        return NullTracker()
    settings = config.get("tracking", {})
    wb_settings = settings.get("wandb", {})
    tb_settings = settings.get("tensorboard", {})
    wb_enabled = bool(wb_settings.get("enabled", False))
    tb_enabled = bool(tb_settings.get("enabled", False))
    sync = wb_enabled and bool(wb_settings.get("sync_tensorboard", tb_enabled))
    if sync and not tb_enabled:
        raise ValueError("tracking.wandb.sync_tensorboard requires tracking.tensorboard.enabled=true.")
    if tb_enabled:
        _summary_writer_type()  # Diagnose missing dependencies before creating a remote run.
    # W&B patches event writers at init, so it must precede SummaryWriter creation.
    wandb = (WandbTracker(config, experiment, wb_settings, sync_tensorboard=sync)
             if wb_enabled else None)
    try:
        tensorboard = TensorBoardTracker(experiment, tb_settings) if tb_enabled else None
    except BaseException:
        if wandb is not None:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass  # Preserve the writer initialization error.
        raise
    if tensorboard is not None and wandb is not None:
        return CompositeTracker(tensorboard, wandb)
    return tensorboard or wandb or NullTracker()


@contextmanager
def tracking_session(config, experiment, *, is_main_process, resume_step=None):
    """Close optional recorders on success, failure or interrupt without consuming RNG."""
    from dobot_nexus.training.resume import preserve_rng_state

    tracker = NullTracker()
    try:
        with preserve_rng_state():
            tracker = build_tracker(config, experiment, is_main_process=is_main_process)
            if resume_step is not None:
                tracker.resume(resume_step)
        yield tracker
    except BaseException:
        try:
            tracker.finish(exit_code=1)
        except Exception:
            logging.getLogger(__name__).exception("Failed to finish the training tracker.")
        raise
    else:
        tracker.finish(exit_code=0)


def device_metrics(device) -> dict[str, float]:
    """Allocator statistics for this rank; queries do not synchronize the GPU."""
    if device.type != "cuda":
        return {}
    import torch

    return {
        "system/cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        "system/cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
    }
