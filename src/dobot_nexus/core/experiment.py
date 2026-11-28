from __future__ import annotations

import fcntl
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from dobot_nexus.core.config import Config, save_config


@dataclass
class Experiment:
    config: Config
    run_dir: Path
    logger: logging.Logger

    _lock_fd: int | None = None
    _writer_pid: int | None = None
    _file_handler: logging.FileHandler | None = None
    resolved_config_written: bool = False

    def write_resolved_config(self, config=None):
        if self._writer_pid != os.getpid():
            return
        with NamedTemporaryFile(dir=self.run_dir, prefix=".config_resolved.", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            save_config(self.config if config is None else config, path)
            os.replace(path, self.run_dir / "config_resolved.yaml")
            self.resolved_config_written = True
        finally:
            path.unlink(missing_ok=True)

    def close(self):
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        if self._lock_fd is not None:
            if self._writer_pid == os.getpid():
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = self._writer_pid = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @classmethod
    def from_config(cls, config: Config, *, runtime=None) -> Experiment:
        exp_cfg = config.setdefault("exp", Config())
        name = str(exp_cfg.get("name", f"{config.model.get('name', 'model')}_{config.task}"))
        output_dir = Path(str(exp_cfg.get("output_dir", "runs"))).expanduser()
        if runtime is None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError(
                "Distributed experiments require an initialized runtime. Use BCTrainer(config) "
                "or Experiment.from_config(config, runtime=runtime)."
            )
        is_main = runtime is None or runtime.is_main_process
        run_path = None
        if is_main:
            if bool(exp_cfg.get("auto_suffix", False)):
                name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_path = str((output_dir / name).resolve())
        if runtime is not None:
            run_path = runtime.broadcast_object(run_path)
        run_dir = Path(run_path)
        # Model-owned data preparation runs after Experiment creation. Expose the
        # resolved run path so derived dataset assets stay with this training run.
        exp_cfg["run_dir"] = str(run_dir)
        logger = logging.getLogger(f"dobot_nexus.{run_dir}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            stream = logging.StreamHandler()
            stream.setFormatter(formatter)
            logger.addHandler(stream)
        experiment = cls(config, run_dir, logger)
        error = None
        if is_main:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                fd = os.open(run_dir / ".experiment.lock", os.O_CREAT | os.O_RDWR, 0o644)
                experiment._lock_fd = fd
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Another experiment writer holds the lock.") from exc
                experiment._writer_pid = os.getpid()
                overwrite = exp_cfg.get("overwrite", False)
                if not isinstance(overwrite, bool):
                    raise ValueError("exp.overwrite must be a boolean.")
                if not config.get("train", {}).get("resume_from"):
                    outputs = (run_dir / key for key in ("checkpoints", "token", "rltoken", "cache"))
                    saved = any(path.is_dir() and any(path.iterdir()) for path in outputs)
                    if saved and overwrite:
                        raise RuntimeError("Refusing to overwrite saved artifacts.")
                    if not overwrite and (saved or (run_dir / "config_resolved.yaml").exists()):
                        raise RuntimeError("Populated run directory. Use a distinct exp.name, "
                                           "exp.auto_suffix=true, train.resume_from=..., or exp.overwrite=true.")
                    experiment.write_resolved_config()
                (run_dir / "logs").mkdir(exist_ok=True)
                handler = logging.FileHandler(run_dir / "logs" / f"{config.task}.log")
                handler.setFormatter(logger.handlers[0].formatter)
                experiment._file_handler = handler
                logger.addHandler(handler)
            except Exception as exc:
                experiment.close()
                error = f"{type(exc).__name__}: {exc}"
        if runtime is not None:
            error = runtime.broadcast_object(error)
        if error is not None:
            raise RuntimeError(f"Cannot initialize experiment {run_dir}: {error}")
        return experiment
