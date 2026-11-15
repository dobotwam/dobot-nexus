from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dobot_nexus.core.config import Config, save_config


@dataclass
class Experiment:
    config: Config
    run_dir: Path
    logger: logging.Logger

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
        error = None
        if is_main:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "logs").mkdir(exist_ok=True)
                save_config(config, run_dir / "config_resolved.yaml")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if runtime is not None:
            error = runtime.broadcast_object(error)
        if error is not None:
            raise RuntimeError(f"Cannot initialize experiment {run_dir}: {error}")

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
            if is_main:
                file_handler = logging.FileHandler(run_dir / "logs" / f"{config.task}.log")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        return cls(config=config, run_dir=run_dir, logger=logger)
