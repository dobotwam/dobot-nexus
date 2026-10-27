from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from dobot_nexus.core.config import algorithm_target, load_config
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.core.import_utils import import_from_string
from dobot_nexus.training import BCTrainer


def trainer_class(config):
    """Validate driver ownership before Experiment creates files."""
    target = algorithm_target(config)
    if config.get("runtime", {}).get("launcher", "local") == "rlinf":
        if target is None:
            raise ValueError("runtime.launcher=rlinf requires algorithm.target.")
        if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
            raise ValueError("RLinf requires one driver; do not launch it with torchrun.")
    return import_from_string(target) if target else BCTrainer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train using the configured algorithm.")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, list(args.overrides), expected_task="train")
    trainer_type = trainer_class(config)
    if trainer_type is BCTrainer:
        trainer = BCTrainer(config)
        experiment = trainer.experiment
    else:
        experiment = Experiment.from_config(config)
        trainer = trainer_type(config, experiment)
    result = trainer.run()
    experiment.logger.info(
        "Training complete: step=%d checkpoint=%s", result.global_step, result.checkpoint
    )


if __name__ == "__main__":
    main()
