from __future__ import annotations

import argparse
from collections.abc import Sequence

from dobot_nexus.core.config import load_config
from dobot_nexus.training import BCTrainer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a VLA with the standard BC interface.")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, list(args.overrides), expected_task="train")
    trainer = BCTrainer(config)
    result = trainer.run()
    trainer.experiment.logger.info(
        "Training complete: step=%d checkpoint=%s", result.global_step, result.checkpoint
    )


if __name__ == "__main__":
    main()
