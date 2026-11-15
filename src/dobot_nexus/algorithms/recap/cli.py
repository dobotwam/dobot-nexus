"""Offline RECAP algorithm CLI; policy training uses the standard Nexus CLI."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from dobot_nexus.core.config import to_plain_dict

from .configuration import load_offline_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("returns", "train-value", "label"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
        if command == "label":
            child.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_offline_config(args.config, args.override)
    if args.command == "returns":
        from .labeling import run_returns

        result = {"manifest": str(run_returns(config))}
    elif args.command == "train-value":
        from .value.training import train_value

        result = to_plain_dict(asdict(train_value(config)))
    else:
        from .labeling import run_labeling

        result = {"manifest": str(run_labeling(config, args.checkpoint))}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
