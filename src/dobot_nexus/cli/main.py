from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dobot-nexus")
    parser.add_argument("command", choices=["train", "infer", "serve", "tensorboard"])
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments[:1])
    remainder = arguments[1:]
    if args.command == "train":
        from dobot_nexus.cli.train import main as command_main
    elif args.command == "infer":
        from dobot_nexus.cli.infer import main as command_main
    elif args.command == "tensorboard":
        from dobot_nexus.cli.tensorboard import main as command_main
    else:
        from dobot_nexus.cli.serve import main as command_main
    command_main(remainder)


if __name__ == "__main__":
    main()
