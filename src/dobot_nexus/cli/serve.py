from __future__ import annotations

import argparse
from collections.abc import Sequence

from dobot_nexus.core.config import load_config
from dobot_nexus.inference import load_policy
from dobot_nexus.inference.server import run_server


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the standardized policy API.")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, list(args.overrides), expected_task="infer")
    policy = load_policy(config)
    server = config.get("server", {})
    if bool(config.get("inference", {}).get("warmup", False)):
        policy.warmup()
    run_server(
        policy,
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 12021)),
    )


if __name__ == "__main__":
    main()
