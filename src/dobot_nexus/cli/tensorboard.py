"""Launch the optional training dashboard without loading a policy or starting training."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logdir', type=Path, default=Path('runs'),
                        help='Parent of training runs, or a single run/tensorboard directory.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--reload-interval', type=int, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.reload_interval <= 0:
        parser.error('port must be in [1,65535] and reload-interval must be positive.')
    logdir = args.logdir.expanduser().resolve()
    if not logdir.is_dir():
        parser.error(f'Training log directory does not exist: {logdir}')
    try:
        from tensorboard import program
    except ImportError as exc:
        raise ImportError(
            "Training dashboard requires `pip install 'dobot-nexus[tensorboard]'`."
        ) from exc
    dashboard = program.TensorBoard()
    dashboard.configure(argv=['tensorboard', '--logdir', str(logdir), '--host', args.host,
                              '--port', str(args.port),
                              '--reload_interval', str(args.reload_interval)])
    # main() serves in the foreground so Ctrl+C owns the dashboard lifetime.
    result = dashboard.main()
    if result:
        raise SystemExit(result)
