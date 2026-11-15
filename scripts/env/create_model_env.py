#!/usr/bin/env python3
"""Create an isolated pip environment for one model family."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("pi05", "pi0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/reuse .venvs/<model> and install that model's pip dependencies."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        choices=("websocket", "serve"),
        help="Also install a service extra from pyproject.toml; may be repeated.",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        help="Environment path (default: <repo>/.venvs/<model>).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to create the environment (Python >= 3.10).",
    )
    parser.add_argument(
        "--system-site-packages",
        action="store_true",
        help="Reuse system packages, useful when a GPU image already provides CUDA PyTorch.",
    )
    parser.add_argument(
        "--skip-tool-upgrade",
        action="store_true",
        help="Do not upgrade pip/setuptools/wheel before installing dependencies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without creating or changing the environment.",
    )
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def environment_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def check_python_version(python: str) -> None:
    command = [
        python,
        "-c",
        (
            "import sys; "
            "assert sys.version_info >= (3, 10), "
            "f'Python >= 3.10 is required, got {sys.version.split()[0]}'"
        ),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    venv = (args.venv or REPO_ROOT / ".venvs" / args.model).expanduser().resolve()
    requirements = REPO_ROOT / "src" / "dobot_nexus" / "models" / args.model / "requirements.txt"
    python_in_venv = environment_python(venv)

    if not requirements.is_file():
        raise FileNotFoundError(f"missing model requirements: {requirements}")

    if not python_in_venv.is_file():
        create_command = [args.python, "-m", "venv"]
        if args.system_site_packages:
            create_command.append("--system-site-packages")
        create_command.append(str(venv))
        if not args.dry_run:
            check_python_version(args.python)
        run(create_command, dry_run=args.dry_run)
    else:
        if not args.dry_run:
            check_python_version(str(python_in_venv))
        if args.system_site_packages:
            print(
                "note: --system-site-packages only applies when creating a new environment; "
                f"reusing {venv}",
                file=sys.stderr,
            )

    pip = [str(python_in_venv), "-m", "pip"]
    if not args.skip_tool_upgrade:
        run([*pip, "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=args.dry_run)
    run([*pip, "install", "-r", str(requirements)], dry_run=args.dry_run)
    extras = sorted(set(args.extra))
    # Requirements own the model environment; service extras are owned by pyproject.toml.
    editable = f"{REPO_ROOT}[{','.join(extras)}]" if extras else str(REPO_ROOT)
    no_deps = [] if extras else ["--no-deps"]
    run([*pip, "install", *no_deps, "-e", editable], dry_run=args.dry_run)

    status = "Environment plan" if args.dry_run else "Environment ready"
    print(f"\n{status}: {venv}")
    check_script = REPO_ROOT / "scripts" / "env" / "check_model_env.py"
    check_command = [str(python_in_venv), str(check_script), "--model", args.model]
    for extra in extras:
        check_command.extend(["--extra", extra])
    print(f"Validate with: {shlex.join(check_command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
