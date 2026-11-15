#!/usr/bin/env python3
"""Validate imports and version boundaries in one model pip environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import subprocess
import sys

MODEL_NAMES = ("pi05", "pi0")
COMMON_IMPORTS = (
    "dobot_nexus",
    "torch",
    "numpy",
    "yaml",
    "accelerate",
    "pandas",
    "pyarrow",
    "imageio",
    "cv2",
    "safetensors",
    "tqdm",
    "tensorboard",
    "torch.utils.tensorboard",
)
MODEL_IMPORTS = {
    "pi05": ("torchvision", "sentencepiece", "einops", "PIL", "datasets", "lerobot"),
    "pi0": ("torchvision", "sentencepiece", "einops", "PIL", "datasets", "lerobot"),
}
EXPECTED_VERSIONS = {}
EXTRA_IMPORTS = {
    "websocket": ("msgpack", "websockets.sync.client", "websockets.asyncio.server"),
    "serve": ("fastapi", "uvicorn"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a model-specific pip environment.")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        choices=tuple(EXTRA_IMPORTS),
        help="Also check an installed service extra; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        capture_output=True,
    )
    failures: list[str] = []
    if pip_check.returncode:
        failures.append(f"pip check: {(pip_check.stdout + pip_check.stderr).strip()}")
    imports = [*COMMON_IMPORTS, *MODEL_IMPORTS[args.model]]
    for extra in args.extra:
        imports.extend(EXTRA_IMPORTS[extra])
    for module_name in dict.fromkeys(imports):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # Import errors include native-library compatibility failures.
            failures.append(f"cannot import {module_name}: {exc}")

    for distribution, expected in EXPECTED_VERSIONS.get(args.model, {}).items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing distribution {distribution}=={expected}")
            continue
        if actual != expected:
            failures.append(f"{distribution} must be {expected}, got {actual}")

    if failures:
        for failure in failures:
            print(f"ERROR {failure}", file=sys.stderr)
        return 1

    extras = ",".join(sorted(set(args.extra))) or "none"
    print(f"OK model={args.model} extras={extras} python={sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
