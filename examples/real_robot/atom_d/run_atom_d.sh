#!/usr/bin/env bash
# Use the vendor SDK for execution and a separate GPU environment for Pi0.5.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -f "$ROOT/.env.atom-d" ]]; then
    source "$ROOT/.env.atom-d"
fi
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${ATOM_PYTHON:-$HOME/miniforge3/envs/Atom/bin/python}" -u -m examples.real_robot.atom_d.inference "$@"
