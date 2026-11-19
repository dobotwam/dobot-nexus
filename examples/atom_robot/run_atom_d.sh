#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/../real_robot/atom_d/run_atom_d.sh" "$@"
