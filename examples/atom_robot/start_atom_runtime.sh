#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/../real_robot/atom_w_0759/start_atom_runtime.sh" "$@"
