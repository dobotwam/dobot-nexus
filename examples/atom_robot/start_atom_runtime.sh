#!/usr/bin/env bash
set -euo pipefail
ATOM_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ATOM_INVOCATION_DIR="$(pwd)"
ATOM_RUNTIME_PYTHON="${ATOM_PYTHON:-python}"
ATOM_RUNTIME_PYTHON="$(command -v "${ATOM_RUNTIME_PYTHON}")" || {
  echo "ATOM_PYTHON was not found." >&2; exit 1;
}
if [[ "${ATOM_RUNTIME_PYTHON}" != /* ]]; then
  ATOM_RUNTIME_PYTHON="${ATOM_INVOCATION_DIR}/${ATOM_RUNTIME_PYTHON}"
fi
# Keep the caller's cwd so relative SDK, camera and report paths retain meaning.
export PYTHONPATH="${ATOM_REPO_ROOT}/src:${ATOM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ATOM_RUNTIME_PYTHON}" -u -m examples.atom_robot.run "$@"
