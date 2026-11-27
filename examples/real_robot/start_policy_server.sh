#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_DIR="$(pwd)"

POLICY_PYTHON="${POLICY_PYTHON:-python}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${OPENPI_TOKENIZER_PATH:-}}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
GPU="${GPU:-0}"
NUM_STEPS="${NUM_STEPS:-}"
SEED="${SEED:-42}"
FIXED_SEED="${FIXED_SEED:-0}"
WARMUP="${WARMUP:-1}"

usage() {
  cat <<'EOF'
Usage:
  examples/real_robot/start_policy_server.sh MODEL_PATH [Pi0.5 server options]

MODEL_PATH may point to a Nexus run directory, checkpoints/latest.txt,
step_XXXXXXXX Artifact directory, deployment/pytorch directory, or a file
inside that Artifact such as deployment/pytorch/full_weights.pt.
BC, RECAP, Memory and RLToken profiles are selected from the Artifact automatically.
RECAP supports --recap-condition base|positive|negative. Memory requires
BRIDGE_MODE=memory on both the robot server and robot client launchers.

The same value may be supplied with MODEL_PATH=/path/to/model for automation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -n "${1:-}" && "${1}" != --* ]]; then
  MODEL_PATH="${1}"
  shift
else
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_PATH:-}}"
fi
if [[ -z "${MODEL_PATH}" ]]; then
  usage >&2
  echo "Error: MODEL_PATH is required." >&2
  exit 2
fi

if ! command -v "${POLICY_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${POLICY_PYTHON}" ]]; then
  echo "POLICY_PYTHON was not found or is not executable: ${POLICY_PYTHON}" >&2
  exit 1
fi
# Fix the caller's executable before cd, without resolving virtualenv symlinks.
POLICY_PYTHON="$(command -v "${POLICY_PYTHON}" || printf '%s' "${POLICY_PYTHON}")"
if [[ "${POLICY_PYTHON}" != /* ]]; then
  POLICY_PYTHON="${INVOCATION_DIR}/${POLICY_PYTHON}"
fi
if [[ -n "${TOKENIZER_PATH}" && ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Pi0.5 tokenizer was not found: ${TOKENIZER_PATH}" >&2
  exit 1
fi

LAUNCH_PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if ! TORCH_DEPENDENCY_INFO="$(
  PYTHONPATH="${LAUNCH_PYTHONPATH}" "${POLICY_PYTHON}" -c '
import torch
from dobot_nexus.models.pi05.openpi_torch.engine import TorchPi05Engine  # noqa: F401
print(f"torch={torch.__version__}")
' 2>&1
)"; then
  echo "POLICY_PYTHON does not provide the required Nexus Torch inference API:" >&2
  echo "${TORCH_DEPENDENCY_INFO}" >&2
  echo "Use the Pi0.5 Torch environment with the pi05 and openpi protocol extras installed." >&2
  exit 1
fi

resolved_output="$(
  PYTHONPATH="${LAUNCH_PYTHONPATH}" \
    "${POLICY_PYTHON}" "${SCRIPT_DIR}/shared/resolve_model.py" "${MODEL_PATH}"
)"
mapfile -t resolved_paths <<< "${resolved_output}"
if [[ "${#resolved_paths[@]}" -ne 2 ]]; then
  echo "Model resolver returned an invalid result." >&2
  exit 1
fi
ARTIFACT_DIR="${resolved_paths[0]}"
TORCH_CHECKPOINT="${resolved_paths[1]}"

if [[ -n "${NUM_STEPS}" ]]; then
  set -- --num-steps "${NUM_STEPS}" "$@"
fi
case "${FIXED_SEED}" in
  0) seed_args=() ;;
  1) seed_args=(--fixed-seed) ;;
  *) echo "FIXED_SEED must be 0 or 1, got ${FIXED_SEED}" >&2; exit 1 ;;
esac
case "${WARMUP}" in
  0) warmup_args=(--no-warmup) ;;
  1) warmup_args=() ;;
  *) echo "WARMUP must be 0 or 1, got ${WARMUP}" >&2; exit 1 ;;
esac
if [[ -n "${TOKENIZER_PATH}" ]]; then
  export OPENPI_TOKENIZER_PATH="${TOKENIZER_PATH}"
fi
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"

echo "Starting Pi0.5 policy service" >&2
echo "  model:    ${MODEL_PATH}" >&2
echo "  artifact: ${ARTIFACT_DIR}" >&2
echo "  weights:  ${TORCH_CHECKPOINT}" >&2
echo "  tokenizer: ${TOKENIZER_PATH:-from Artifact}" >&2
echo "  runtime:  ${TORCH_DEPENDENCY_INFO}" >&2
echo "  endpoint: ws://${POLICY_HOST}:${POLICY_PORT}" >&2
echo "  gpu:      ${GPU}" >&2

cd "${REPO_ROOT}"
PYTHONPATH="${LAUNCH_PYTHONPATH}" exec "${POLICY_PYTHON}" -m dobot_nexus.models.pi05.deploy.server \
  --artifact "${ARTIFACT_DIR}" \
  --backend torch \
  --host "${POLICY_HOST}" \
  --port "${POLICY_PORT}" \
  --gpu "${GPU}" \
  --device cuda \
  --seed "${SEED}" \
  "${seed_args[@]}" \
  "${warmup_args[@]}" \
  "$@"
