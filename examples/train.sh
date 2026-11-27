#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INVOCATION_DIR="$(pwd)"

CONFIG=""
GPUS="0"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME=""
OUTPUT_DIR=""
DRY_RUN=0

usage() {
  printf '%s\n' \
    "Usage:" \
    "  examples/train.sh --config PATH [options] [-- OVERRIDES...]" \
    "" \
    "Options:" \
    "  --config PATH            Model/task training YAML (required)" \
    "  --gpus LIST              CUDA devices; default: 0" \
    "  --python PATH            Python executable; default: \${PYTHON_BIN:-python}" \
    "  --run-name NAME          Override exp.name" \
    "  --output-dir PATH        Override exp.output_dir" \
    "  --dry-run                Print the resolved command without running it" \
    "  -- OVERRIDES...          Additional config overrides" \
    "  -h, --help               Show this help" \
    "" \
    "Examples:" \
    "  examples/train.sh --config examples/configs/train/pi05.yaml --gpus 0,1 \\" \
    "    -- data.path=/path/to/task model.checkpoint=/path/to/pi05"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'GPU list must contain comma-separated non-negative indices: %s\n' "${GPUS}" >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python executable was not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
# Fix the caller's executable before cd, without resolving virtualenv symlinks.
PYTHON_BIN="$(command -v "${PYTHON_BIN}" || printf '%s' "${PYTHON_BIN}")"
if [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${INVOCATION_DIR}/${PYTHON_BIN}"
fi

if [[ "${CONFIG}" != /* ]]; then
  if [[ -f "${INVOCATION_DIR}/${CONFIG}" ]]; then
    CONFIG="${INVOCATION_DIR}/${CONFIG}"
  else
    CONFIG="${REPO_ROOT}/${CONFIG}"
  fi
fi
if [[ ! -f "${CONFIG}" ]]; then
  printf 'Training config was not found: %s\n' "${CONFIG}" >&2
  exit 1
fi

if [[ -n "${RUN_NAME}" ]]; then
  set -- "$@" "exp.name=${RUN_NAME}"
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  set -- "$@" "exp.output_dir=${OUTPUT_DIR}"
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
LAUNCH_PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_INFO="$(
  PYTHONPATH="${LAUNCH_PYTHONPATH}" "${PYTHON_BIN}" -c '
import sys
from dobot_nexus.core.config import load_config, resolve_training_runtime
from dobot_nexus.runtime.config import validate_launcher_env

config = load_config(sys.argv[1], sys.argv[3:], expected_task="train")
try:
    runtime = config.get("runtime", {})
    if runtime.get("launcher") == "rlinf":
        validate_launcher_env(config)
    else:
        runtime = resolve_training_runtime(config, world_size=int(sys.argv[2]))
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(2)
launcher = str(runtime.get("launcher", "local"))
fsdp_enabled = bool(runtime.get("fsdp", {}).get("enabled", False))
print(f"{launcher}\t{str(fsdp_enabled).lower()}")
' "${CONFIG}" "${NPROC_PER_NODE}" "$@"
)"
IFS=$'\t' read -r RUNTIME_LAUNCHER FSDP_ENABLED <<< "${RUNTIME_INFO}"
if [[ "${RUNTIME_LAUNCHER}" == "rlinf" ]]; then
  NPROC_PER_NODE=1
  COMMAND=("${PYTHON_BIN}" -m dobot_nexus.cli.main train)
elif [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
  # This is a fresh single-process job, even when invoked from a torchrun worker.
  # Include the clean environment in COMMAND so --dry-run prints the same launch.
  COMMAND=(
    env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE
    -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT
    "${PYTHON_BIN}" -m dobot_nexus.cli.main train
  )
else
  COMMAND=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    "--nproc_per_node=${NPROC_PER_NODE}"
    -m dobot_nexus.cli.main train
  )
fi
COMMAND+=(--config "${CONFIG}" "$@")

printf 'Training: config=%s launcher=%s fsdp=%s gpus=%s processes=%s\n' \
  "${CONFIG}" "${RUNTIME_LAUNCHER}" "${FSDP_ENABLED}" "${GPUS}" "${NPROC_PER_NODE}" >&2

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s PYTHONPATH=%q ' \
    "${GPUS}" "${LAUNCH_PYTHONPATH}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${LAUNCH_PYTHONPATH}"
cd "${REPO_ROOT}"
exec "${COMMAND[@]}"
