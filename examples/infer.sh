#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INVOCATION_DIR="$(pwd)"

MODEL=""
ARTIFACT=""
INPUT=""
OUTPUT="actions.npz"
CONFIG=""
GPU="0"
DEVICE="cuda"
PYTHON_BIN="${PYTHON_BIN:-python}"
SERVE=0
HOST=""
PORT=""
DRY_RUN=0
EXTRA_OVERRIDES=()

usage() {
  printf '%s\n' \
    "Usage:" \
    "  examples/infer.sh --model MODEL --artifact PATH --input observation.npz [options]" \
    "  examples/infer.sh --model MODEL --artifact PATH --serve [options]" \
    "" \
    "Models:" \
    "  pi05, pi0" \
    "" \
    "Options:" \
    "  --model MODEL            Model family (required)" \
    "  --artifact PATH          Nexus run, latest pointer, or Artifact directory" \
    "  --input PATH             Observation .npz/.json for offline inference" \
    "  --output PATH            Offline actions output; default: actions.npz" \
    "  --serve                  Start the standard Nexus HTTP policy server" \
    "  --host HOST              Optional server.host override" \
    "  --port PORT              Optional server.port override" \
    "  --config PATH            Override the model's default inference YAML" \
    "  --gpu INDEX              Physical GPU; default: 0" \
    "  --device DEVICE          Runtime device; default: cuda" \
    "  --python PATH            Python executable; default: \${PYTHON_BIN:-python}" \
    "  --dry-run                Print the resolved command" \
    "  -- OVERRIDES...          Additional config overrides"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --artifact) ARTIFACT="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --serve) SERVE=1; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_OVERRIDES=("$@"); break ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${MODEL}" in
  pi05|pi0) ;;
  *) printf 'Model must be one of: pi05, pi0.\n' >&2; usage >&2; exit 2 ;;
esac
if [[ -z "${ARTIFACT}" ]]; then
  usage >&2
  exit 2
fi
if [[ "${SERVE}" == "0" && -z "${INPUT}" ]]; then
  printf 'Offline inference requires --input; otherwise select --serve.\n' >&2
  exit 2
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  printf 'GPU must be a non-negative index: %s\n' "${GPU}" >&2
  exit 2
fi
if [[ -n "${PORT}" && ! "${PORT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Port must be a positive integer: %s\n' "${PORT}" >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python executable was not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi

absolute_from_invocation() {
  if [[ "$1" == /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "${INVOCATION_DIR}" "$1"
  fi
}

ARTIFACT="$(absolute_from_invocation "${ARTIFACT}")"
if [[ -z "${CONFIG}" ]]; then
  CONFIG="${REPO_ROOT}/examples/configs/infer/${MODEL}.yaml"
elif [[ "${CONFIG}" != /* ]]; then
  if [[ -f "${INVOCATION_DIR}/${CONFIG}" ]]; then
    CONFIG="${INVOCATION_DIR}/${CONFIG}"
  else
    CONFIG="${REPO_ROOT}/${CONFIG}"
  fi
fi
if [[ "${SERVE}" == "0" ]]; then
  INPUT="$(absolute_from_invocation "${INPUT}")"
  OUTPUT="$(absolute_from_invocation "${OUTPUT}")"
fi

if [[ ! -e "${ARTIFACT}" ]]; then
  printf 'Artifact path was not found: %s\n' "${ARTIFACT}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  printf 'Inference config was not found: %s\n' "${CONFIG}" >&2
  exit 1
fi
if [[ "${SERVE}" == "0" && ! -f "${INPUT}" ]]; then
  printf 'Observation input was not found: %s\n' "${INPUT}" >&2
  exit 1
fi

OVERRIDES=(
  "model.artifact=${ARTIFACT}"
  "runtime.device=${DEVICE}"
)
if [[ -n "${HOST}" ]]; then
  OVERRIDES+=("server.host=${HOST}")
fi
if [[ -n "${PORT}" ]]; then
  OVERRIDES+=("server.port=${PORT}")
fi
OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

if [[ "${SERVE}" == "1" ]]; then
  COMMAND=(
    "${PYTHON_BIN}" -m dobot_nexus.cli.main serve
    --config "${CONFIG}"
    "${OVERRIDES[@]}"
  )
else
  COMMAND=(
    "${PYTHON_BIN}" -m dobot_nexus.cli.main infer
    --config "${CONFIG}"
    --input "${INPUT}"
    --output "${OUTPUT}"
    "${OVERRIDES[@]}"
  )
fi

printf '%s inference: artifact=%s gpu=%s serve=%s\n' \
  "${MODEL}" "${ARTIFACT}" "${GPU}" "${SERVE}" >&2

LAUNCH_PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s PYTHONPATH=%q ' "${GPU}" "${LAUNCH_PYTHONPATH}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ "${SERVE}" == "0" ]]; then
  mkdir -p "$(dirname -- "${OUTPUT}")"
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${LAUNCH_PYTHONPATH}"
cd "${REPO_ROOT}"
exec "${COMMAND[@]}"
