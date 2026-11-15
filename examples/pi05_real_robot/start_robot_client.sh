#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
BRIDGE_MODE="${BRIDGE_MODE:-standard}"
case "${BRIDGE_MODE}" in
  standard) default_bridge="${SCRIPT_DIR}/_runtime/dobot_dual_arm_14d.py"; history_args=() ;;
  memory) default_bridge="${SCRIPT_DIR}/_runtime/memory.py"
    history_args=(--history-fps "${HISTORY_FPS:-30}") ;;
  *) echo "BRIDGE_MODE must be standard or memory." >&2; exit 2 ;;
esac
BRIDGE_SCRIPT="${BRIDGE_SCRIPT:-${default_bridge}}"
CLIENT_PYTHON="${CLIENT_PYTHON:-python}"

ROBOT_HOST="${ROBOT_HOST:-10.40.1.26}"
ROBOT_PORT="${ROBOT_PORT:-12006}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
ACTIONS_PER_RESPONSE="${ACTIONS_PER_RESPONSE:-8}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-20}"
TASK_OVERRIDE="${TASK_OVERRIDE:-}"

if [[ ! -f "${BRIDGE_SCRIPT}" ]]; then
  echo "BRIDGE_SCRIPT was not found: ${BRIDGE_SCRIPT}" >&2
  exit 1
fi
if ! command -v "${CLIENT_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${CLIENT_PYTHON}" ]]; then
  echo "CLIENT_PYTHON was not found or is not executable: ${CLIENT_PYTHON}" >&2
  exit 1
fi

proxy_bypass="${ROBOT_HOST},${POLICY_HOST},127.0.0.1,localhost"
export no_proxy="${proxy_bypass}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${proxy_bypass}${NO_PROXY:+,${NO_PROXY}}"

command_args=(
  "${CLIENT_PYTHON}" "${BRIDGE_SCRIPT}" openpi-client
  --robot-host "${ROBOT_HOST}"
  --robot-port "${ROBOT_PORT}"
  --policy-host "${POLICY_HOST}"
  --policy-port "${POLICY_PORT}"
  --actions-per-response "${ACTIONS_PER_RESPONSE}"
  --connect-timeout "${CONNECT_TIMEOUT}"
  "${history_args[@]}"
)
if [[ -n "${TASK_OVERRIDE}" ]]; then
  command_args+=(--task "${TASK_OVERRIDE}")
fi
if [[ -n "${STATE_DIM:-}" ]]; then
  command_args+=(--state-dim "${STATE_DIM}")
fi
if [[ -n "${ACTION_DIM:-}" ]]; then
  command_args+=(--action-dim "${ACTION_DIM}")
fi

echo "Starting robot ↔ policy bridge" >&2
echo "  robot: tcp://${ROBOT_HOST}:${ROBOT_PORT}" >&2
echo "  policy: ws://${POLICY_HOST}:${POLICY_PORT}" >&2
echo "  bridge: ${BRIDGE_MODE} (${BRIDGE_SCRIPT})" >&2

exec "${command_args[@]}" "$@"
