#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_MODE="${BRIDGE_MODE:-standard}"
case "${BRIDGE_MODE}" in
  standard) default_bridge="${SCRIPT_DIR}/_runtime/dobot_dual_arm_14d.py"; history_args=() ;;
  memory) default_bridge="${SCRIPT_DIR}/_runtime/memory.py"
    history_args=(--history-fps "${HISTORY_FPS:-30}") ;;
  *) echo "BRIDGE_MODE must be standard or memory." >&2; exit 2 ;;
esac
BRIDGE_SCRIPT="${BRIDGE_SCRIPT:-${default_bridge}}"
ROBOT_PYTHON="${ROBOT_PYTHON:-python}"
ENV_SRC="${ENV_SRC:-}"

ROBOT_BIND_HOST="${ROBOT_BIND_HOST:-127.0.0.1}"
ROBOT_PORT="${ROBOT_PORT:-12006}"
TASK="${TASK:-}"
ACTIONS_PER_RESPONSE="${ACTIONS_PER_RESPONSE:-8}"
STEP_DT="${STEP_DT:-0.05}"
SETTLE_DT="${SETTLE_DT:-0.0}"
MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-0.15}"
MAX_STEPS="${MAX_STEPS:-80}"
TOP_ROI_XYWH="${TOP_ROI_XYWH:-}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-20}"

TOP_CAMERA_ID="${TOP_CAMERA_ID:-218622274004}"
LEFT_CAMERA_ID="${LEFT_CAMERA_ID:-218622272117}"
RIGHT_CAMERA_ID="${RIGHT_CAMERA_ID:-218622277219}"
LEFT_GRIPPER_PORT="${LEFT_GRIPPER_PORT:-/dev/ttyUSB0}"
RIGHT_GRIPPER_PORT="${RIGHT_GRIPPER_PORT:-/dev/ttyUSB1}"
LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.5.1}"
RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.5.2}"
LEFT_TELOP_PORT="${LEFT_TELOP_PORT:-/dev/ttyACM0}"
RIGHT_TELOP_PORT="${RIGHT_TELOP_PORT:-/dev/ttyACM1}"

# Safety defaults: collect observations and print predicted actions, but never call
# robot.step(). Physical execution must be explicitly armed with ROBOT_EXECUTE=1.
ROBOT_EXECUTE="${ROBOT_EXECUTE:-0}"
CONFIRM_EACH_ACTION="${CONFIRM_EACH_ACTION:-1}"
HOME_ON_START="${HOME_ON_START:-1}"

if [[ -z "${ENV_SRC}" || -z "${TASK//[[:space:]]/}" || -z "${TOP_ROI_XYWH}" ]]; then
  echo "Set ENV_SRC, TASK (the trained instruction), and TOP_ROI_XYWH (X Y W H, or none for the full frame)." >&2
  exit 2
fi
if [[ ! -f "${BRIDGE_SCRIPT}" ]]; then
  echo "BRIDGE_SCRIPT was not found: ${BRIDGE_SCRIPT}" >&2
  exit 1
fi
if ! command -v "${ROBOT_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${ROBOT_PYTHON}" ]]; then
  echo "ROBOT_PYTHON was not found or is not executable: ${ROBOT_PYTHON}" >&2
  exit 1
fi

if [[ "${TOP_ROI_XYWH}" == none ]]; then
  roi_args=(--no-top-crop)
else
  read -r -a top_roi <<< "${TOP_ROI_XYWH}"
  if [[ "${#top_roi[@]}" -ne 4 ]]; then
    echo "TOP_ROI_XYWH must be none or four integers: X Y W H" >&2
    exit 1
  fi
  roi_args=(--top-roi-xywh "${top_roi[@]}")
fi

command_args=(
  "${ROBOT_PYTHON}" "${BRIDGE_SCRIPT}" robot-server
  --host "${ROBOT_BIND_HOST}"
  --port "${ROBOT_PORT}"
  --env-src "${ENV_SRC}"
  --task "${TASK}"
  --actions-per-response "${ACTIONS_PER_RESPONSE}"
  --step-dt "${STEP_DT}"
  --settle-dt "${SETTLE_DT}"
  --max-joint-delta "${MAX_JOINT_DELTA}"
  --max-steps "${MAX_STEPS}"
  "${roi_args[@]}"
  --camera-fps "${CAMERA_FPS}"
  --control-frequency "${CONTROL_FREQUENCY}"
  "${history_args[@]}"
  --top-camera-id "${TOP_CAMERA_ID}"
  --left-camera-id "${LEFT_CAMERA_ID}"
  --right-camera-id "${RIGHT_CAMERA_ID}"
  --left-gripper-port "${LEFT_GRIPPER_PORT}"
  --right-gripper-port "${RIGHT_GRIPPER_PORT}"
  --left-dobot-arm-ip "${LEFT_ARM_IP}"
  --right-dobot-arm-ip "${RIGHT_ARM_IP}"
  --left-telop-arm-port "${LEFT_TELOP_PORT}"
  --right-telop-arm-port "${RIGHT_TELOP_PORT}"
)

case "${ROBOT_EXECUTE}" in
  0) echo "Starting robot server in DRY-RUN mode (no robot.step calls)." >&2 ;;
  1)
    command_args+=(--execute)
    case "${HOME_ON_START}" in
      0) ;;
      1) command_args+=(--home-on-start) ;;
      *) echo "HOME_ON_START must be 0 or 1, got ${HOME_ON_START}" >&2; exit 1 ;;
    esac
    echo "Starting robot server with PHYSICAL EXECUTION enabled." >&2
    ;;
  *) echo "ROBOT_EXECUTE must be 0 or 1, got ${ROBOT_EXECUTE}" >&2; exit 1 ;;
esac

case "${CONFIRM_EACH_ACTION}" in
  0) command_args+=(--no-confirm-each-action) ;;
  1) command_args+=(--confirm-each-action) ;;
  *) echo "CONFIRM_EACH_ACTION must be 0 or 1, got ${CONFIRM_EACH_ACTION}" >&2; exit 1 ;;
esac

exec "${command_args[@]}" "$@"
