#!/usr/bin/env bash
# RoboDojo eval orchestration: same contract as XPolicyLab policy eval.sh.
set -euo pipefail

if [[ $# -lt 11 ]]; then
  echo "usage: bash scripts/internal/run_policy_eval.sh POLICY_DIR DATASET TASK CKPT ENV_CFG [EXPERT_NUM] ACTION_TYPE SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV" >&2
  exit 2
fi

policy_dir="$(cd "$1" && pwd)"
shift

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
shift 4

if [[ $# -eq 6 ]]; then
  action_type=$1
  seed=$2
  policy_gpu_id=$3
  env_gpu_id=$4
  policy_conda_env=$5
  eval_env_conda_env=$6
  shift 6
elif [[ $# -eq 7 ]]; then
  _expert_num=$1
  action_type=$2
  seed=$3
  policy_gpu_id=$4
  env_gpu_id=$5
  policy_conda_env=$6
  eval_env_conda_env=$7
  shift 7
else
  echo "[run_policy_eval] unexpected trailing argument count: $#" >&2
  exit 2
fi

if [[ $# -ne 0 ]]; then
  echo "[run_policy_eval] unexpected extra arguments: $*" >&2
  exit 2
fi

ROOT_DIR="$(cd "${policy_dir}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
SERVER_SCRIPT="${policy_dir}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${policy_dir}/setup_eval_env_client.sh"

if [[ ! -f "${SERVER_SCRIPT}" || ! -f "${CLIENT_SCRIPT}" ]]; then
  echo "[run_policy_eval] missing setup scripts under ${policy_dir}" >&2
  exit 1
fi

policy_server_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

_kill_process_tree() {
  local pid=$1
  local sig=${2:-TERM}
  local child
  # Depth-first so children die before parents; covers conda/bash wrappers.
  while read -r child; do
    [[ -n "${child}" ]] || continue
    _kill_process_tree "${child}" "${sig}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "-${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
  # Prevent re-entry from nested EXIT after INT/TERM.
  trap '' EXIT INT TERM
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "[MAIN] kill server tree ${SERVER_PID}"
    _kill_process_tree "${SERVER_PID}" TERM
    local _i
    for _i in 1 2 3 4 5; do
      kill -0 "${SERVER_PID}" 2>/dev/null || {
        SERVER_PID=""
        return 0
      }
      sleep 0.2
    done
    _kill_process_tree "${SERVER_PID}" KILL
    SERVER_PID=""
  fi
}
trap cleanup EXIT INT TERM

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

(
  cd "${policy_dir}"
  exec bash setup_eval_policy_server.sh \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}"
) &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
  "${policy_server_ip}" \
  "${policy_server_port}" \
  "${SERVER_PID}" \
  "Policy server" \
  600

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
  "${bench_name}" \
  "${task_name}" \
  "${ckpt_name}" \
  "${env_cfg_type}" \
  "${action_type}" \
  "${seed}" \
  "${env_gpu_id}" \
  "${eval_env_conda_env}" \
  "${additional_info}" \
  "${policy_server_port}" \
  "${policy_server_ip}"

echo "[MAIN] eval finished"
