#!/usr/bin/env bash
# Start an XPolicyLab policy server only (no sim client). Used by robodojo.sh server.
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "usage: bash scripts/internal/run_policy_server.sh POLICY_DIR DATASET TASK CKPT ENV_CFG ACTION_TYPE SEED POLICY_GPU POLICY_ENV PORT [BIND_HOST]" >&2
  exit 2
fi

policy_dir="$(cd "$1" && pwd)"
shift

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host="${10:-0.0.0.0}"

ROOT_DIR="$(cd "${policy_dir}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
SERVER_SCRIPT="${policy_dir}/setup_eval_policy_server.sh"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
  echo "[run_policy_server] missing setup script: ${SERVER_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${policy_server_port}" ]]; then
  policy_server_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
fi

echo "[run_policy_server] policy_dir=${policy_dir}"
echo "[run_policy_server] task=${task_name} bind=${policy_server_host}:${policy_server_port}"
echo "[run_policy_server] remote clients: bash scripts/robodojo.sh client --policy-host <this_host_ip> --policy-port ${policy_server_port} ..."

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
  "${policy_server_port}" \
  "${policy_server_host}"
