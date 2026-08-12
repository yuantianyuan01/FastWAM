#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
policy_server_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-ws}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

bash "${root_dir}/scripts/eval_policy.sh" \
    --bench_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --host "${policy_server_ip}" \
    --port "${policy_server_port}" \
    --protocol "${protocol}" \
    --eval_batch "${eval_batch}" \
    --root_dir "${root_dir}" \
    --device_id "${env_gpu_id}" \
    --additional_info "${additional_info}" \
    --seed "${seed}"