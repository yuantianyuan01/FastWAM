#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

if [[ -z "${FASTWAM_ROOT:-}" ]]; then
    FASTWAM_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${FASTWAM_ROOT}" || ! -d "${FASTWAM_ROOT}/src/fastwam" ]]; then
    echo "FASTWAM_ROOT must point to the FastWAM repository root." >&2
    exit 1
fi
CKPT_ROOT="${FASTWAM_CKPT_ROOT:-${SCRIPT_DIR}/checkpoints}"
ckpt_setting="${FASTWAM_CKPT_SETTING:-${ckpt_name}}"
run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"

if [[ -n "${FASTWAM_CKPT_SETTING:-}" ]]; then
    ckpt_dir="${CKPT_ROOT}/${ckpt_setting}"
elif [[ "${ckpt_name}" == /* ]]; then
    ckpt_dir="${ckpt_name}"
elif [[ "${ckpt_name}" == */* ]]; then
    ckpt_dir="${SCRIPT_DIR}/${ckpt_name}"
elif [[ -d "${CKPT_ROOT}/${run_dir_name}" ]]; then
    ckpt_dir="${CKPT_ROOT}/${run_dir_name}"
else
    ckpt_dir="${CKPT_ROOT}/${ckpt_setting}"
fi

weights_dir="${ckpt_dir}/checkpoints/weights"
dataset_stats_path="${FASTWAM_DATASET_STATS_PATH:-${ckpt_dir}/dataset_stats.json}"

allow_dummy_policy="${FASTWAM_ALLOW_DUMMY_POLICY:-false}"
checkpoint_path="${FASTWAM_CHECKPOINT_PATH:-}"
if [[ -z "${checkpoint_path}" && "${allow_dummy_policy}" != "true" ]]; then
    if [[ -d "${weights_dir}" ]]; then
        checkpoint_path=$(find "${weights_dir}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)
    fi
    if [[ -z "${checkpoint_path}" ]]; then
        checkpoint_path="${weights_dir}/step_latest.pt"
    fi
fi
if [[ -n "${checkpoint_path}" && -d "${checkpoint_path}" ]]; then
    indexed_checkpoint=$(find "${checkpoint_path}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)
    if [[ -n "${indexed_checkpoint}" ]]; then
        checkpoint_path="${indexed_checkpoint}"
    fi
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name} task=${task_name} ckpt=${ckpt_name}"
echo "[SERVER] checkpoint_path=${checkpoint_path:-<dummy-policy>}"
echo "[SERVER] dataset_stats_path=${dataset_stats_path}"
echo "[SERVER] ${policy_server_host}:${policy_server_port} gpu=${policy_gpu_id}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    FASTWAM_ROOT="${FASTWAM_ROOT}" \
    PYTHONPATH="${BENCH_ROOT}:${FASTWAM_ROOT}:${FASTWAM_ROOT}/src:${PYTHONPATH:-}" \
    DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            checkpoint_path="${checkpoint_path}" \
            dataset_stats_path="${dataset_stats_path}" \
            allow_dummy_policy="${allow_dummy_policy}"
