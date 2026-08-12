#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?Usage: bash experiments/vlabench/run_vlabench_parallel.sh <checkpoint.pt> <dataset_stats.json> [output_dir]}"
STATS="${2:?Usage: bash experiments/vlabench/run_vlabench_parallel.sh <checkpoint.pt> <dataset_stats.json> [output_dir]}"
OUT_ROOT="${3:-evaluate_results/vlabench/fastwam_$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${NUM_GPUS:-8}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-1}"
N_EPISODE="${N_EPISODE:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-4}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
METRICS="${METRICS:-success_rate,intention_score,progress_score}"

tasks=(
  select_painting
  select_book
  select_drink
  select_chemistry_tube
  select_poker
  select_mahjong
  select_toy
  select_fruit
  add_condiment
  insert_flower
)

mkdir -p "${OUT_ROOT}/logs"
pids=()
job_idx=0

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
}
trap cleanup INT TERM

for task in "${tasks[@]}"; do
  gpu=$((job_idx % NUM_GPUS))
  log="${OUT_ROOT}/logs/${task}_gpu${gpu}.log"
  echo "[launch] task=${task} gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
    python experiments/vlabench/run_vlabench_manager.py \
      ckpt="${CKPT}" \
      EVALUATION.dataset_stats_path="${STATS}" \
      EVALUATION.task_name="${task}" \
      EVALUATION.output_dir="${OUT_ROOT}/${task}" \
      EVALUATION.n_episode="${N_EPISODE}" \
      EVALUATION.replan_steps="${REPLAN_STEPS}" \
      EVALUATION.save_video="${SAVE_VIDEO}" \
      EVALUATION.metrics="[${METRICS}]" \
      >"${log}" 2>&1 &
  pids+=("$!")
  job_idx=$((job_idx + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$((NUM_GPUS * MAX_PROCS_PER_GPU))" ]; do
    sleep 2
    wait -n || true
  done
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
exit "${failed}"
