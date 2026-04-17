#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

RAW_ARGS=("$@")
EXTRA_ARGS=()

NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"

# Convert short aliases to real Hydra overrides
for ((i = 0; i < ${#RAW_ARGS[@]}; i++)); do
  arg="${RAW_ARGS[$i]}"

  case "${arg}" in
    checkpoint_path=*)
      EXTRA_ARGS+=("model.checkpoint_path=${arg#checkpoint_path=}")
      ;;
    batch_size=*)
      EXTRA_ARGS+=("batch_size=${arg#batch_size=}")
      ;;
    num_workers=*)
      EXTRA_ARGS+=("num_workers=${arg#num_workers=}")
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      ;;
  esac

  # Keep your original task name parsing
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#RAW_ARGS[@]})); then
        next="${RAW_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta
import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"

export WANDB_API_KEY="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"

LAUNCH_ARGS=(
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml
  --num_processes "${NPROC_PER_NODE}"
)

if (( NUM_MACHINES > 1 )); then
  LAUNCH_ARGS+=(
    --num_machines "${NUM_MACHINES}"
    --machine_rank "${MACHINE_RANK}"
    --main_process_ip "${MAIN_PROCESS_IP}"
    --main_process_port "${MAIN_PROCESS_PORT}"
  )
fi

TRAIN_ARGS=(
  "wandb.enabled=true"
  "wandb.project=causalwam"
  "wandb.name=${TASK_BASENAME}"
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}"
)

accelerate launch \
  "${LAUNCH_ARGS[@]}" \
  scripts/train.py \
  "${TRAIN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"