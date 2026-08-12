#!/bin/bash

task_names=(texas_holdem 
            set_study_table
            hammer_nail_and_hang_picture
            book_rearrange
            cluster_billiards
            play_math_game
            rearrange_tube
            get_coffee)

while [[ $# -gt 0 ]]; do
  case $1 in
    --tasks)
      IFS=',' read -r -a task_names <<< "$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

PROCESS_NUM_PER_GPU=8
N_PROCESS=$((NUM_GPUS * PROCESS_NUM_PER_GPU))

# TODO: relace the parameters as you desire.
N_SAMPLE=500
MAX_EPISODE=2000
OFFSET=0
SAVE_DIR='./datasets'
export MUJOCO_GL=egl

for task_name in "${task_names[@]}"; do
    commands=()
    for ((i = 0; i < N_PROCESS; i++)); do
        device_id=$((i % 8))
        start_id=$((i * N_SAMPLE + OFFSET))
        cmd="MUJOCO_EGL_DEVICE_ID=${device_id} python scripts/trajectory_generation.py \
            --task-name $task_name \
            --n-sample $N_SAMPLE \
            --start-id $start_id \
            --save-dir $SAVE_DIR \
            --max-episode $MAX_EPISODE"
        commands+=("$cmd")
    done

    echo "Running tasks for: $task_name"

    for cmd in "${commands[@]}"; do
        echo "Starting: $cmd"
        eval "$cmd"  &
    done

    wait
    echo "Completed tasks for: $task_name"
done
echo "All processes for all tasks have completed."

