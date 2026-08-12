#bin/bash

export MUJOCO_GL=egl
NUM_GPUS=$(nvidia-smi -L | wc -l)
NUM_TRIALS=50
MAX_PROCS_PER_GPU=2 # process number of per gpu
MAX_PROCS=$((NUM_GPUS * MAX_PROCS_PER_GPU))
EPISODE=50
POLICY="openvla"
LORA_CKPT= # TODO
METRICS="success_rate intention_score progress_score"

tasks_to_eval="select_painting select_book select_drink select_chemistry_tube select_poker select_mahjong select_toy select_fruit add_condiment insert_flower" 
eval_tracks=("track_1_in_distribution" "track_2_cross_category" "track_3_common_sense" "track_4_semantic_instruction" "track_6_unseen_texture")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)
            TRACK_OPT="$2"
            shift 2
            ;;
        --task)
            TASK_OPT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

for TRACK in "${TRACKS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        GPU_ID=$((job_idx % NUM_GPUS))
        NOTE="${CKPT_BASENAME}"

        echo "[INFO] Submit JOB: ckpt=$CKPT, track=$TRACK, task=$TASK, gpu=$GPU_ID"

        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_EGL_DEVICE_ID=$GPU_ID \
            python scripts/evaluate_policy.py \
                --tasks $TASK \
                --eval-track $TRACK \
                --n-episode $EPISODE \
                --policy $POLICY \
                --lora_ckpt $LORA_CKPT\
                --visulization \
                --metrics $METRICS &

        job_idx=$((job_idx+1))
        while [ $(jobs -rp | wc -l) -ge $MAX_PROCS ]; do
            sleep 2
            wait -n
        done
    done
done

for eval_track in "${eval_tracks[@]}"; do
    echo "Evaluating on eval track: $eval_track"
    python scripts/evaluate_policy.py \
        --tasks $tasks_to_eval \
        --eval-track $eval_track \
        --n-episode $EPISODE \
        --policy $POLICY \
        --lora_ckpt $LORA_CKPT\
        --visulization \
        --metrics $METRICS \

done