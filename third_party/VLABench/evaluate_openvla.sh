#bin/bash

tasks_to_eval="select_fruit select_toy select_poker select_painting select_mahjong" 
eval_tracks=("track_1_in_distribution" "track_2_cross_category" "track_3_common_sense" "track_4_semantic_instruction" "track_6_unseen_texture")

EPISODE=50
POLICY="openvla"
LORA_CKPT="/remote-home1/sdzhang/project/openvla/weights/openvla-7b+vlabench_primitive+b16+lr-0.0005+lora-r32+dropout-0.0"

METRICS="success_rate intention_score progress_score"

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