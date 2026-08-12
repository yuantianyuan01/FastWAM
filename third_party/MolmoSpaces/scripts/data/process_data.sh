#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_root_or_split_dir>"
    exit 1
fi

INPUT_PATH=$1

process_split() {
    local SPLIT_DIR=$1
    python scripts/data/repair_video_paths.py "$SPLIT_DIR"
    python scripts/data/validate_trajectories.py "$SPLIT_DIR" --check-visibility droid_shoulder_light_randomization pickup_obj --check-visibility droid_shoulder_light_randomization place_receptacle
    python scripts/data/calculate_stats.py "$SPLIT_DIR" --keys actions obs/agent/qpos
}

# Check if input path contains split directories (train, test, val)
SPLITS_FOUND=false
for split in train test val; do
    if [ -d "$INPUT_PATH/$split" ]; then
        SPLITS_FOUND=true
        break
    fi
done

if [ "$SPLITS_FOUND" = true ]; then
    # Input is a data root, process each existing split
    for split in train test val; do
        if [ -d "$INPUT_PATH/$split" ]; then
            echo "Processing split: $split"
            process_split "$INPUT_PATH/$split"
        fi
    done
else
    # Input is a split directory, process it directly
    process_split "$INPUT_PATH"
fi
