#!/bin/bash
set -e

ROOT_DIR="$1"
env_cfg_type="$2"

python3 -c '
import sys, os, json

root_dir = sys.argv[1]
env_cfg_type = sys.argv[2]

robot_action_dim_info = json.load(
    open(os.path.join(root_dir, "XPolicyLab", "utils", "robot", "_robot_info.json"), "r", encoding="utf-8")
)[env_cfg_type]

print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "${ROOT_DIR}" "${env_cfg_type}"