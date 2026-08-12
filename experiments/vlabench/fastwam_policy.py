"""FastWAM policy adapter for VLABench evaluation."""

from __future__ import annotations

import collections
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(os.environ.get("FASTWAM_ROOT", Path(__file__).resolve().parents[2])).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, PROJECT_ROOT / "third_party" / "VLABench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.fastwam_policy.deploy_policy import get_model
from VLABench.evaluation.model.policy.base import Policy
from VLABench.utils.utils import quaternion_to_euler


def _as_rgb(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class FastWAMVLABenchPolicy(Policy):
    """VLABench policy returning target EEF pose from 7-D absolute EEF chunks."""

    def __init__(
        self,
        checkpoint_path: str,
        dataset_stats_path: str,
        sim_cfg_name: str = "sim_vlabench.yaml",
        sim_task: str = "vlabench_uncond_3cam_384_1e-4",
        action_horizon: int = 32,
        replan_steps: int = 4,
        num_inference_steps: int = 10,
        device: str = "cuda",
        mixed_precision: str = "bf16",
    ) -> None:
        self.replan_steps = int(replan_steps)
        self.pending_actions: collections.deque[np.ndarray] = collections.deque()
        self.model = get_model(
            {
                "ckpt_setting": checkpoint_path,
                "dataset_stats_path": dataset_stats_path,
                "sim_cfg_name": sim_cfg_name,
                "sim_task": sim_task,
                "action_horizon": action_horizon,
                "replan_steps": replan_steps,
                "num_inference_steps": num_inference_steps,
                "device": device,
                "mixed_precision": mixed_precision,
            }
        )
        super().__init__(self.model)

    @property
    def name(self):
        return "FastWAM"

    @property
    def control_mode(self):
        return "ee"

    def reset(self):
        self.pending_actions.clear()
        if hasattr(self.model, "reset"):
            self.model.reset()

    def _obs_to_fastwam(self, obs: dict[str, Any]) -> dict[str, Any]:
        right, left, front, image_wrist = obs["rgb"]
        ee_state = np.asarray(obs["ee_state"], dtype=np.float32)
        if len(ee_state) == 8:
            pos, quat, gripper = ee_state[:3], ee_state[3:7], ee_state[-1:]
            euler = quaternion_to_euler(quat)
        elif len(ee_state) == 7:
            pos, euler, gripper = ee_state[:3], ee_state[3:6], ee_state[-1:]
        else:
            raise ValueError(f"Unsupported VLABench ee_state shape: {ee_state.shape}")
        state = np.concatenate([pos, euler, gripper], axis=0).astype(np.float32)
        return {
            "observation": {
                "head_camera": {"rgb": _as_rgb(front)},
                "left_camera": {"rgb": _as_rgb(left)},
                "right_camera": {"rgb": _as_rgb(image_wrist)},
            },
            "joint_action": {"vector": state[:7]},
        }

    def _fill_action_queue(self, obs: dict[str, Any]) -> None:
        action_chunk = np.asarray(self.model._infer_action_chunk(self._obs_to_fastwam(obs), obs["instruction"]))
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        for action in action_chunk[: self.replan_steps]:
            self.pending_actions.append(np.asarray(action[:7], dtype=np.float32))

    def predict(self, obs, **kwargs):
        if not self.pending_actions:
            self._fill_action_queue(obs)
        action = self.pending_actions.popleft()
        target_pos = action[:3]
        target_euler = action[3:6]
        gripper_state = np.ones(2) * 0.04 if action[6] >= 0.1 else np.zeros(2)
        return target_pos, target_euler, gripper_state
