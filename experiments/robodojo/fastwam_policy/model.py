import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent


def _find_fastwam_root() -> Path:
    raw = os.environ.get("FASTWAM_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    for candidate in [POLICY_DIR, *POLICY_DIR.parents]:
        if (candidate / "src" / "fastwam").exists():
            return candidate.resolve()
    raise FileNotFoundError("Cannot find FastWAM root. Set FASTWAM_ROOT explicitly.")


FASTWAM_ROOT = _find_fastwam_root()
FASTWAM_SRC = FASTWAM_ROOT / "src"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _standardize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (240, 320):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    return image


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return text if text else fallback


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg["action_type"]
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.action_horizon = 1
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 24)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.model = None

        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get("ckpt_setting")
        dataset_stats_path = self.model_cfg.get("dataset_stats_path")

        if self.allow_dummy_policy:
            print("[FastWAM] allow_dummy_policy=true; checkpoint loading is skipped.")
            return

        if _is_none_like(checkpoint_path):
            raise FileNotFoundError("FastWAM requires checkpoint_path/ckpt_setting for RoboDojo evaluation.")
        if _is_none_like(dataset_stats_path):
            raise FileNotFoundError("FastWAM requires dataset_stats_path for RoboDojo evaluation.")

        for path in (str(FASTWAM_ROOT), str(FASTWAM_SRC)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from experiments.robotwin.fastwam_policy.deploy_policy import get_model

        upstream_cfg = dict(self.model_cfg)
        upstream_cfg["ckpt_setting"] = str(Path(checkpoint_path).expanduser().resolve())
        upstream_cfg["dataset_stats_path"] = str(Path(dataset_stats_path).expanduser().resolve())
        upstream_cfg.setdefault("sim_cfg_name", "sim_robotwin.yaml")
        upstream_cfg.setdefault("sim_task", "robotwin_uncond_3cam_384_1e-4")
        self.model = get_model(upstream_cfg)
        self.action_horizon = int(self.model.action_horizon)
        self.replan_steps = int(self.model.replan_steps)

    def _encode_obs_for_fastwam(self, obs: dict) -> dict:
        vision = obs["vision"]
        return {
            "observation": {
                "head_camera": {"rgb": _standardize_rgb(vision["cam_head"]["color"])},
                "left_camera": {"rgb": _standardize_rgb(vision["cam_left_wrist"]["color"])},
                "right_camera": {"rgb": _standardize_rgb(vision["cam_right_wrist"]["color"])},
            },
            "joint_action": {
                "vector": pack_robot_state(
                    obs,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                    state_type="state",
                ).astype(np.float32)
            },
        }

    def update_obs(self, obs):
        self.last_obs = self._encode_obs_for_fastwam(obs)
        self.last_instruction = _get_instruction(obs, self.default_instruction)

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._batch_obs = {}
        self._batch_instruction = {}
        for obs in obs_list:
            env_idx = int(obs["env_idx"])
            self._batch_obs[env_idx] = self._encode_obs_for_fastwam(obs)
            self._batch_instruction[env_idx] = _get_instruction(obs, self.default_instruction)

    def _zero_actions(self):
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        zeros = np.zeros((self.replan_steps, dim), dtype=np.float32)
        return unpack_robot_state(zeros, self.action_type, self.robot_action_dim_info, source_type="obs")

    def _infer_actions(self, obs, instruction):
        if self.allow_dummy_policy:
            return self._zero_actions()
        if obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        action_chunk = np.asarray(self.model._infer_action_chunk(obs, instruction), dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        action_chunk = action_chunk[: min(self.replan_steps, action_chunk.shape[0])]
        return unpack_robot_state(action_chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        return self._infer_actions(self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list):
        if not hasattr(self, "_batch_obs"):
            raise ValueError("No batch observation is available. Call update_obs_batch() first.")
        return [
            self._infer_actions(self._batch_obs[int(env_idx)], self._batch_instruction[int(env_idx)])
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        if self.model is not None:
            self.model.reset()
