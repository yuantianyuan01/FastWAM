"""FastWAM policy adapter for MolmoSpaces JSON benchmark evaluation.

The adapter follows MolmoSpaces' external-repo contract:

1. a policy class extending ``BasePolicy``;
2. a policy config extending ``BasePolicyConfig``;
3. an eval config extending ``JsonBenchmarkEvalConfig``.

It reuses the FastWAM RobotWin deployment helper to load the model, normalize
state, build the multi-camera input image, and denormalize action chunks.
"""

from __future__ import annotations

import collections
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(os.environ.get("FASTWAM_ROOT", Path(__file__).resolve().parents[2])).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, PROJECT_ROOT / "third_party" / "MolmoSpaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.fastwam_policy.deploy_policy import get_model
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.policy.base_policy import BasePolicy


def _first_existing(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of the requested keys are present: {keys}")


def _as_rgb(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class FastWAMMolmoSpacesPolicy(BasePolicy):
    """MolmoSpaces policy returning 8-D relative joint actions.

    Output action contract:
    ``{"arm": action[:7], "gripper": action[7:8]}``.
    """

    def __init__(self, config, task=None) -> None:
        super().__init__(config, task)
        policy_cfg = config.policy_config
        self.camera_names = list(policy_cfg.camera_names)
        self.replan_steps = int(policy_cfg.replan_steps)
        self.pending_actions: collections.deque[np.ndarray] = collections.deque()
        self.model = get_model(
            {
                "ckpt_setting": str(policy_cfg.checkpoint_path),
                "dataset_stats_path": str(policy_cfg.dataset_stats_path),
                "sim_cfg_name": str(policy_cfg.sim_cfg_name),
                "sim_task": str(policy_cfg.sim_task),
                "action_horizon": int(policy_cfg.action_horizon),
                "replan_steps": int(policy_cfg.replan_steps),
                "num_inference_steps": int(policy_cfg.num_inference_steps),
                "device": str(policy_cfg.device),
                "mixed_precision": str(policy_cfg.mixed_precision),
            }
        )

    def reset(self):
        self.pending_actions.clear()
        if hasattr(self.model, "reset"):
            self.model.reset()

    def _obs_to_fastwam(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs = observation[0] if isinstance(observation, list) else observation
        robot_state = obs.get("robot_state", {})
        qpos = np.asarray(_first_existing(robot_state, ["qpos", "joint_pos", "state"]), dtype=np.float32)
        if qpos.shape[0] < 8:
            raise ValueError(f"MolmoSpaces FastWAM adapter expects at least 8 robot-state dims, got {qpos.shape}")
        qpos = qpos[:8]

        images = {name: _as_rgb(obs[name]) for name in self.camera_names if name in obs}
        exterior = _first_existing(images, [self.camera_names[0]])
        second = _first_existing(images, [self.camera_names[1] if len(self.camera_names) > 1 else self.camera_names[0]])
        wrist = _first_existing(images, [self.camera_names[2] if len(self.camera_names) > 2 else self.camera_names[-1]])

        return {
            "observation": {
                "head_camera": {"rgb": exterior},
                "left_camera": {"rgb": second},
                "right_camera": {"rgb": wrist},
            },
            "joint_action": {"vector": qpos},
        }

    def _instruction(self, observation: dict[str, Any]) -> str:
        obs = observation[0] if isinstance(observation, list) else observation
        for key in ("instruction", "language_instruction", "task_instruction", "prompt"):
            value = obs.get(key)
            if value:
                return str(value)
        return "perform the manipulation task"

    def _fill_action_queue(self, observation: dict[str, Any]) -> None:
        fastwam_obs = self._obs_to_fastwam(observation)
        action_chunk = np.asarray(self.model._infer_action_chunk(fastwam_obs, self._instruction(observation)))
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        for action in action_chunk[: self.replan_steps]:
            self.pending_actions.append(np.asarray(action[:8], dtype=np.float32))

    def get_action(self, observation):
        if not self.pending_actions:
            self._fill_action_queue(observation)
        action = self.pending_actions.popleft()
        return {
            "arm": action[:7],
            "gripper": action[7:8],
        }


class FastWAMMolmoSpacesPolicyConfig(BasePolicyConfig):
    policy_type: str = "learned"
    policy_cls: type = None
    policy_factory: object | None = None

    checkpoint_path: str = "checkpoints/molmospaces/checkpoint.pt"
    dataset_stats_path: str = "data/molmospaces/dataset_stats.json"
    sim_cfg_name: str = "sim_molmospaces.yaml"
    sim_task: str = "molmospaces_uncond_3cam_384_1e-4"
    camera_names: list[str] = [
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left",
    ]
    action_horizon: int = 10
    replan_steps: int = 10
    num_inference_steps: int = 10
    mixed_precision: str = "bf16"
    device: str = "cuda"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        exterior = os.environ.get("FASTWAM_MOLMOSPACES_EXTERIOR_CAMERA")
        wrist = os.environ.get("FASTWAM_MOLMOSPACES_WRIST_CAMERA")
        if exterior and wrist:
            self.camera_names = [exterior, exterior, wrist]
        stats = os.environ.get("FASTWAM_MOLMOSPACES_DATASET_STATS_PATH")
        if stats:
            self.dataset_stats_path = stats
        if os.environ.get("FASTWAM_MOLMOSPACES_ACTION_HORIZON"):
            self.action_horizon = int(os.environ["FASTWAM_MOLMOSPACES_ACTION_HORIZON"])
        if os.environ.get("FASTWAM_MOLMOSPACES_REPLAN_STEPS"):
            self.replan_steps = int(os.environ["FASTWAM_MOLMOSPACES_REPLAN_STEPS"])
        if os.environ.get("FASTWAM_MOLMOSPACES_NUM_INFERENCE_STEPS"):
            self.num_inference_steps = int(os.environ["FASTWAM_MOLMOSPACES_NUM_INFERENCE_STEPS"])
        if self.policy_cls is None:
            self.policy_cls = FastWAMMolmoSpacesPolicy
            self.policy_factory = FastWAMMolmoSpacesPolicy


class FastWAMMolmoSpacesEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: FastWAMMolmoSpacesPolicyConfig = FastWAMMolmoSpacesPolicyConfig()
    policy_dt_ms: float = 66.0
    end_on_success: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)
