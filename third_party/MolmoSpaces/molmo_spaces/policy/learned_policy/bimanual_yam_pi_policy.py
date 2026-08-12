"""Bimanual YAM Pi0.5 Policy using LeRobot gRPC inference.

This policy connects to a LeRobot async inference server to run Pi0.5
for bimanual manipulation with the YAM robot.
"""

import logging
import os
import time

import cv2
import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.lerobot_grpc_client import LeRobotGRPCClient

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BimanualYamPiPolicy(InferencePolicy):
    """Policy for bimanual YAM robot using Pi0.5 via LeRobot gRPC server.

    This policy:
    - Connects to a LeRobot async inference server
    - Sends observations (3 cameras + 14-dim state + task prompt)
    - Receives 14-dim action chunks (left arm, left gripper, right arm, right gripper)
    - Buffers actions to avoid calling the model every step

    Expected observation format from MuJoCo:
        - "left_wrist_camera": RGB image from left wrist
        - "right_wrist_camera": RGB image from right wrist
        - "exo_camera": RGB image from top-down exo camera
        - "qpos": {"left_arm": (6,), "right_arm": (6,), "left_gripper": (N,), "right_gripper": (N,)}

    Action format to robot:
        - "left_arm": (6,) joint positions
        - "right_arm": (6,) joint positions
        - "left_gripper": (1,) gripper command
        - "right_gripper": (1,) gripper command
    """

    # Gripper normalization constant (YAM gripper range is 0.0 to 0.041)
    GRIPPER_MAX = 0.041

    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        """Initialize the policy.

        Args:
            exp_config: Experiment configuration containing policy config
        """
        super().__init__(exp_config, exp_config.task_type)

        policy_config = exp_config.policy_config
        self.remote_config = policy_config.remote_config
        self.checkpoint_path = policy_config.checkpoint_path
        self.grasping_type = getattr(policy_config, "grasping_type", "binary")
        self.buffer_length = 50

        # Camera mapping: MuJoCo camera name -> LeRobot feature key
        self.camera_mapping = getattr(
            policy_config,
            "camera_mapping",
            {
                "left_wrist_camera": "observation.images.left",
                "right_wrist_camera": "observation.images.right",
                "exo_camera": "observation.images.top",
            },
        )

        # Will be initialized in prepare_model()
        self.client: LeRobotGRPCClient | None = None
        self.actions_buffer: list[np.ndarray] | None = None
        self.current_buffer_index: int = 0
        self.starting_time: float | None = None

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None

        # Reset client timestep counter
        if self.client is not None:
            self.client.reset()

    # State names matching the training dataset (Jiafei1224/c)
    STATE_NAMES = [
        "left_joint_0.pos",
        "left_joint_1.pos",
        "left_joint_2.pos",
        "left_joint_3.pos",
        "left_joint_4.pos",
        "left_joint_5.pos",
        "left_gripper.pos",
        "right_joint_0.pos",
        "right_joint_1.pos",
        "right_joint_2.pos",
        "right_joint_3.pos",
        "right_joint_4.pos",
        "right_joint_5.pos",
        "right_gripper.pos",
    ]

    def prepare_model(self) -> None:
        """Connect to the LeRobot gRPC server and load the policy."""
        self.model_name = os.path.basename(self.checkpoint_path)

        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 8080)

        self.client = LeRobotGRPCClient(host=host, port=port)

        policy_type = self.remote_config.get("policy_type", "pi05")
        device = self.remote_config.get("device", "cuda")

        # Define lerobot_features matching the training dataset format (Jiafei1224/c)
        # These features tell build_dataset_frame how to transform raw observations
        lerobot_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": self.STATE_NAMES,
            },
            "observation.images.left": {
                "dtype": "video",
                "shape": [360, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.right": {
                "dtype": "video",
                "shape": [360, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": [360, 640, 3],
                "names": ["height", "width", "channels"],
            },
        }

        self.client.connect(
            pretrained_name_or_path=self.checkpoint_path,
            policy_type=policy_type,
            device=device,
            actions_per_chunk=self.buffer_length,
            lerobot_features=lerobot_features,
        )

        log.info(f"BimanualYamPiPolicy connected to {host}:{port}")

    def render(self, obs: dict) -> None:
        """Display camera views for debugging.

        Args:
            obs: Observation dictionary with camera images
        """
        images = []
        for mj_cam in self.camera_mapping:
            if mj_cam in obs:
                images.append(obs[mj_cam])

        if images:
            combined = np.concatenate(images, axis=1)
            cv2.imshow("BimanualYam Views", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def _normalize_gripper(self, gripper_qpos) -> float:
        """Normalize gripper position to [0, 1] range.

        Args:
            gripper_qpos: Raw gripper joint position(s) - can be list, np.ndarray, or scalar

        Returns:
            Normalized gripper value (0 = closed, 1 = open)
        """
        # Take first joint if multiple (coupled gripper)
        if isinstance(gripper_qpos, np.ndarray):
            grip_val = gripper_qpos[0] if gripper_qpos.ndim > 0 else float(gripper_qpos)
        elif isinstance(gripper_qpos, list | tuple):
            grip_val = float(gripper_qpos[0])
        else:
            grip_val = float(gripper_qpos)

        # Normalize to [0, 1]
        normalized = np.clip(grip_val / self.GRIPPER_MAX, 0.0, 1.0)
        return float(normalized)

    def obs_to_model_input(self, obs: dict) -> dict:
        """Transform MuJoCo observations to raw format for LeRobot's build_dataset_frame.

        The server's build_dataset_frame expects:
        - Individual state values with keys matching STATE_NAMES
        - Image keys without "observation.images." prefix (e.g., "left", "right", "top")
        - "task" key for the text prompt

        Args:
            obs: MuJoCo observation dictionary with cameras and joint states

        Returns:
            Dictionary in raw format that build_dataset_frame will transform
        """
        model_input = {}

        # Process camera images - use short keys (build_dataset_frame strips "observation.images." prefix)
        camera_short_keys = {
            "left_wrist_camera": "left",
            "right_wrist_camera": "right",
            "exo_camera": "top",
        }
        for mj_cam, short_key in camera_short_keys.items():
            if mj_cam in obs:
                model_input[short_key] = obs[mj_cam]

        # Build state values as individual keys matching STATE_NAMES
        # Order: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        qpos = obs.get("qpos", {})

        left_arm = np.array(qpos.get("left_arm", np.zeros(6)))[:6]
        right_arm = np.array(qpos.get("right_arm", np.zeros(6)))[:6]

        left_grip = self._normalize_gripper(qpos.get("left_gripper", np.array([0.0])))
        right_grip = self._normalize_gripper(qpos.get("right_gripper", np.array([0.0])))

        # Combine into 14-dim array
        state_values = np.concatenate(
            [
                left_arm,  # [0:6]
                np.array([left_grip]),  # [6]
                right_arm,  # [7:13]
                np.array([right_grip]),  # [13]
            ]
        ).astype(np.float32)

        # Add individual state values with keys matching STATE_NAMES
        for i, name in enumerate(self.STATE_NAMES):
            model_input[name] = float(state_values[i])

        # Add task prompt
        prompt = self.task.get_task_description()
        model_input["task"] = prompt

        return model_input

    def inference_model(self, model_input: dict) -> np.ndarray:
        """Run inference with action buffering.

        Only calls the remote model when the buffer is empty or exhausted.

        Args:
            model_input: Preprocessed observation in LeRobot format

        Returns:
            Single action from the buffer
        """
        if self.starting_time is None:
            self.starting_time = time.time()

        # Refill buffer if needed
        if self.actions_buffer is None or self.current_buffer_index >= len(self.actions_buffer):
            # Debug: check for NaN/inf in observation before sending to server
            for key, value in model_input.items():
                if isinstance(value, np.ndarray):
                    if np.isnan(value).any():
                        log.error(f"NaN detected in observation key '{key}'")
                    if np.isinf(value).any():
                        log.error(f"Inf detected in observation key '{key}'")
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    log.error(f"NaN/Inf detected in observation key '{key}': {value}")

            self.actions_buffer = self.client.infer(model_input)
            log.info(f"Got action buffer as output from inference: {self.actions_buffer}")
            self.current_buffer_index = 0

            if not self.actions_buffer:
                log.warning("Received empty action buffer from server")
                # Return zeros as fallback
                return np.zeros(14, dtype=np.float32)

        # Get next action from buffer
        action = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1

        return action

    def _scale_gripper_action(self, gripper_value: float) -> np.ndarray:
        """Convert normalized gripper value to robot command.

        Args:
            gripper_value: Normalized gripper value from model (0-1)

        Returns:
            Gripper command array
        """
        if self.grasping_type == "continuous":
            # Scale to gripper range
            return np.array([gripper_value * self.GRIPPER_MAX])
        else:
            # Binary: threshold at 0.5
            return np.array([self.GRIPPER_MAX if gripper_value > 0.5 else 0.0])

    def model_output_to_action(self, model_output: np.ndarray) -> dict:
        """Convert model output to bimanual robot action dictionary.

        Model output format (14 dims):
            [0:6]  - left arm joint positions
            [6]    - left gripper
            [7:13] - right arm joint positions
            [13]   - right gripper

        Args:
            model_output: 14-dim action array from model

        Returns:
            Dictionary with keys: left_arm, right_arm, left_gripper, right_gripper
        """
        # Extract components from model output
        left_arm = model_output[0:6].astype(np.float32)
        left_gripper_val = float(model_output[6])
        right_arm = model_output[7:13].astype(np.float32)
        right_gripper_val = float(model_output[13])

        action = {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_gripper": self._scale_gripper_action(left_gripper_val),
            "right_gripper": self._scale_gripper_action(right_gripper_val),
        }

        log.info(
            f"[ACTION] step={self.current_buffer_index} "
            f"left_arm={np.array2string(left_arm, precision=3, separator=', ')} "
            f"left_grip={left_gripper_val:.3f} "
            f"right_arm={np.array2string(right_arm, precision=3, separator=', ')} "
            f"right_grip={right_gripper_val:.3f}"
        )

        return action

    def get_info(self) -> dict:
        """Get policy information for logging.

        Returns:
            Dictionary with policy metadata
        """
        info = super().get_info()
        info["policy_name"] = "bimanual_yam_pi"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.buffer_length
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
