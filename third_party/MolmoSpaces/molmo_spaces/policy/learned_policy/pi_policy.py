import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PI_PolicyState:
    actions_buffer: list[np.ndarray] | None = None
    current_buffer_index: int = 0
    starting_time: float | None = None


class PI_Policy(InferencePolicy, StatefulPolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = exp_config.policy_config.grasping_type
        self.chunk_size = exp_config.policy_config.chunk_size
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.model = None  # don't init model till inference to allow multiprocessing

    def get_state(self):
        return PI_PolicyState(
            actions_buffer=self.actions_buffer,
            current_buffer_index=self.current_buffer_index,
            starting_time=self.starting_time,
        )

    def set_state(self, state: PI_PolicyState):
        self.actions_buffer = state.actions_buffer
        self.current_buffer_index = state.current_buffer_index
        self.starting_time = state.starting_time

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None

    def prepare_model(self):
        self.model_name = os.path.basename(self.checkpoint_path)
        if self.remote_config is not None:
            self._prepare_remote_model(self.checkpoint_path)
        else:
            self._prepare_local_model(self.checkpoint_path)
        # self.reset()

    def _prepare_local_model(self, checkpoint_path: str):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        self.config = _config.get_config(os.path.basename(checkpoint_path.rstrip("/")))
        self.model = _policy_config.create_trained_policy(self.config, checkpoint_path)

    def _prepare_remote_model(self, checkpoint_path: str):
        try:
            from openpi_client import websocket_client_policy
        except ImportError as e:
            log.warning(
                "openpi_client package is required for remote model inference. "
                "Install it with: pip install openpi-client"
            )
            raise e

        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 8000)
        self.checkpoint_path = checkpoint_path

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = websocket_client_policy.WebsocketClientPolicy(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to remote model at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to remote model after {max_retries} attempts")
                    raise

    def render(self, obs):
        views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        # self.render(obs)
        if isinstance(obs, list):
            if len(obs) > 1:
                log.warning(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "WARNING: obs list has %d elements but only using the first one!\n"
                    "This may indicate a batching issue - expected single observation.\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    len(obs),
                )
            obs = obs[0]
        model_input = {**obs}
        prompt = self.task.get_task_description()

        # For local eval
        if isinstance(obs, list | tuple):
            obs = obs[0]

        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        exo_camera_key = (
            "droid_shoulder_light_randomization"
            if "droid_shoulder_light_randomization" in obs
            else "exo_camera_1"
        )
        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )
        model_input = {
            "observation/exterior_image_1_left": resize_with_pad(obs[exo_camera_key], 224, 224),
            "observation/wrist_image_left": resize_with_pad(obs[wrist_camera_key], 224, 224),
            "observation/joint_position": np.array(obs["qpos"]["arm"][:7]).reshape(
                7,
            ),
            "observation/gripper_position": np.array(grip).reshape(
                1,
            ),
            "prompt": prompt.lower(),
        }
        return model_input

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            import websockets

            try:
                self.actions_buffer = self.model.infer(model_input)["actions"]
            except websockets.exceptions.ConnectionClosedError:
                log.error("Connection closed error. Attempting to reset connection...")
                self.prepare_model()
                log.info("Sleeping 5s...")
                time.sleep(5)
                log.info("Retrying inference...")
                self.actions_buffer = self.model.infer(model_input)["actions"]
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        if self.grasping_type == "continuous":
            gripper_pos = model_output[7] * np.array([255.0])
        else:  # binary
            gripper_pos = (
                np.array([255.0]) if model_output[7] > self.grasping_threshold else np.array([0.0])
            )

        arm_output = model_output[:7].reshape(
            7,
        )
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_action_chunk(self, observation: Any) -> list[dict[str, np.ndarray]]:
        """Return the whole predicted chunk, so only its first action needs an observation.

        The model predicts ``chunk_size`` actions per inference. ``get_action``
        triggers that inference and consumes the first action; the rest are drained
        from the buffer here. The returned list is a chunk of single-env actions,
        not one batched action -- see ``BasePolicy.get_action_chunk``.
        """
        first_action = self.get_action(observation)
        buffered_outputs = self.actions_buffer[self.current_buffer_index : self.chunk_size]
        self.current_buffer_index = self.chunk_size
        return [first_action] + [self.model_output_to_action(out) for out in buffered_outputs]

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = (
            self.model.get_server_metadata().get("policy_name", "pi")
            if hasattr(self.model, "get_server_metadata")
            else "pi"
        )
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        log.info(f"Current prompt: {info['prompt']}")

        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
