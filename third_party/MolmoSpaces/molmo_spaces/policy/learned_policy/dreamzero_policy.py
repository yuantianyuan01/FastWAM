import logging
import os
import time
import uuid

import cv2
import msgpack_numpy
import numpy as np
import websockets.exceptions
import websockets.sync.client

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class DreamZeroWebsocketClient:
    """Websocket client that adds endpoint field for DreamZero server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()
        # store the URI that actually worked so reconnects reuse it
        self._connected_uri = self._uri

    def _connect_once(self, uri: str) -> tuple[websockets.sync.client.ClientConnection, dict]:
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn, metadata = self._connect_once(self._uri)
            return conn, metadata
        except Exception:
            logging.info("Connection with ws:// failed. Trying wss:// ...")

        wss_uri = "wss://" + self._uri.split("//")[1]
        conn, metadata = self._connect_once(wss_uri)
        self._uri = wss_uri
        return conn, metadata

    def _reconnect(self) -> None:
        retry_delay = 2
        while True:
            logging.warning(
                f"WebSocket connection closed. Reconnecting to {self._connected_uri}..."
            )
            try:
                self._ws, self._server_metadata = self._connect_once(self._connected_uri)
                logging.info("Reconnected to server.")
                return
            except Exception as e:
                logging.warning(f"Reconnect failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    def infer(self, obs: dict) -> dict:
        obs["endpoint"] = "infer"
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            logging.warning("ConnectionClosedError during infer. Reconnecting and retrying...")
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: dict = None) -> None:
        if reset_info is None:
            reset_info = {}
        reset_info["endpoint"] = "reset"
        data = self._packer.pack(reset_info)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            logging.warning("ConnectionClosedError during reset. Reconnecting and retrying...")
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        return response


class DreamZero_Policy(InferencePolicy):
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
        self.model = None
        self.session_id = None

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None
        self.session_id = str(uuid.uuid4())

    def prepare_model(self):
        self.model_name = (
            os.path.basename(self.checkpoint_path) if self.checkpoint_path else "dreamzero"
        )
        if self.remote_config is not None:
            self._prepare_remote_model()
        else:
            raise NotImplementedError("DreamZero policy only supports remote model inference")

    def _prepare_remote_model(self):
        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 6000)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = DreamZeroWebsocketClient(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to DreamZero model at {host}:{port}")
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
        prompt = self.task.get_task_description()
        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        if grip < 0.1:
            grip = 0.00
        exo_camera_key = (
            "droid_shoulder_light_randomization"
            if "droid_shoulder_light_randomization" in obs
            else "exo_camera_1"
        )
        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )

        model_input = {
            "observation/exterior_image_0_left": resize_with_pad(obs[exo_camera_key], 180, 320),
            "observation/exterior_image_1_left": resize_with_pad(obs[exo_camera_key], 180, 320),
            "observation/wrist_image_left": resize_with_pad(obs[wrist_camera_key], 180, 320),
            "observation/joint_position": np.array(
                obs["qpos"]["arm"][:7], dtype=np.float64
            ).reshape(
                7,
            ),
            "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
            "observation/gripper_position": np.array(grip, dtype=np.float64).reshape(
                1,
            ),
            "prompt": prompt,
            "session_id": self.session_id,
        }
        return model_input

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            result = self.model.infer(model_input)
            self.actions_buffer = result["actions"]
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        gripper_pos = np.clip(model_output[7], 0.0, 1.0)

        if self.grasping_type == "binary":
            gripper_pos = (
                np.array([255.0]) if gripper_pos >= self.grasping_threshold else np.array([0.0])
            )
        elif self.grasping_type == "semi_binary":
            gripper_pos = (
                gripper_pos * np.array([255.0])
                if gripper_pos <= self.grasping_threshold
                else np.array([255.0])
            )
        elif self.grasping_type == "continuous":
            gripper_pos = gripper_pos * np.array([255.0])
        else:
            raise ValueError(f"Invalid grasping type: {self.grasping_type}")

        arm_output = model_output[:7].reshape(
            7,
        )
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "dreamzero"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        info["session_id"] = self.session_id
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
