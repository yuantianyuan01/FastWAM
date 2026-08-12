import zmq
import torch
import collections
import cv2

from io import BytesIO
from typing import Any, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel
import numpy as np

from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""

class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError

class BaseInferenceClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class RobotInferenceClient(BaseInferenceClient, BasePolicy):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)


class Gr00tPolicy:
    def __init__(self, host: str = "localhost", port: int = 5555, replan_steps: int = 4) -> None:
        self.policy = RobotInferenceClient(host=host, port=port)
        self.name = "Gr00t"
        self.control_mode = "ee"
        self.timestep = 0
        self.replan_steps = replan_steps
        self.action_eep = collections.deque(maxlen=self.replan_steps)
        self.action_eer = collections.deque(maxlen=self.replan_steps)
        self.action_gripper = 1.   # seems like sth wrong with gripper state unnormalize, we use a single number for one action chunk

    def predict(self, observation, **kwargs):
        if self.timestep%4 == 0:
            right, left, front, image_wrist = observation["rgb"]
            state = observation["ee_state"]
            pos, quat, gripper_state = state[:3], state[3:7], state[-1]
            ee_euler = quaternion_to_euler(quat)
            pos -= np.array([0, -0.4, 0.78])
            obs = {
                    "video.left_view": np.expand_dims(cv2.resize(left, (224, 224)), axis=0),
                    "video.right_view" : np.expand_dims(cv2.resize(right, (224, 224)), axis=0),
                    "video.wrist_view" : np.expand_dims(cv2.resize(image_wrist, (224, 224)), axis=0),
                    "state.end_effector_position_relative": np.expand_dims(pos.squeeze(), axis=0),
                    "state.end_effector_rotation_relative": np.expand_dims(ee_euler.squeeze(), axis=0),
                    "state.gripper_qpos": np.expand_dims(np.expand_dims(gripper_state, axis=0), axis=0),
                    "annotation.human.action.task_description": [observation["instruction"]],
                }
            action = self.policy.get_action(obs)
            self.action_eep.extend(action['action.end_effector_position'][:self.replan_steps])
            self.action_eer.extend(action['action.end_effector_rotation'][:self.replan_steps])
            self.action_gripper = 1. if sum(action['action.gripper_close']) >= action['action.gripper_close'].shape[0] / 2 else 0.
        target_pos = self.action_eep.popleft()
        target_euler = self.action_eer.popleft()
        gripper = self.action_gripper
        if gripper == 1.:
            gripper_state = np.ones(2)*0.04
        else:
            gripper_state = np.zeros(2)
        target_pos += np.array([0, -0.4, 0.78])
        return target_pos, target_euler, gripper_state

    def reset(self):
        self.timestep = 0
        self.action_plan = collections.deque(maxlen=4)