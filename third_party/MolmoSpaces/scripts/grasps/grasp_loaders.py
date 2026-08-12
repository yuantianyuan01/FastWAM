"""Abstraction for loading grasps from various sources (files, models, etc.)."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import logging

log = logging.getLogger(__name__)


class GraspLoader(ABC):
    """Abstract base class for loading grasps from various sources."""

    @abstractmethod
    def load_grasps_for_object(
        self,
        object_name: str,
        num_grasps: int = 1000
    ) -> Tuple[str, np.ndarray]:
        """Load grasps for a static/pickup object.

        Args:
            object_name: Name/ID of the object
            num_grasps: Maximum number of grasps to return

        Returns:
            Tuple of (gripper_type, grasps) where:
            - gripper_type: String identifier for the gripper (e.g., "droid", "rum")
            - grasps: numpy array of shape (N, 4, 4) containing 4x4 transformation matrices
                      in object-local frame
        """
        pass

    @abstractmethod
    def load_grasps_for_joint(
        self,
        object_name: str,
        joint_name: str,
        num_grasps: int = 1000
    ) -> Tuple[str, np.ndarray]:
        """Load grasps for an articulated object joint.

        Args:
            object_name: Name/ID of the object
            joint_name: Name of the joint to load grasps for
            num_grasps: Maximum number of grasps to return

        Returns:
            Tuple of (gripper_type, grasps) where:
            - gripper_type: String identifier for the gripper (e.g., "droid", "rum")
            - grasps: numpy array of shape (N, 4, 4) containing 4x4 transformation matrices
                      in object-local frame
        """
        pass


class FileBasedGraspLoader(GraspLoader):
    """Loads grasps from pre-computed files (current implementation)."""

    def __init__(self):
        from molmo_spaces.utils.grasp_sample import (
            load_grasps_for_object as _load_grasps_for_object,
            load_grasps_for_object_per_joint as _load_grasps_for_object_per_joint,
        )
        self._load_grasps_for_object = _load_grasps_for_object
        self._load_grasps_for_object_per_joint = _load_grasps_for_object_per_joint

    def load_grasps_for_object(
        self,
        object_name: str,
        num_grasps: int = 1000
    ) -> Tuple[str, np.ndarray]:
        """Load grasps from files for a static object."""
        return self._load_grasps_for_object(object_name, num_grasps=num_grasps)

    def load_grasps_for_joint(
        self,
        object_name: str,
        joint_name: str,
        num_grasps: int = 1000
    ) -> Tuple[str, np.ndarray]:
        """Load grasps from files for an articulated object joint."""
        gripper, grasps = self._load_grasps_for_object_per_joint(
            object_name, joint_name, num_grasps=num_grasps
        )
        if gripper is None:
            raise ValueError(f"No grasps found for {object_name}/{joint_name}")
        return gripper, grasps


class PredictionModelGraspLoader(GraspLoader):
    """Loads grasps from a prediction model (example implementation).

    This is a template for implementing grasp prediction models.
    Subclass this and implement the prediction logic.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """Initialize the prediction model.

        Args:
            model_path: Path to the model checkpoint
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        # TODO: Load your model here
        # self.model = load_your_model(model_path)
        log.warning("PredictionModelGraspLoader is not fully implemented")

    def load_grasps_for_object(
        self,
        object_name: str,
        num_grasps: int = 1000
    ) -> Tuple[str, np.ndarray]:
        """Predict grasps using the model for a static object.

        Args:
            object_name: Name/ID of the object
            num_grasps: Number of grasps to predict

        Returns:
            Tuple of (gripper_type, grasps)
        """
        # TODO: Implement your prediction logic here
        # Example:
        # 1. Load object mesh/point cloud
        # 2. Run model inference
        # 3. Convert predictions to 4x4 transformation matrices
        # 4. Return (gripper_type, grasps)

        raise NotImplementedError(
            "Subclass PredictionModelGraspLoader and implement load_grasps_for_object"
        )

    def load_grasps_for_joint(
        self,
        object_name: str,
        joint_name: str,
        num_grasps: int = 1000
    ) -> Tuple[str, np.ndarray]:
        """Predict grasps using the model for an articulated object joint.

        Args:
            object_name: Name/ID of the object
            joint_name: Name of the joint
            num_grasps: Number of grasps to predict

        Returns:
            Tuple of (gripper_type, grasps)
        """
        # TODO: Implement your prediction logic here
        raise NotImplementedError(
            "Subclass PredictionModelGraspLoader and implement load_grasps_for_joint"
        )
