from typing import Any
import numpy as np

from molmo_spaces.configs.camera_configs import MjcfCameraConfig, RobotMountedCameraConfig

def patch_droid_config_for_rum(frozen_config, data: Any = None):

    assert "FrankaRobotConfig" in str(type(frozen_config.robot_config))

    if isinstance(frozen_config.camera_config.cameras[0], MjcfCameraConfig):
        assert frozen_config.camera_config.cameras[0].mjcf_name == "gripper/wrist_camera"
        frozen_config.camera_config.cameras[0].mjcf_name = "wrist_cam"
    elif isinstance(frozen_config.camera_config.cameras[0], RobotMountedCameraConfig):
        assert frozen_config.camera_config.cameras[0].reference_body_names == [
            "robot_0/gripper/base"
        ]
        frozen_config.camera_config.cameras[0].reference_body_names = ["robot_0/base"]

    frozen_config.camera_config.cameras[0].record_depth = True
    frozen_config.camera_config.cameras[0].camera_offset = [0.0, 0.0, 0.0]
    frozen_config.camera_config.cameras[0].lookat_offset = [0.0, 0.0, 0.08]
    frozen_config.camera_config.cameras[0].camera_quaternion = [1.0, 0.0, 0.0, 0.0]
    frozen_config.camera_config.cameras[0].fov = 53.1
    frozen_config.camera_config.img_resolution = (960, 720)
    frozen_config.camera_config.cameras[0].reference_body_names = ["robot_0/wrist_cam_body"]

    assert frozen_config.camera_config.cameras[1].reference_body_names == ["robot_0/fr3_link0"]
    frozen_config.camera_config.cameras[1].reference_body_names = ["robot_0/base"]

    task_class_str = frozen_config.task_cls_str.lower()
    frozen_config.task_config.robot_base_pose[2] += 0.8
    if "close" in task_class_str or "open" in task_class_str:
        grasp_pose = data["obs"]["extra"]["grasp_pose"][0]
        robot_desired_pose = frozen_config.task_config.robot_base_pose.copy()
        robot_desired_pose[2] = grasp_pose[2]
        delta = np.array(robot_desired_pose[0:2]) - np.array(grasp_pose[0:2])
        min_dist = 0.75
        if np.linalg.norm(delta) < min_dist:
            robot_desired_pose[:2] = np.array(
                grasp_pose[0:2] + delta / np.linalg.norm(delta) * min_dist
            ).tolist()
        frozen_config.task_config.robot_base_pose = robot_desired_pose

    return frozen_config
