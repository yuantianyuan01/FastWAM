import logging
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.rum_client import RUMClient

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def action_tensor_to_matrix(action_tensor, rot_unit):
    affine = np.eye(4)
    if rot_unit == "euler":
        r = R.from_euler("xyz", action_tensor[3:6], degrees=False)
    elif rot_unit == "axis":
        r = R.from_rotvec(action_tensor[3:6])
    else:
        raise NotImplementedError
    affine[:3, :3] = r.as_matrix()
    affine[:3, -1] = action_tensor[:3]
    return affine


class CAP_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.grasping_type = exp_config.policy_config.grasping_type
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.use_vlm = exp_config.policy_config.use_vlm
        self.use_exo = exp_config.policy_config.exo_vlm
        self.model = None

    def prepare_model(self):
        host = "localhost"
        port = 8765

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.model = RUMClient(host=host, port=port)
                metadata = self.model.get_server_metadata()
                self.model_name = metadata.get("checkpoint", "rum")
                log.info(f"Connected to RUM server at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to RUM server after {max_retries} attempts")
                    raise

    def reset(self):
        self.starting_time = None
        self.T_world_object = None
        self.T_world_camera = None
        self.T_world_rum = None
        self.is_grasping = False
        self.step_counter = 0

    def render(self, obs):
        views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        if self.model is None:
            self.prepare_model()

        if hasattr(self, "T_world_object") is False or self.T_world_object is None:
            if not self.use_vlm:
                T_base_object = obs["object_poses"][self.task.config.task_config.pickup_obj_name]
                T_world_base = np.eye(4)
                T_world_base[:3, 3] = obs["robot_base_pose"][:3]
                T_world_base[:3, :3] = R.from_quat(
                    obs["robot_base_pose"][3:7], scalar_first=True
                ).as_matrix()
                self.T_world_object = T_world_base @ T_base_object
            else:
                exo_depth = obs["exo_camera_1_depth"]
                exo_rgb = obs["exo_camera_1"]
                ego_depth = obs["wrist_camera_depth"]
                ego_rgb = obs["wrist_camera"]
                point_norm = self.model.infer_point(
                    rgb=exo_rgb if self.use_exo else ego_rgb,
                    object_name=self.task.config.task_config.referral_expressions[
                        "pickup_obj_name"
                    ],
                    task=self.config.task_type,
                )
                if self.use_exo:
                    K = np.array(obs["sensor_param_exo_camera_1"]["intrinsic_cv"])
                else:
                    K = np.array(obs["sensor_param_wrist_camera"]["intrinsic_cv"])
                fovy = 2 * np.arctan((2 * K[1, 2]) / (2 * K[1, 1]))
                x_norm, y_norm = point_norm
                width, height = (
                    (exo_rgb.shape[1], exo_rgb.shape[0])
                    if self.use_exo
                    else (ego_rgb.shape[1], ego_rgb.shape[0])
                )
                x = int(x_norm * width)
                y = int(y_norm * height)
                depth_value = exo_depth[y, x] + 0.03 if self.use_exo else ego_depth[y, x] + 0.03
                f = height / (2 * np.tan(fovy / 2))
                cam_mat = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
                cx = cam_mat[0, 2]
                cy = cam_mat[1, 2]
                fx = cam_mat[0, 0]
                fy = cam_mat[1, 1]
                z_cam = -depth_value
                x_cam = -(x - cx) * z_cam / fx
                y_cam = -(cy - y) * z_cam / fy
                p_cam = np.array([x_cam, y_cam, z_cam])
                T_corr = np.eye(4)
                T_corr[:3, :3] = np.diag([1, -1, -1])
                if self.use_exo:
                    camera_pose = (
                        np.array(obs["sensor_param_exo_camera_1"]["cam2world_gl"].copy()) @ T_corr
                    )
                else:
                    camera_pose = (
                        np.array(obs["sensor_param_wrist_camera"]["cam2world_gl"].copy()) @ T_corr
                    )
                p_world = camera_pose[:3, :3] @ p_cam + camera_pose[:3, 3]
                self.T_world_object = np.eye(4)
                self.T_world_object[:3, 3] = p_world

        T_base_ego = np.eye(4)
        T_base_ego[:3, 3] = obs["tcp_pose"][:3]
        T_base_ego[:3, :3] = R.from_quat(obs["tcp_pose"][3:7], scalar_first=True).as_matrix()

        T_world_base = np.eye(4)
        T_world_base[:3, 3] = obs["robot_base_pose"][:3]
        T_world_base[:3, :3] = R.from_quat(
            obs["robot_base_pose"][3:7], scalar_first=True
        ).as_matrix()

        T_world_ego = T_world_base @ T_base_ego
        self.T_world_camera = T_world_ego
        T_camera_object = np.linalg.inv(T_world_ego) @ self.T_world_object
        object_3d_position = T_camera_object[:3, 3]
        object_3d_position = np.array(
            [-T_camera_object[0, 3], -T_camera_object[2, 3], -T_camera_object[1, 3]]
        )

        if self.is_grasping:
            object_3d_position = np.array([0.00, 0.18, 0.04])
        return {
            "rgb_ego": cv2.resize(obs["wrist_camera"], (224, 224)),
            "object_3d_position": object_3d_position,
        }

    def inference_model(self, model_input):
        self.step_counter += 1

        model_output = self.model.infer(model_input)
        model_output[0][:3] = np.array(
            [-model_output[0][0], -model_output[0][2], -model_output[0][1]]
        )
        model_output[0][3:6] = np.array(
            [-model_output[0][3], -model_output[0][5], -model_output[0][4]]
        )
        self.is_grasping = max(self.is_grasping, model_output[0][6] < self.grasping_threshold)
        delta_pose_mat = action_tensor_to_matrix(model_output[0][:6], "euler")
        T_world_ego = self.T_world_camera @ delta_pose_mat
        self.T_world_rum = T_world_ego

        goal_pose_7d = np.array(
            list(self.T_world_rum[:3, 3])
            + list(R.from_matrix(self.T_world_rum[:3, :3]).as_quat(scalar_first=True))
        )

        goal_pose_homogeneous = np.eye(4)
        goal_pose_homogeneous[:3, 3] = goal_pose_7d[:3]
        goal_pose_homogeneous[:3, :3] = R.from_quat(
            goal_pose_7d[3:7], scalar_first=True
        ).as_matrix()

        kinematics = self.task.env.current_robot.kinematics
        robot_view = self.task.env.current_robot.robot_view
        gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
        mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]
        new_pose = goal_pose_homogeneous.copy()

        jp = kinematics.ik(
            "arm",
            new_pose,
            mgs_except_gripper,
            robot_view.get_qpos_dict(),
            robot_view.base.pose,
            rel_to_base=False,
        )
        action = robot_view.get_ctrl_dict()
        if jp is not None:
            action.update({mg_id: jp[mg_id] for mg_id in mgs_except_gripper})

        if self.grasping_type == "binary":
            if self.is_grasping:
                action["gripper"] = np.array([-255.0])
            else:
                action["gripper"] = np.array([0.0])
        else:
            action["gripper"] = (1 - model_output[0][6]) * np.array([-255.0])
            if self.is_grasping:
                action["gripper"] = np.array([-255.0])
        return action

    def model_output_to_action(self, model_output):
        return model_output

    def get_info(self) -> dict:
        info = super().get_info()

        info["policy_checkpoint"] = self.model_name
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
