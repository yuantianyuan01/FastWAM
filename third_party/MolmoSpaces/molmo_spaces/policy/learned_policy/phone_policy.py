import logging
import time

import cv2
import numpy as np
import teledex
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Phone_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.robot_type = exp_config.robot_config.name
        self.session = teledex.Session()
        self.session.start()
        while self.session.get_latest_data()["position"] is None:
            time.sleep(0.1)
        log.info(f"Teledex session connected. Robot type: {self.robot_type}")

    def reset(self):
        self.init_robot_pose = None
        self.init_phone_pose = None
        self.init_tcp_pose = None
        self.current_grasp = self.session.get_latest_data()["toggle"]

    def render(self, obs):
        if self.robot_type == "floating_rum":
            views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)

            try:
                window_exists = cv2.getWindowProperty("views", cv2.WND_PROP_VISIBLE) >= 0
            except cv2.error:
                window_exists = False
            if not window_exists:
                screen_res = (1920, 1080)
                aspect_ratio = views.shape[0] / views.shape[1]
                new_width = int(screen_res[0] * 0.9)
                new_height = int(new_width * aspect_ratio)
                cv2.namedWindow("views", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("views", new_width, new_height)
                cv2.moveWindow(
                    "views", (screen_res[0] - new_width) // 2, (screen_res[1] - new_height) // 2
                )
            cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        else:
            views = np.concatenate(
                [np.flip(np.flip(obs["wrist_camera"], axis=0), axis=1), obs["exo_camera_1"]], axis=1
            )

            try:
                window_exists = cv2.getWindowProperty("views", cv2.WND_PROP_VISIBLE) >= 0
            except cv2.error:
                window_exists = False
            if not window_exists:
                screen_res = (1920, 1080)
                aspect_ratio = views.shape[0] / views.shape[1]
                new_width = int(screen_res[0] * 0.9)
                new_height = int(new_width * aspect_ratio)
                cv2.namedWindow("views", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("views", new_width, new_height)
                cv2.moveWindow(
                    "views", (screen_res[0] - new_width) // 2, (screen_res[1] - new_height) // 2
                )
            cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        if self.robot_type == "floating_rum":
            self.render(obs)

            T_world_robot = np.eye(4)
            T_world_robot[:3, 3] = obs["robot_base_pose"][:3]
            T_world_robot[:3, :3] = R.from_quat(
                obs["robot_base_pose"][3:7], scalar_first=True
            ).as_matrix()

            if self.init_robot_pose is None:
                self.init_robot_pose = T_world_robot

            return {
                "T_world_robot": T_world_robot,
            }
        else:
            self.render(obs)
            robot_pose = obs["robot_base_pose"]
            tcp_pose = obs["tcp_pose"]
            qpos = obs["qpos"]["arm"]

            T_world_robot = np.eye(4)
            T_world_robot[:3, :3] = R.from_quat(robot_pose[3:], scalar_first=True).as_matrix()
            T_world_robot[:3, 3] = robot_pose[:3]

            T_robot_tcp = np.eye(4)
            T_robot_tcp[:3, :3] = R.from_quat(tcp_pose[3:], scalar_first=True).as_matrix()
            T_robot_tcp[:3, 3] = tcp_pose[:3]

            if self.init_tcp_pose is None:
                self.init_tcp_pose = T_robot_tcp

            T_world_tcp = T_world_robot @ T_robot_tcp

            return {
                "qpos": qpos,
                "T_world_robot": T_world_robot,
                "T_world_tcp": T_world_tcp,
                "T_robot_tcp": T_robot_tcp,
            }

    def inference_model(self, model_input):
        if self.session.get_latest_data()["button"]:
            return None
        if self.robot_type == "floating_rum":
            phone_pose = np.eye(4)
            phone_pose[:3, :3] = self.session.get_latest_data()["rotation"].reshape(3, 3)
            phone_pose[:3, 3] = self.session.get_latest_data()["position"].reshape(3)

            if self.init_phone_pose is None:
                self.init_phone_pose = phone_pose

            transform = np.eye(4)
            transform[:3, :3] = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
            delta_phone_pose = (
                transform
                @ np.linalg.inv(self.init_phone_pose)
                @ phone_pose
                @ np.linalg.inv(transform)
            )
            delta_phone_pose[:3, 3] *= 3  # sensitivity

            goal_pose = self.init_robot_pose @ delta_phone_pose

            goal_pose_7d = np.array(
                list(goal_pose[:3, 3])
                + list(R.from_matrix(goal_pose[:3, :3]).as_quat(scalar_first=True))
            )
            action = {
                "base": goal_pose_7d,
                "gripper": np.array([0.0])
                if self.session.get_latest_data()["toggle"] == self.current_grasp
                else np.array([-255.0]),
            }
            return action
        else:
            kinematics = self.task.env.current_robot.kinematics
            robot_view = self.task.env.current_robot.robot_view

            gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
            mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]

            phone_pose = np.eye(4)
            phone_pose[:3, :3] = self.session.get_latest_data()["rotation"].reshape(3, 3)
            phone_pose[:3, 3] = self.session.get_latest_data()["position"].reshape(3)

            if self.init_phone_pose is None:
                self.init_phone_pose = phone_pose

            delta_phone_pose = self.init_phone_pose @ np.linalg.inv(phone_pose)
            delta_phone_position = -delta_phone_pose[:3, 3] * 2  # Negate to flip directions
            delta_phone_rotation = delta_phone_pose[:3, :3].T  # Transpose to invert rotation
            new_pose = np.eye(4)
            new_pose[:3, :3] = delta_phone_rotation @ self.init_tcp_pose[:3, :3]
            new_pose[:3, 3] = self.init_tcp_pose[:3, 3] + delta_phone_position

            jp = kinematics.ik(
                "arm",
                new_pose,
                mgs_except_gripper,
                robot_view.get_qpos_dict(),
                robot_view.base.pose,
                rel_to_base=True,
            )
            action = robot_view.get_ctrl_dict()
            if jp is not None:
                action.update({mg_id: jp[mg_id] for mg_id in mgs_except_gripper})

            if self.session.get_latest_data()["toggle"] != self.current_grasp:
                action["gripper"] = np.array([255.0])
            else:
                action["gripper"] = np.array([0.0])
            return action

    def model_output_to_action(self, model_output):
        return model_output

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "mujoco_ar"
        info["timestamp"] = time.time()
        return info
