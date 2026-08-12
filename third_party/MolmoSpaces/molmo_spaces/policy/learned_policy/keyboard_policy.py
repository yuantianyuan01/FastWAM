import logging
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Keyboard_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        from pynput import keyboard

        self._keyboard = keyboard
        self.robot_type = exp_config.robot_config.name
        self.step_size = exp_config.policy_config.step_size
        self.rot_step = exp_config.policy_config.rot_step
        self._pressed = set()
        self._gripper_open = True
        self._listener = self._keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        log.info(
            "Keyboard policy started. "
            "w/s: up/down (z), arrows: x/y. "
            "a/d: yaw, e/r: pitch, z/c: roll. "
            "Space: toggle gripper. q: pause."
        )

    def _on_press(self, key):
        self._pressed.add(key)
        if key == self._keyboard.Key.space:
            self._gripper_open = not self._gripper_open

    def _on_release(self, key):
        self._pressed.discard(key)

    def _key(self, char):
        return self._keyboard.KeyCode.from_char(char) in self._pressed

    def _get_delta_position(self):
        dx, dy, dz = 0.0, 0.0, 0.0
        if self._keyboard.Key.up in self._pressed:
            dx += self.step_size
        if self._keyboard.Key.down in self._pressed:
            dx -= self.step_size
        if self._keyboard.Key.left in self._pressed:
            dy += self.step_size
        if self._keyboard.Key.right in self._pressed:
            dy -= self.step_size
        if self._key("w"):
            dz -= self.step_size
        if self._key("s"):
            dz += self.step_size
        return np.array([dx, dy, dz])

    def _get_delta_rotation(self):
        roll, pitch, yaw = 0.0, 0.0, 0.0
        if self._key("a"):
            yaw += self.rot_step
        if self._key("d"):
            yaw -= self.rot_step
        if self._key("e"):
            pitch += self.rot_step
        if self._key("r"):
            pitch -= self.rot_step
        if self._key("z"):
            roll += self.rot_step
        if self._key("c"):
            roll -= self.rot_step
        return R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    def _is_paused(self):
        return self._key("q")

    def prepare_model(self):
        pass

    def reset(self):
        self.init_robot_pose = None
        self.init_tcp_pose = None
        self.current_position = None
        self.current_rotation = None
        self._gripper_open = True

    def render(self, obs):
        if self.robot_type == "floating_rum":
            try:
                window_exists = cv2.getWindowProperty("views", cv2.WND_PROP_VISIBLE) >= 0
            except cv2.error:
                window_exists = False

            if not window_exists:
                screen_res = (1920, 1080)
                aspect_ratio = obs["wrist_camera"].shape[0] / obs["wrist_camera"].shape[1]
                new_width = int(screen_res[0] * 0.9)
                new_height = int(new_width * aspect_ratio)
                cv2.namedWindow("views", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("views", new_width, new_height)
                cv2.moveWindow(
                    "views", (screen_res[0] - new_width) // 2, (screen_res[1] - new_height) // 2
                )

            cv2.imshow("views", cv2.cvtColor(obs["wrist_camera"], cv2.COLOR_RGB2BGR))
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
        if isinstance(obs, list):
            obs = obs[0]
        self.render(obs)

        robot_pose = obs["robot_base_pose"]
        T_world_robot = np.eye(4)
        T_world_robot[:3, :3] = R.from_quat(robot_pose[3:], scalar_first=True).as_matrix()
        T_world_robot[:3, 3] = robot_pose[:3]

        if self.robot_type == "floating_rum":
            if self.init_robot_pose is None:
                self.init_robot_pose = T_world_robot
                self.current_position = T_world_robot[:3, 3].copy()
                self.current_rotation = T_world_robot[:3, :3].copy()
            return {"T_world_robot": T_world_robot}

        tcp_pose = obs["tcp_pose"]
        qpos = obs["qpos"]["arm"]

        T_robot_tcp = np.eye(4)
        T_robot_tcp[:3, :3] = R.from_quat(tcp_pose[3:], scalar_first=True).as_matrix()
        T_robot_tcp[:3, 3] = tcp_pose[:3]

        if self.init_tcp_pose is None:
            self.init_tcp_pose = T_robot_tcp.copy()
            self.current_position = T_robot_tcp[:3, 3].copy()
            self.current_rotation = T_robot_tcp[:3, :3].copy()

        T_world_tcp = T_world_robot @ T_robot_tcp
        return {
            "qpos": qpos,
            "T_world_robot": T_world_robot,
            "T_world_tcp": T_world_tcp,
            "T_robot_tcp": T_robot_tcp,
        }

    def inference_model(self, model_input):
        if self._is_paused():
            return None

        self.current_position += self.current_rotation @ self._get_delta_position()
        self.current_rotation = self._get_delta_rotation() @ self.current_rotation

        if self.robot_type == "floating_rum":
            goal_pose = self.init_robot_pose.copy()
            goal_pose[:3, 3] = self.current_position
            goal_pose[:3, :3] = self.current_rotation

            goal_pose_7d = np.array(
                list(goal_pose[:3, 3])
                + list(R.from_matrix(goal_pose[:3, :3]).as_quat(scalar_first=True))
            )
            return {
                "base": goal_pose_7d,
                "gripper": np.array([0.0]) if self._gripper_open else np.array([-255.0]),
            }
        else:
            kinematics = self.task.env.current_robot.kinematics
            robot_view = self.task.env.current_robot.robot_view

            gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
            mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]

            new_pose = np.eye(4)
            new_pose[:3, 3] = self.current_position
            new_pose[:3, :3] = self.current_rotation

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

            action["gripper"] = np.array([0.0]) if self._gripper_open else np.array([255.0])
            return action

    def model_output_to_action(self, model_output):
        return model_output

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "keyboard"
        info["timestamp"] = time.time()
        return info
