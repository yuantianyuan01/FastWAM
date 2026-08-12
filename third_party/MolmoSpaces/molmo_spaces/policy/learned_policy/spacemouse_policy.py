import logging
import threading
import time

import cv2
import numpy as np
from pynput import keyboard

try:
    import hid
except ImportError as e:
    raise ImportError(
        f"Failed to import 'hid': {e}\n\n"
        "The SpaceMouse policy requires libhidapi. To fix this:\n"
        "  1. Install the system library:\n"
        "       sudo apt-get install libhidapi-hidraw0\n"
        "  2. Make sure the Python package is installed:\n"
        "       pip install hidapi\n"
        "  3. You may also need udev rules to access the device without sudo:\n"
        '       echo \'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"\' | sudo tee /etc/udev/rules.d/99-spacemouse.rules\n'
        "       sudo udevadm control --reload-rules && sudo udevadm trigger\n"
    ) from e
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VENDOR_ID = 9583
DOUBLE_CLICK_TIME = 0.3


def _to_int16(y1, y2):
    x = y1 | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def _convert(b1, b2, scale=350.0):
    x = _to_int16(b1, b2) / scale
    return min(max(x, -1.0), 1.0)


class _SpaceMouseReader:
    def __init__(self, product_id):
        self._control = [0.0] * 6
        self.grasp_state = False
        self.engage = False
        self._lock = threading.Lock()
        self._t_last_right = -1.0

        self._dev = hid.device()
        try:
            self._dev.open(VENDOR_ID, product_id)
        except OSError as e:
            raise OSError(
                f"Failed to open SpaceMouse (vendor=0x{VENDOR_ID:04x}, product=0x{product_id:04x}): {e}\n\n"
                "Possible fixes:\n"
                "  1. Unplug and replug the SpaceMouse after setting udev rules.\n"
                "  2. If udev rules are not set yet:\n"
                '       echo \'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"\' | sudo tee /etc/udev/rules.d/99-spacemouse.rules\n'
                "       sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                "  3. Verify the product ID matches your device:\n"
                "       python -c \"import hid; [print(hex(d['vendor_id']), hex(d['product_id']), d['product_string']) for d in hid.enumerate(0x256f, 0)]\"\n"
            ) from e
        log.info(
            f"SpaceMouse: {self._dev.get_manufacturer_string()} {self._dev.get_product_string()}"
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        x, y, z, roll, pitch, yaw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        prev_left = False
        prev_right = False

        while True:
            d = self._dev.read(13)
            if not d:
                continue

            if d[0] == 1:
                x = _convert(d[1], d[2])
                y = _convert(d[3], d[4])
                z = _convert(d[5], d[6])
                with self._lock:
                    self._control[:3] = [y, -x, z]

            elif d[0] == 2:
                roll = _convert(d[3], d[4])
                pitch = _convert(d[1], d[2]) * -1.0
                yaw = _convert(d[5], d[6])
                with self._lock:
                    self._control[3:] = [roll, pitch, yaw]

            elif d[0] == 3:
                cur_left = bool(d[1] & 1)
                cur_right = bool(d[1] & 2)

                if cur_right and not prev_right:
                    now = time.time()
                    if now - self._t_last_right < DOUBLE_CLICK_TIME:
                        with self._lock:
                            self.engage = False
                            self._control = [0.0] * 6
                    else:
                        with self._lock:
                            self.engage = True
                    self._t_last_right = now

                if not cur_left and prev_left:
                    with self._lock:
                        self.grasp_state = not self.grasp_state

                prev_left = cur_left
                prev_right = cur_right

    @property
    def control(self):
        with self._lock:
            return list(self._control)

    @property
    def gripper_open(self):
        with self._lock:
            return not self.grasp_state

    @property
    def engaged(self):
        with self._lock:
            return self.engage

    def reset_grasp(self):
        with self._lock:
            self.grasp_state = False


class SpaceMouse_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.robot_type = exp_config.robot_config.name
        self.pos_sensitivity = exp_config.policy_config.pos_sensitivity
        self.rot_sensitivity = exp_config.policy_config.rot_sensitivity
        self._mouse = _SpaceMouseReader(exp_config.policy_config.product_id)
        self._pressed = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        log.info(
            "SpaceMouse ready. Right click: engage. Left click: toggle gripper. Right double-click: disengage. q: pause."
        )

    def _on_press(self, key):
        self._pressed.add(key)

    def _on_release(self, key):
        self._pressed.discard(key)

    def _key(self, char):
        return keyboard.KeyCode.from_char(char) in self._pressed

    def _is_paused(self):
        return self._key("q")

    def reset(self):
        self.init_robot_pose = None
        self.init_tcp_pose = None
        self.current_position = None
        self.current_rotation = None
        self._mouse.reset_grasp()

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

        ctrl = self._mouse.control if self._mouse.engaged else [0.0] * 6
        delta_pos = np.array(ctrl[:3]) * self.pos_sensitivity
        delta_rot = R.from_euler("xyz", np.array(ctrl[3:]) * self.rot_sensitivity).as_matrix()

        new_position = self.current_position + self.current_rotation @ delta_pos
        new_rotation = delta_rot @ self.current_rotation

        gripper_open = self._mouse.gripper_open

        if self.robot_type == "floating_rum":
            self.current_position = new_position
            self.current_rotation = new_rotation
            goal_pose = self.init_robot_pose.copy()
            goal_pose[:3, 3] = self.current_position
            goal_pose[:3, :3] = self.current_rotation

            goal_pose_7d = np.array(
                list(goal_pose[:3, 3])
                + list(R.from_matrix(goal_pose[:3, :3]).as_quat(scalar_first=True))
            )
            return {
                "base": goal_pose_7d,
                "gripper": np.array([0.0]) if gripper_open else np.array([-255.0]),
            }
        else:
            kinematics = self.task.env.current_robot.kinematics
            robot_view = self.task.env.current_robot.robot_view

            gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
            mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]

            new_pose = np.eye(4)
            new_pose[:3, 3] = new_position
            new_pose[:3, :3] = new_rotation

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
                self.current_position = new_position
                self.current_rotation = new_rotation
                action.update({mg_id: jp[mg_id] for mg_id in mgs_except_gripper})

            action["gripper"] = np.array([0.0]) if gripper_open else np.array([255.0])
            return action

    def model_output_to_action(self, model_output):
        return model_output

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "spacemouse"
        info["timestamp"] = time.time()
        return info
