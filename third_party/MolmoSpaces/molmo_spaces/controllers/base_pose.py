from abc import ABC, abstractmethod

import numpy as np

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.robots.robot_views.abstract import RobotBaseGroup
from molmo_spaces.utils.controller_utils import optimize_all_steer_and_drive


class BasePoseController(Controller, ABC):
    """
    Abstract pose controller class for a simple 2D pose controller for a robot base.
    Computes velocities for the robot base's actuators based on target pose, and clips to control limits.
    NOTE: Assumes base actuators (eg. wheels) use velocity inputs.

    This class cannot be instantiated directly and must be subclassed with an implementation
    of the "compute_base_velocities" method.
    """

    def __init__(self, robot_move_group: RobotBaseGroup) -> None:
        super().__init__(robot_move_group)

        self.ctrl_dim = robot_move_group.n_actuators
        self.ctrl_range = robot_move_group.ctrl_limits

        self._stationary = True
        self._target = self.robot_move_group.pose.copy()  # Initial target pose is current base pose

        # TODO: Check if the move_group's actuators support this control type

        self.reset()

    @property
    def stationary(self):
        """Whether the controller is in stationary mode"""
        return self._stationary

    @property
    def target(self):
        """The target pose for the controller"""
        return self._target

    def set_target(self, target_base_pose) -> None:
        """Set the target pose for the controller"""
        self._stationary = False  # Exit stationary mode when target is provided

        self._target = target_base_pose.copy()

    def set_to_stationary(self) -> None:
        """
        This method sets the robot to stationary mode and computes the targets to hold the
        robot stationary.
        This is useful when the robot needs to be stopped at a certain position and not drift.
        """
        # Set to stationary mode and set target to current base pose to hold it stationary
        self._stationary = True
        current_base_pose = self.robot_move_group.pose
        self._target = current_base_pose.copy()

    def compute_ctrl_inputs(self):
        """
        Compute the control inputs based on the current state and the target set by the user.

        Returns:
            The control inputs to be applied to the robot actuators, in this case: positions
        """
        # velocity control inputs: compute base velocities to achieve target pose
        ctrl_inputs = self.compute_base_velocities(self._target)
        # Clip to control limits
        ctrl_inputs = np.clip(ctrl_inputs, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

        return ctrl_inputs

    @abstractmethod
    def compute_base_velocities(self, target_base_pose):
        """
        Abstract method to compute base velocities to achieve the target pose.
        Must be implemented by subclasses.
        Methods could include: Differential-drive, Ackermann steering, Omni-wheel control etc.
        """
        pass

    def reset(self) -> None:
        """Reset the controller to its initial state, clearing any internal state or targets"""
        self.set_to_stationary()  # Explicit reset to stationary mode


class DiffDriveBasePoseController(BasePoseController):
    def __init__(self, robot_config, robot_move_group: RobotBaseGroup) -> None:
        """
        Differential drive base pose controller.

        Args:
            robot_config: The configuration of the robot, containing special information about the control
            robot_move_group: The move group of the robot base to be controlled
        """
        self.robot_config = robot_config
        self.wheel_base = robot_config.wheel_base  # Distance between the wheels
        self.wheel_radius = robot_config.wheel_radius  # Radius of the wheels

        super().__init__(robot_move_group)

    def compute_base_velocities(self, target_base_pose):
        """
        Compute wheel velocities to drive the robot base to the target pose using a simple proportional controller.

        Args:
            target_base_pose: Target [x, y, theta] pose in world frame.

        Returns:
            np.ndarray: [left_wheel_velocity, right_wheel_velocity]
        """
        # TODO: Test this controller
        # TODO: Handle case where base_pose frame is not in the center of the wheels

        # Current pose
        current_pose = self.robot_move_group.pose  # [x, y, theta]
        x, y, theta = current_pose
        x_t, y_t, theta_t = target_base_pose

        # Compute pose error in world frame
        dx = x_t - x
        dy = y_t - y

        # Transform error to robot frame
        error_x = np.cos(theta) * dx + np.sin(theta) * dy
        -np.sin(theta) * dx + np.cos(theta) * dy
        error_theta = np.arctan2(np.sin(theta_t - theta), np.cos(theta_t - theta))

        # Proportional gains
        k_rho = 1.0
        k_alpha = 2.0

        # Compute control signals
        v = k_rho * error_x
        omega = k_alpha * error_theta

        # Convert (v, omega) to wheel velocities
        v_l = (v - (self.wheel_base / 2.0) * omega) / self.wheel_radius
        v_r = (v + (self.wheel_base / 2.0) * omega) / self.wheel_radius

        return np.array([v_l, v_r])


class SwerveBasePoseController(BasePoseController):
    def __init__(self, robot_config, robot_move_group: RobotBaseGroup) -> None:
        """
        Swerve drive base pose controller.

        Args:
            robot_config: The configuration of the robot, containing special information about the control
            robot_move_group: The move group of the robot base to be controlled
        """
        self.robot_config = robot_config
        self.steer_track = robot_config.steer_track
        self.wheel_base = robot_config.wheel_base  # Distance between the wheels
        self.wheel_radius = robot_config.wheel_radius  # Radius of the wheels
        self.steer_angle_range = robot_config.steer_angle_range
        self.max_wheel_speed = robot_config.max_wheel_speed
        self.current_steer_angles = np.zeros(4)

        super().__init__(robot_move_group)

    def compute_base_velocities(self, target_base_pose):
        """
        Compute wheel velocities and steer angles to drive the robot base to the target pose using a simple proportional controller.

        Args:
            target_base_pose: Target [x, y, theta] pose in world frame.

        Returns:
            np.ndarray: [front_left_steer_angle,   front_right_steer_angle,    back_left_steer_angle,    back_right_steer_angle,
                        front_left_wheel_velocity, front_right_wheel_velocity, back_left_wheel_velocity, back_right_wheel_velocity]

        Assert local frame defination:

                           ↑  x
                           |
                   fl      |       fr
                    o------|------o
                    |      |      |
                    |      |      |
                    |      |      |
                    |      |      |
              y <----------o      |
                    |             |
                    |             |
                    |             |
                    |             |
                    o-------------o
                   rl              rr
        """
        # TODO: Test this controller
        # TODO: Handle case where base_pose frame is not in the center of the wheels

        # Current pose
        current_pose = self.robot_move_group.pose  # [x, y, theta]
        x, y, theta = current_pose
        x_t, y_t, theta_t = target_base_pose

        # Compute pose error in world frame
        dx = x_t - x
        dy = y_t - y

        # Transform error to robot frame
        error_x = np.cos(theta) * dx + np.sin(theta) * dy
        error_y = -np.sin(theta) * dx + np.cos(theta) * dy
        error_theta = np.arctan2(np.sin(theta_t - theta), np.cos(theta_t - theta))

        # Proportional gains
        k_rho = 1.0
        k_alpha = 2.0

        # Compute control signals
        v_lin_x = k_rho * error_x
        v_lin_y = k_rho * error_y
        v_yaw = k_alpha * error_theta

        if np.abs(v_lin_y) < 0.01:
            v_lin_y = 0.0
        if np.abs(v_lin_x) < 0.01:
            v_lin_x = 0.0
        if np.abs(v_yaw) < 0.01:
            v_yaw = 0.0
        if np.abs(v_lin_y) < 0.01 and np.abs(v_lin_x) < 0.01 and np.abs(v_yaw) < 0.01:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if np.linalg.norm([v_lin_x, v_lin_y]) < 0.03:
            v_lin_x, v_lin_y = 0.0, 0.0
            v_yaw = np.sign(v_yaw) * np.pi * 2

        # Kinematic model: convert vx, vy, vw to steer angles and wheel velocities
        wheel_coords = [
            (-self.wheel_base / 2, self.steer_track / 2),  # fl
            (self.wheel_base / 2, self.steer_track / 2),  # fr
            (-self.wheel_base / 2, -self.steer_track / 2),  # rl
            (self.wheel_base / 2, -self.steer_track / 2),  # rr
        ]

        steer_ang = []
        drive_vel = []
        for wx, wy in wheel_coords:
            vix = v_lin_x + wx * v_yaw
            viy = v_lin_y + wy * v_yaw
            angle = np.arctan2(viy, vix)
            speed = np.hypot(viy, vix) / self.wheel_radius
            steer_ang.append(angle)
            drive_vel.append(speed)
        steer_ang = np.array(steer_ang)
        drive_vel = np.array(drive_vel)

        # Optimize steer angle and wheel speed to find nearest equivalent angle and speed
        optimized_steer_ang, optimized_drive_vel = optimize_all_steer_and_drive(
            self.current_steer_angles,
            steer_ang,
            drive_vel,
            self.steer_angle_range,
            self.max_wheel_speed,
        )
        self.current_steer_angles = optimized_steer_ang.copy()

        # Update the joint velocity and position.
        return np.concatenate([optimized_steer_ang, optimized_drive_vel])
