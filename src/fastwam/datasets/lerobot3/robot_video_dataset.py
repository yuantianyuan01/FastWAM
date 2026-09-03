"""FastWAM robot-video dataset backed by the LeRobot v3.0 reader."""

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset as _RobotVideoDataset

from .base_lerobot_dataset import BaseLerobotDataset


class RobotVideoDataset(_RobotVideoDataset):
    base_dataset_cls = BaseLerobotDataset
