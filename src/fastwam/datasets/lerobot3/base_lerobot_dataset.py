"""FastWAM adapter for the LeRobot v3.0 reader."""

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset as _BaseLerobotDataset

from .lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset


class BaseLerobotDataset(_BaseLerobotDataset):
    metadata_cls = LeRobotDatasetMetadata
    multi_dataset_cls = MultiLeRobotDataset
    presample_images = True
