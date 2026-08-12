from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py

# Disable remote downloads, as you are working with local datasets
tfds.disable_progress_bar()
tfds.core.utils.gcs_utils.gcs_access = False


class DemoBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        # self.path = "data/train/episode_*.npy"
        self.path = "*.hdf5"
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'image_1': tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'front': tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist': tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, [6 joint, 1 gripper]',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=self.path),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            with h5py.File(episode_path, 'a') as f:
                data = f["data"]
                timestamps = data.keys()
                for i, ts in enumerate(timestamps):
                    new_data = data[ts]
                    trajectory = np.asarray(new_data['trajectory'])
                    images = np.asarray(new_data["observation"]["rgb"])
                    # qpos = np.asarray(new_data["observation"]["qpos"])
                    # ee_state = np.asarray(new_data["observation"]["ee_state"])
                    episode_length = trajectory.shape[0]
                    assert images.shape[0] == episode_length == trajectory.shape[0]
                    # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                    episode = []
                    # for i, step in enumerate(data):
                    for i in range(episode_length):
                        action = trajectory[i]
                        if action[-1] > 0.03:
                            action = np.concatenate([action[:6], np.array([1])]).astype(np.float32)
                        else:
                            action = np.concatenate([action[:6], np.array([0])]).astype(np.float32)
                        episode.append({
                            'observation': {
                                'image_0': images[i][0],
                                'image_1': images[i][1],
                                'front': images[i][2],
                                'wrist': images[i][3],
                            },
                            'action': action,
                            'discount': 1.0,
                            'reward': float(i == (episode_length - 1)),
                            'is_first': i == 0,
                            'is_last': i == (episode_length - 1),
                            'is_terminal': i == (episode_length - 1),
                            'language_instruction': np.asarray(new_data['instruction'])[0].decode('utf-8'),
                        })

                    # create output data sample
                    sample = {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': episode_path
                        }
                    }

                    # if you want to skip an example for whatever reason, simply return None
                    return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            if sample.endswith("hdf5"):
                try:
                    with h5py.File(sample, 'a') as f:
                        data = f["data"]
                except Exception as e:
                    print(f"Error reading {sample}: {e}, pass")
                    continue
            yield _parse_example(sample)


