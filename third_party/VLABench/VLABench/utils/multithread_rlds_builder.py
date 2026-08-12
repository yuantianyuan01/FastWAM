from typing import Tuple, Any, Dict, Union, Callable, Iterable, Iterator
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import itertools
from multiprocessing import Pool
from functools import partial
import glob
import h5py
import cv2
import tensorflow as tf
from tensorflow_datasets.core import download
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import utils
from tensorflow_datasets.core import writer as writer_lib
from tensorflow_datasets.core import example_serializer
from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import file_adapters
from scipy.spatial.transform import Rotation as R

Key = Union[str, int]
# The nested example dict passed to `features.encode_example`
Example = Dict[str, Any]
KeyExample = Tuple[Key, Example]


class DemoBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 100             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1000

    def __init__(self, *args, **kwargs):
        self.path = "**/*.hdf5"
        super().__init__(*args, **kwargs)
    
    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            'train': glob.glob('**/*.hdf5', recursive=True),
        }
    
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
            try:
                with h5py.File(episode_path, 'a') as f:
                    data = f["data"]
                    timestamps = data.keys()
                    for i, ts in enumerate(timestamps):
                        new_data = data[ts]
                        # load episode config
                        episode_config_bytes = np.asarray(new_data["meta_info"]["episode_config"]).astype('S')
                        episode_config = episode_config_bytes.item().decode('utf-8')
                        episode_config = json.loads(episode_config)
                        if episode_config.get("robot") is not None:
                            robot_frame_pos = np.array(episode_config["robot"]["position"])
                        else:
                            robot_frame_pos = np.array([0, -0.4, 0.78])
                        
                        trajectory = np.asarray(new_data['trajectory'])
                        images = np.asarray(new_data["observation"]["rgb"]) # shape (t, n_image, h, w, c)
                        t, n_image, h, w, c = images.shape
                        # resize 
                        # new_images = np.zeros((t, n_image, 256, 256, c), dtype=images.dtype)
                        # for i in range(t):
                        #     for j in range(n_image):
                        #         new_images[i, j] = cv2.resize(images[i, j], (256, 256))
                        # qpos = np.asarray(new_data["observation"]["qpos"])
                        ee_state = np.asarray(new_data["observation"]["ee_state"])
                        # process ee_state
                        ee_pos, ee_quat, gripper = ee_state[:, :3], ee_state[:, 3:7], ee_state[:, 7]
                        ee_euler = np.array([quat2euler(q) for q in ee_quat])
                        # transform ee_state to robot frame
                        ee_pos -= robot_frame_pos
                        ee_state = np.concatenate([ee_pos, ee_euler, gripper.reshape(-1, 1)], axis=1).astype(np.float32)
                        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                        episode = []
                        episode_length = trajectory.shape[0]
                        assert images.shape[0] == episode_length == ee_state.shape[0]
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
                                    'ee_state':ee_state[i]
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
            except Exception as e:
                print(f"[WARN] Error reading {episode_path}: {e}, skipping this file.")
                return None
        # create list of all examples
        episode_paths = path

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            if sample.endswith("hdf5"):
                try:
                    with h5py.File(sample, 'a') as f:
                        data = f["data"]
                except Exception as e:
                    print(f"Error reading {sample}: {e}, pass")
                    continue
            result = _parse_example(sample)
            if result is not None:
                yield result
            

    def _download_and_prepare(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
            self,
            dl_manager: download.DownloadManager,
            download_config: download.DownloadConfig,
    ) -> None:
        """Generate all splits and returns the computed split infos."""
        split_builder = ParallelSplitBuilder(
            split_dict=self.info.splits,
            features=self.info.features,
            dataset_size=self.info.dataset_size,
            max_examples_per_split=download_config.max_examples_per_split,
            beam_options=download_config.beam_options,
            beam_runner=download_config.beam_runner,
            file_format=self.info.file_format,
            shard_config=download_config.get_shard_config(),
            split_paths=self._split_paths(),
            parse_function=self._generate_examples,
            n_workers=self.N_WORKERS,
            max_paths_in_memory=self.MAX_PATHS_IN_MEMORY,
        )
        split_generators = self._split_generators(dl_manager)
        split_generators = split_builder.normalize_legacy_split_generators(
            split_generators=split_generators,
            generator_fn=self._generate_examples,
            is_beam=False,
        )
        dataset_builder._check_split_names(split_generators.keys())

        # Start generating data for all splits
        path_suffix = file_adapters.ADAPTER_FOR_FORMAT[
            self.info.file_format
        ].FILE_SUFFIX

        split_info_futures = []
        for split_name, generator in utils.tqdm(
                split_generators.items(),
                desc="Generating splits...",
                unit=" splits",
                leave=False,
        ):
            filename_template = naming.ShardedFileTemplate(
                split=split_name,
                dataset_name=self.name,
                data_dir=self.data_path,
                filetype_suffix=path_suffix,
            )
            future = split_builder.submit_split_generation(
                split_name=split_name,
                generator=generator,
                filename_template=filename_template,
                disable_shuffling=self.info.disable_shuffling,
            )
            split_info_futures.append(future)

        # Finalize the splits (after apache beam completed, if it was used)
        split_infos = [future.result() for future in split_info_futures]

        # Update the info object with the splits.
        split_dict = splits_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)

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
                        'ee_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot ee state, [6 joint, 1 gripper]',
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

class _SplitInfoFuture:
    """Future containing the `tfds.core.SplitInfo` result."""

    def __init__(self, callback: Callable[[], splits_lib.SplitInfo]):
        self._callback = callback

    def result(self) -> splits_lib.SplitInfo:
        return self._callback()


def parse_examples_from_generator(paths, fcn, split_name, total_num_examples, features, serializer):
    generator = fcn(paths)
    outputs = []
    for sample in utils.tqdm(
            generator,
            desc=f'Generating {split_name} examples...',
            unit=' examples',
            total=total_num_examples,
            leave=False,
            mininterval=1.0,
    ):
        if sample is None: continue
        key, example = sample
        try:
            example = features.encode_example(example)
        except Exception as e:  # pylint: disable=broad-except
            utils.reraise(e, prefix=f'Failed to encode example:\n{example}\n')
        outputs.append((key, serializer.serialize_example(example)))
    return outputs


class ParallelSplitBuilder(split_builder_lib.SplitBuilder):
    def __init__(self, *args, split_paths, parse_function, n_workers, max_paths_in_memory, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_paths = split_paths
        self._parse_function = parse_function
        self._n_workers = n_workers
        self._max_paths_in_memory = max_paths_in_memory

    def _build_from_generator(
            self,
            split_name: str,
            generator: Iterable[KeyExample],
            filename_template: naming.ShardedFileTemplate,
            disable_shuffling: bool,
    ) -> _SplitInfoFuture:
        """Split generator for example generators.

        Args:
          split_name: str,
          generator: Iterable[KeyExample],
          filename_template: Template to format the filename for a shard.
          disable_shuffling: Specifies whether to shuffle the examples,

        Returns:
          future: The future containing the `tfds.core.SplitInfo`.
        """
        total_num_examples = None
        serialized_info = self._features.get_serialized_info()
        writer = writer_lib.Writer(
            serializer=example_serializer.ExampleSerializer(serialized_info),
            filename_template=filename_template,
            hash_salt=split_name,
            disable_shuffling=disable_shuffling,
            file_format=self._file_format,
            shard_config=self._shard_config,
        )

        del generator  # use parallel generators instead
        paths = self._split_paths[split_name]
        path_lists = chunk_max(paths, self._n_workers, self._max_paths_in_memory)  # generate N file lists
        print(f"Generating with {self._n_workers} workers!")
        pool = Pool(processes=self._n_workers)
        for i, paths in enumerate(path_lists):
            print(f"Processing chunk {i + 1} of {len(path_lists)}.")
            results = pool.map(
                partial(
                    parse_examples_from_generator,
                    fcn=self._parse_function,
                    split_name=split_name,
                    total_num_examples=total_num_examples,
                    serializer=writer._serializer,
                    features=self._features
                ),
                paths
            )
            # write results to shuffler --> this will automatically offload to disk if necessary
            print("Writing conversion results...")
            for result in itertools.chain(*results):
                key, serialized_example = result
                writer._shuffler.add(key, serialized_example)
                writer._num_examples += 1
        pool.close()

        print("Finishing split conversion...")
        shard_lengths, total_size = writer.finalize()

        split_info = splits_lib.SplitInfo(
            name=split_name,
            shard_lengths=shard_lengths,
            num_bytes=total_size,
            filename_template=filename_template,
        )
        return _SplitInfoFuture(lambda: split_info)


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL, t)) for t in zip(*DL.values())]

def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si:si + (d + 1 if i < r else d)]

def chunk_max(l, n, max_chunk_sum):
    out = []
    for _ in range(int(np.ceil(len(l) / max_chunk_sum))):
        out.append(list(chunks(l[:max_chunk_sum], n)))
        l = l[max_chunk_sum:]
    return out

def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles