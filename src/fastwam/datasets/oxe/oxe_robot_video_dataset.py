import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
# from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from fastwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)

from oxe_utils.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK
from oxe_utils.data_utils import tree_map
from oxe_utils.rlds import make_interleaved_dataset, make_single_dataset
from oxe_utils.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class OXERobotVideoDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset_dirs: str,
        data_mix: str,
        num_frames: int,
        frame_size: tuple[int, int],
        video_size: tuple[int, int],
        shuffle_buffer_size: int,
        image_aug: bool = False,
        text_embedding_cache_dir=None,
        no_prefix_padding: bool = False,
        context_len=128,
        is_training_set=False,
        action_video_freq_ratio: int = 1,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        seed: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        # Per-rank tf.data seeding. The RLDS pipeline in oxe_utils calls
        # `.shuffle(...)` and `sample_from_datasets(...)` without explicit
        # op-level seeds, so they inherit the global tf seed. Setting it
        # before the pipeline is constructed makes each rank see a distinct
        # but reproducible random stream (base_seed + rank).
        resolved_seed = 42 if seed is None else int(seed)
        if rank is None:
            rank_env = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            try:
                resolved_rank = int(rank_env)
            except ValueError:
                resolved_rank = 0
        else:
            resolved_rank = int(rank)
        import tensorflow as tf
        tf.random.set_seed(resolved_seed + resolved_rank)
        self._tf_seed = resolved_seed + resolved_rank
        self._rank = resolved_rank

        self.dataset_dirs, self.data_mix = dataset_dirs, data_mix
        
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]
        load_camera_views = ("primary", "secondary")

        # Upstream obs_transforms expects `resize_size` as a dict
        # {camera_name: (H, W)}. Convert to plain tuples so that this works
        # regardless of whether `frame_size` came in as tuple/list/OmegaConf
        # ListConfig (Hydra instantiation yields ListConfig).
        frame_size = tuple(int(x) for x in frame_size)
        resize_size_per_camera = {name: frame_size for name in load_camera_views}

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.dataset_dirs,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=num_frames,
                future_action_window_size=0,
                goal_relabeling_strategy="uniform",
                skip_unlabeled=True,
                no_prefix_padding=no_prefix_padding, # if true, this will remove trajectories that are shorter than num_frames, and make sure every frame is valid.
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_size_per_camera,
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs, 
            shuffle_buffer_size=shuffle_buffer_size, 
            sample_weights=weights, 
            balance_weights=True, 
            traj_transform_threads=len(mixture_spec), 
            traj_read_threads=len(mixture_spec), 
            train=is_training_set, 
        )
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )})
        
        self.action_video_freq_ratio = action_video_freq_ratio
        self.frame_size = frame_size
        self.video_size = video_size
        self.concat_multi_camera = concat_multi_camera
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}, 
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}, 
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5}, 
        )
        self.dataset, self.dataset_length, self.dataset_statistics = make_interleaved_dataset(
            **rlds_config, 
        )
        
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.override_instruction = override_instruction

    def transform(self, rlds_traj):
        
        observation_image_keys = ["image_primary", "image_secondary"]
        observation_proprio_key = "proprio"
        language_key = "language_instruction"
        
        if len(observation_image_keys) > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([
                    torch.from_numpy(rlds_traj["observation"][k]) for k in observation_image_keys
                ], dim=-2)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([
                    torch.from_numpy(rlds_traj["observation"][k]) for k in observation_image_keys
                ], dim=-3)
        else:
            video = rlds_traj["observation"][observation_image_keys[0]]
        video = self.normalize_transform(video)
        # video = self.resize_transform(video) # very slow, don't know why
        # video = self.crop_transform(video)
        video_subsample_indices = list(range(0, video.shape[0], self.action_video_freq_ratio))
        video = video[video_subsample_indices]
        
        video = video.permute(3, 0, 1, 2)

        action = rlds_traj["action"]
        
        proprio = rlds_traj["observation"][observation_proprio_key]
        
        instruction = rlds_traj["task"][language_key]
        if isinstance(instruction, (bytes, bytearray)):
            instruction = instruction.decode("utf-8")
        else:
            instruction = str(instruction)
        instruction = instruction.strip()
        if self.override_instruction is not None:
            instruction = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=instruction)

        context, context_mask = self._get_cached_text_context(instruction)

        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        data = {
            "video": video, 
            "action": action, 
            "proprio": proprio, 
            "prompt": instruction, 
            "context": context, 
            "context_mask": context_mask, 
        }
        return data
    
    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __len__(self):
        return self.dataset_length

    def __iter__(self):
        for rlds_traj in self.dataset.as_numpy_iterator():
            yield self.transform(rlds_traj)


if __name__ == "__main__":
    """Quick sanity check: iterate a few samples from the OXE dataset,
    save the videos as mp4, and print action / proprio / instruction.

    Usage:
        python -m fastwam.datasets.oxe.oxe_robot_video_dataset
    """
    import argparse

    import numpy as np
    from PIL import Image

    from fastwam.utils.video_io import save_mp4

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dirs", type=str, default="./data/oxe")
    parser.add_argument("--data_mix", type=str, default="bridge")
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--frame_h", type=int, default=224)
    parser.add_argument("--frame_w", type=int, default=224)
    parser.add_argument("--video_h", type=int, default=224)
    parser.add_argument("--video_w", type=int, default=448)
    parser.add_argument("--shuffle_buffer_size", type=int, default=128)
    parser.add_argument("--action_video_freq_ratio", type=int, default=1)
    parser.add_argument("--concat_multi_camera", type=str, default="horizontal")
    parser.add_argument("--image_aug", action="store_true")
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./runs/debug/oxe_samples")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = OXERobotVideoDataset(
        dataset_dirs=args.dataset_dirs,
        data_mix=args.data_mix,
        num_frames=args.num_frames,
        frame_size=(args.frame_h, args.frame_w),
        video_size=(args.video_h, args.video_w),
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=args.image_aug,
        text_embedding_cache_dir=f"./data/cache/{args.data_mix}/",
        context_len=128,
        is_training_set=True,
        action_video_freq_ratio=args.action_video_freq_ratio,
        concat_multi_camera=args.concat_multi_camera,
        no_prefix_padding=True, 
    )

    iterator = iter(dataset)
    # for i, sample in tqdm(enumerate(dataset)):
    for i in range(args.num_batches):
        sample = next(iterator)
        video = sample["video"]          # [C, T, H, W] in [-1, 1]
        action = sample["action"]
        proprio = sample["proprio"]
        prompt = sample["prompt"]

        print(f"\n===== Sample {i} =====")
        print(f"video   shape={tuple(video.shape)} dtype={video.dtype} "
              f"min={float(video.min()):.3f} max={float(video.max()):.3f}")
        if isinstance(action, torch.Tensor):
            print(f"action  shape={tuple(action.shape)} dtype={action.dtype}")
        else:
            print(f"action  shape={np.asarray(action).shape} type={type(action).__name__}")
        print(f"action  = {np.asarray(action)}")
        if isinstance(proprio, torch.Tensor):
            print(f"proprio shape={tuple(proprio.shape)} dtype={proprio.dtype}")
        else:
            print(f"proprio shape={np.asarray(proprio).shape} type={type(proprio).__name__}")
        print(f"proprio = {np.asarray(proprio)}")
        print(f"prompt  = {prompt}")

        # [C, T, H, W] in [-1, 1]  ->  list of PIL [H, W, C] uint8
        vid = video.detach().cpu().float()
        vid = ((vid + 1.0) / 2.0).clamp(0, 1)
        vid = (vid * 255.0).to(torch.uint8)
        vid = vid.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
        frames = [Image.fromarray(f) for f in vid]

        out_path = os.path.join(args.output_dir, f"sample_{i:03d}.mp4")
        save_mp4(frames, out_path, fps=args.fps)
        print(f"saved video -> {out_path}")

    # Release tf.data pipeline before interpreter shutdown to avoid a
    # harmless TypeError in tensorflow/.../atomic_function.py:__del__,
    # which fires when TF module globals are torn down before its
    # AtomicFunction finalizers run.
    del iterator
    del dataset
    import gc
    gc.collect()