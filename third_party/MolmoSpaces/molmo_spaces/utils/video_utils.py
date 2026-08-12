"""Copied from video2sim_pipeline/video2sim/utils/video_utils.py"""

import os

import cv2
import ffmpeg  # pip install ffmpeg-python
import numpy as np
import torch
from einops import rearrange


def ffmpeg_save_video(
    frames,
    output_path: str,
    fps: float = 30.0,
    codec: str = "libx264",
    quality: int = 23,  # Lower CRF means higher quality (18-28 is good range)
    pix_fmt="rgb24",  # opencv
):
    """
    Save a video using ffmpeg.

    Args:
        frames: Video frames as numpy array (T, H, W, 3) or torch tensor (T, 3, H, W)
        output_path: Path to save the video file
        fps: Frames per second
        codec: Video codec to use
        quality: CRF value (lower is better quality, 18-28 is reasonable)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert torch tensor to numpy if needed
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
        if frames.shape[1] == 3:  # Convert from TCHW to THWC
            frames = rearrange(frames, "T C H W -> T H W C")

    # Ensure frames are uint8
    if isinstance(frames, list):
        if frames[0].dtype != np.uint8:
            frames = [frame.astype(np.uint8) for frame in frames]
        frames = np.array(frames)
    else:
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)

    assert frames.ndim == 4 and frames.shape[-1] == 3, (
        f"Expected THWC format, got shape {frames.shape}"
    )

    # Set up ffmpeg process
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt=pix_fmt,  # "bgr24", #"rgb24",
            s=f"{frames.shape[2]}x{frames.shape[1]}",
            r=fps,
        )
        .output(output_path, pix_fmt="yuv420p", vcodec=codec, crf=quality)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)  # .run_async(pipe_stdin=True, quiet=True)
    )

    # # Write frames
    for i, frame in enumerate(frames):
        try:
            process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print(f"[FFMPEG ERROR] Broken pipe after writing frame {i}.")
            stderr_output = process.stderr.read().decode() if process.stderr else "No stderr"
            print(f"[FFMPEG STDERR]\n{stderr_output}")

    process.stdin.close()
    process.wait()

    return output_path


def resize_with_padding(image, target_width, target_height, pad_color=(0, 0, 0)):
    """
    Resize an image to fit within the target size while maintaining its original aspect ratio.
    Padding (letterbox) is added to ensure the final image matches the target dimensions.

    Args:
        image (np.array): Input image.
        target_width (int): Desired width.
        target_height (int): Desired height.
        pad_color (tuple): Color for the padding (default is black).

    Returns:
        np.array: Resized image with padding.
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h))

    pad_w = target_width - new_w
    pad_h = target_height - new_h

    # Calculate padding for each side
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded_image
