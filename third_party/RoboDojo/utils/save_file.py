import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


def format_video_saved_message(
    path: str,
    n_frames: int,
    width: int,
    height: int,
    fps: float,
) -> str:
    return (
        f"🎬 Video is saved to `{path}`, containing "
        f"\033[94m{n_frames}\033[0m frames at {width}×{height} "
        f"resolution and {fps} FPS."
    )


class VideoStreamWriter:
    """Stream frames to an mp4 through a persistent ffmpeg pipe.

    Keeps a single ffmpeg process alive and flushes each frame to disk as soon
    as it arrives, so an episode's frames are never all held in memory at once.
    Use ``append`` per frame, then ``close`` to finalize a valid file or
    ``abort`` to discard the partial output.
    """

    def __init__(
        self,
        out_path: str,
        height: int,
        width: int,
        channels: int,
        fps: float = 30.0,
        is_rgb: bool = True,
    ) -> None:
        if channels == 3:
            pixel_format = "rgb24" if is_rgb else "bgr24"
        elif channels == 4:
            pixel_format = "rgba"
        else:
            raise ValueError(f"Unsupported channel count for video: {channels}")
        self.out_path = out_path
        self.height = height
        self.width = width
        self.channels = channels
        self.fps = fps
        self.n_frames = 0
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                pixel_format,
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                out_path,
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, frame: np.ndarray) -> None:
        if self.proc is None:
            raise RuntimeError("Cannot append to a closed VideoStreamWriter.")
        if frame.ndim != 3 or frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame shape {tuple(frame.shape)} does not match writer ({self.height}x{self.width}x{self.channels})."
            )
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        self.proc.stdin.write(frame.tobytes())
        self.n_frames += 1

    def close(self, *, announce: bool = True) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise OSError(f"ffmpeg failed while finalizing `{self.out_path}`.")
        finally:
            self.proc = None
        if announce:
            print(
                format_video_saved_message(
                    self.out_path,
                    self.n_frames,
                    self.width,
                    self.height,
                    self.fps,
                )
            )

    def abort(self) -> None:
        """Kill the ffmpeg process and remove the partial output file."""
        if self.proc is not None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.kill()
                self.proc.wait()
            except Exception:
                pass
            self.proc = None
        try:
            if os.path.exists(self.out_path):
                os.remove(self.out_path)
        except Exception:
            pass


def save_json(
    data: Any,
    path: str | os.PathLike,
    overwrite: bool = True,
    make_dirs: bool = True,
    sort_keys: bool = False,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    p = Path(path)

    if make_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists and overwrite=False")

    tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(
                data,
                f,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                indent=indent,
            )
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            raise
