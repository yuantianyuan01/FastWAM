"""
Import this module to configure the renderer flags for the current platform.
"""

import os
import sys


def patch_renderer_flags():
    if sys.platform.startswith("linux"):
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "PYOPENGL_PLATFORM" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
