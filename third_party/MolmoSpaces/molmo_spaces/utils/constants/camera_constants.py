"""Camera hardware constants for fisheye warping and image processing."""

# GoPro Hero 11 Black specifications (wide FOV mode)
GOPRO_VERTICAL_FOV = 139.0  # degrees
GOPRO_CAMERA_WIDTH = 640
GOPRO_CAMERA_HEIGHT = 480

# Model output dimensions (4:3 aspect ratio)
MODEL_43_WIDTH = 320
MODEL_43_HEIGHT = 240

# Fisheye distortion parameters
# Default parameters calibrated for GoPro fisheye distortion
DEFAULT_DISTORTION_PARAMETERS = {
    "k1": 0.051,
    "k2": 0.144,
    "k3": 0.015,
    "k4": -0.018,
}

NULL_DISTORTION_PARAMETERS = {
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "k4": 0.0,
}

# Default crop percentage to remove edge distortion
DEFAULT_CROP_PERCENT = 0.30
