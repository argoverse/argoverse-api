#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import logging
from typing import Optional, Tuple

from argoverse.sensor_dataset_config import ArgoverseConfig

"""
Since we use images of different sizes (ring vs. stereo), we cannot
fix the image size throughout -- must be adaptive.
"""

RING_CAMERA_LIST = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
]

STEREO_CAMERA_LIST = ["stereo_front_left", "stereo_front_right"]

RECTIFIED_STEREO_CAMERA_LIST = ["stereo_front_left_rect", "stereo_front_right_rect"]

CAMERA_LIST = RING_CAMERA_LIST + STEREO_CAMERA_LIST

logger = logging.getLogger(__name__)


def get_image_dims_for_camera(camera_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Get image dimensions for camera.
    Args:
        camera_name: Camera name.

    Returns:
        Tuple of [img_width, image_height] in pixels
    """
    if ArgoverseConfig.camera_sensors.has_camera(camera_name):
        camera_sensor_config = getattr(ArgoverseConfig.camera_sensors, camera_name)
        img_width = camera_sensor_config.img_width
        img_height = camera_sensor_config.img_height
    else:
        logger.error(f"{camera_name} not recognized")
        return None, None
    return img_width, img_height
