#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import logging
from typing import List, Optional, Tuple

"""
Since we use images of different sizes (ring vs. stereo), we cannot
fix the image size throughout -- must be adaptive.
"""
STEREO_IMG_WIDTH = 2464
STEREO_IMG_HEIGHT = 2056

RING_IMG_WIDTH = 1920
RING_IMG_HEIGHT = 1200

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

CAMERA_LIST = RING_CAMERA_LIST + STEREO_CAMERA_LIST

logger = logging.getLogger(__name__)


def get_image_dims_for_camera(camera_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Get image dimensions for camera.
    Args:
        camera_name: Camera name.

    Returns:
        Tuple of [img_width, image_height] in pixels
    """
    if camera_name in RING_CAMERA_LIST:
        img_width = RING_IMG_WIDTH
        img_height = RING_IMG_HEIGHT
    elif camera_name in STEREO_CAMERA_LIST:
        img_width = STEREO_IMG_WIDTH
        img_height = STEREO_IMG_HEIGHT
    else:
        logger.error(f"{camera_name} not recognized")
        return None, None
    return img_width, img_height
