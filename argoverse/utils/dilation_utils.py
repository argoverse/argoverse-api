# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Utility functions for dilation."""

import cv2
import numpy as np


def dilate_by_l2(img: np.ndarray, dilation_thresh: float = 5.0) -> np.ndarray:
    """Dilate a mask using the L2 distance from a zero pixel.

    OpenCV's distance transform calculates the DISTANCE TO THE CLOSEST ZERO PIXEL for each
    pixel of the source image. Although the distance type could be L1, L2, etc, we use L2.

    We specify the "maskSize", which represents the size of the distance transform mask. It can
    be 3, 5, or CV_DIST_MASK_PRECISE (the latter option is only supported by the first function).

    For us, foreground values are 1 and background values are 0.

    Args:
        img: Array of shape (M, N) representing an 8-bit single-channel (binary) source image
        dilation_thresh: Threshold for how far away a zero pixel can be

    Returns:
        An image with the same size with the dilated mask
    """
    mask_diff = np.ones_like(img).astype(np.uint8) - img
    distance_mask = cv2.distanceTransform(mask_diff, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
    distance_mask = distance_mask.astype(np.float32)
    return (distance_mask <= dilation_thresh).astype(np.uint8)
