
from typing import List

import numpy as np
from PIL import Image, ImageDraw


def get_mask_from_polygons(polygons: List[np.ndarray], img_h: int, img_w: int) -> np.ndarray:
    """Rasterize multiple polygons onto a single 2d array.

    Args:
        polygons: list of (N,2) numpy float arrays, where N is variable per polygon.
        img_h: height of the image to generate, in pixels
        img_w: width of the image to generate, in pixels

    Returns:
        mask: 2d array with 0/1 values representing a binary segmentation mask
    """
    mask_img = Image.new("L", size=(img_w, img_h), color=0)
    for polygon in polygons:
        polygon = [tuple([x, y]) for (x, y) in polygon]

        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)

    mask = np.array(mask_img)
    return mask
