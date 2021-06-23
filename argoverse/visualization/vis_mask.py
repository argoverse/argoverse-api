# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
# <Modifications copyright (C) 2019, Argo AI, LLC>
"""
This tool is loosely based off of Facebook's Mask R-CNN visualization tool.
https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
"""

import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from argoverse.visualization.colormap import colormap

plt.rcParams["pdf.fonttype"] = 42  # For editing in Adobe Illustrator

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

Segment = Tuple[float, float, float, float]


def vis_mask(image: np.ndarray, mask: np.ndarray, color: Union[float, np.ndarray], alpha: float = 0.4) -> np.ndarray:
    """Visualize a single binary mask by blending a colored mask with image.

    Args:
        image: The input image (either RGB or BGR) w/ values in the [0,255] range
        mask: The mask to visualize. Integer array, with values in [0,1]
            representing mask region
        color: The color for the mask, either single float or length 3 array
            of integers in [0,255] representing RGB or BGR values
        alpha: The alpha level for the mask. Represents blending coefficient
            (higher alpha shows more of mask, lower alpha preserves original image)

    Returns:
        The modified 3-color image. Represents a blended image
            of original RGB image and specified colors in mask region.
    """

    image = image.astype(np.float32)
    idx = np.nonzero(mask)
    image[idx[0], idx[1], :] *= 1.0 - alpha
    image[idx[0], idx[1], :] += alpha * color

    return image.astype(np.uint8)


def vis_class(
    image: np.ndarray,
    pos: Tuple[float, float],
    class_str: str,
    font_scale: float = 50.0,
) -> np.ndarray:
    """Visualizes a class.

    Args:
        image: The image
        pos: The position for the text
        class_str: The name of the class
        font_scale: Text size

    Returns:
        The modified image
    """
    image = image.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])

    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)

    # Show text.
    txt_tl = x0, y0 + int(0.3 * txt_h)
    cv2.putText(image, txt, txt_tl, font, font_scale, _WHITE, lineType=cv2.LINE_AA)
    return image


def vis_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], thickness: int = 1) -> np.ndarray:
    """Visualize a bounding box.
    Args:
        image: The input image
        bbox: Bounding box
        thickness: Line thickness

    Returns:
        The modified image
    """

    image = image.astype(np.uint8)
    x0, y0, w, h = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(image, (x0, y0), (x1, y1), _GREEN, thickness=thickness)
    return image


def decode_segment_to_mask(segm: Segment, image: np.ndarray) -> np.ndarray:
    """Create a mask from a segment

    Args:
        segm: The segment
        image: The associated image

    Returns:
        A mask built from the given segment and image
    """
    xmin, ymin, xmax, ymax = segm
    mask = np.zeros((image.shape[0], image.shape[1]))

    mask[int(ymin) : int(ymax), int(xmin) : int(xmax)] = 1
    return mask


def vis_one_image_opencv(
    image: np.ndarray,
    boxes: np.ndarray,
    segms: Optional[Sequence[Segment]] = None,
    show_box: bool = False,
    show_class: bool = True,
) -> np.ndarray:
    """Constructs a numpy array with the detections visualized.

    Args:
        image: The image data
        boxes: The box data
        segms: Segmentations
        show_box: Whether to show the boxes
        show_class: Whether to show the object classes

    Return:
        The newly constructed image
    """

    if boxes is None or boxes.shape[0] == 0:
        return image

    if segms:
        color_list = colormap()
        mask_color_id = 0

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        boxes[i, -1]

        # show box (off by default)
        if show_box:
            image = vis_bbox(image, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # show class (off by default)
        if show_class:
            class_str = "hello"
            image = vis_class(image, (bbox[0], bbox[1] - 2), class_str)

        # show mask
        if segms and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            mask = decode_segment_to_mask(segms[i], image)
            image = vis_mask(image, mask, color_mask)

    return image


def vis_one_image(
    image: np.ndarray,
    image_name: str,
    output_dir: str,
    boxes: np.ndarray,
    segms: Optional[Sequence[Segment]] = None,
    dpi: int = 200,
    box_alpha: float = 0.0,
    show_class: bool = True,
    extension: str = "pdf",
) -> None:
    """Visual debugging of detections.

    Args:
        image: The image data
        image_name: The name of the image
        output_dir: Directory to output to
        boxes: Boxes
        segms: Segmentations
        dpi: DPI
        box_alpha: Alpha channel of the boxes
        show_class: Whether to show object classes
        extension: Extension of the output file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if boxes is None or boxes.shape[0] == 0:
        return

    color_list = colormap(rgb=True) / 255
    plt.get_cmap("rainbow")

    fig = plt.figure(frameon=False)
    fig.set_size_inches(image.shape[1] / dpi, image.shape[0] / dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    fig.add_axes(ax)
    ax.imshow(image)

    sorted_inds: Union[List[Any], np.ndarray]
    if boxes is None:
        sorted_inds = []  # avoid crash when 'boxes' is None
    else:
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="g",
                linewidth=0.5,
                alpha=box_alpha,
            )
        )

        if show_class:
            ax.text(
                bbox[0],
                bbox[1] - 2,
                "WHERE IS THE TEXT car",
                fontsize=30,
                family="serif",
                bbox=dict(facecolor="g", alpha=0.4, pad=0, edgecolor="none"),
                color="white",
            )

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(image.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = 0.4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]

            e = decode_segment_to_mask(segms[i], image)
            e = e.astype(np.uint8)

            _, contours, hier = cv2.findContours(e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                polygon = Polygon(
                    contour.reshape((-1, 2)),
                    fill=True,
                    facecolor=color_mask,
                    edgecolor="w",
                    linewidth=1.2,
                    alpha=0.5,
                )
                ax.add_patch(polygon)

    output_name = os.path.basename(image_name) + "." + extension
    fig.savefig(os.path.join(output_dir, "{}".format(output_name)), dpi=dpi)
    plt.close("all")
