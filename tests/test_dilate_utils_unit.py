# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Unit tests for dilation_utils.py."""

import numpy as np

from argoverse.utils.dilation_utils import dilate_by_l2


def test_dilate_by_l2_no_change() -> None:
    """Test dilate_by_l2() and expect no change.

    Do not modify the following 8-bit image (0 px dilation)
    0 0 0 0 0
    0 x x x 0
    0 x x x 0
    0 x x x 0
    0 0 0 0 0
    """
    img = np.zeros((5, 5), dtype=np.uint8)
    img[1:4, 1:4] = 1
    dilated_img = dilate_by_l2(img.copy(), dilation_thresh=0.0)
    assert np.allclose(img, dilated_img)


def test_dilate_by_l2_2x2square_1px() -> None:
    """Test dilate_by_l2() with a 1 pixel dilation.

    Dilate the following 8-bit image w/ 1 px border

    0 0 0 0 0 0    0 0 0 0 0 0
    0 0 0 0 0 0    0 0 x x 0 0
    0 0 x x 0 0 -> 0 x x x x 0
    0 0 x x 0 0    0 x x x x 0
    0 0 0 0 0 0    0 0 x x 0 0
    0 0 0 0 0 0    0 0 0 0 0 0
    """
    img = np.zeros((6, 6), dtype=np.uint8)
    img[2:4, 2:4] = 1
    dilated_img = dilate_by_l2(img, dilation_thresh=1.0)
    print(dilated_img)
    # ground truth
    dilated_img_gt = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.allclose(dilated_img, dilated_img_gt)


def test_dilate_by_l2_horizline_1point42px() -> None:
    """Test dilate_by_l2() with a non-integer dilation that should get cut off.

    Dilate the following 8-bit image w/ 1.42 px border,
    allowing us to access diagonal neighbors:

    0 0 0 0 0    0 0 0 0 0
    0 0 0 0 0    0 x x x x
    0 0 x x 0 -> 0 x x x x
    0 0 0 0 0    0 x x x x
    0 0 0 0 0    0 0 0 0 0
    """
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2:4] = 1
    dilated_img = dilate_by_l2(img, dilation_thresh=1.42)
    # ground truth
    dilated_img_gt = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=np.uint8
    )
    assert np.allclose(dilated_img, dilated_img_gt)
