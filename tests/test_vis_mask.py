# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Smoketest our "mask" drawing utility, e.g. for drawing Mask RCNN masks on an image."""


import cv2
import matplotlib.pyplot as plt
import numpy as np

from argoverse.visualization.vis_mask import decode_segment_to_mask, vis_mask, vis_one_image, vis_one_image_opencv


def test_vis_mask() -> None:
    # Ordered Z first for easy reading
    img = np.array(
        [
            [[0, 0, 0], [0, 10, 10], [0, 10, 10]],
            [[0, 0, 0], [0, 20, 20], [0, 20, 20]],
            [[0, 0, 0], [0, 40, 40], [0, 40, 40]],
        ]
    )

    img = img.swapaxes(0, 2)

    mask = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])

    masked_image = vis_mask(img, mask, color=10.0, alpha=0.2)

    masked_image = masked_image.swapaxes(0, 2)

    # Ordered Z first for easy reading
    expected_img = np.array(
        [
            [[0, 0, 0], [0, 10, 10], [0, 10, 10]],
            [[0, 0, 0], [0, 18, 18], [0, 18, 18]],
            [[0, 0, 0], [0, 34, 34], [0, 34, 34]],
        ]
    )

    assert (expected_img == masked_image).all()


def test_decode_segment_to_mask() -> None:
    mask = decode_segment_to_mask((2, 1, 3, 3), np.zeros((3, 3, 3)))

    expected_mask = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]])

    assert (mask == expected_mask).all()


def mask_vis_unit_test() -> None:
    unit_test_dir = "test_data/vis_mask/0d2ee2db-4061-36b2-a330-8301bdce3fe8/00035"
    img_fpath = f"{unit_test_dir}/image_raw_ring_side_left_000000035.jpg"
    img = cv2.imread(img_fpath)

    np.load(f"{unit_test_dir}/00035_image_raw_ring_side_left_000000035_detection_scores.npy")
    np.load(f"{unit_test_dir}/00035_image_raw_ring_side_left_000000035_detection_labels.npy")
    det_bboxes = np.load(f"{unit_test_dir}/00035_image_raw_ring_side_left_000000035_detection_bbox.npy")

    vis_one_image(
        img[:, :, ::-1],
        "00035_side_left_dets",
        "mask_vis",
        det_bboxes,
        segms=det_bboxes,
        box_alpha=0.8,
        show_class=True,
    )

    img = vis_one_image_opencv(img, det_bboxes, segms=det_bboxes, show_box=False, show_class=True)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    mask_vis_unit_test()
