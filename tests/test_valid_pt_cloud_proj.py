#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tom-bu
"""

import numpy as np

from argoverse.utils.calibration import CameraConfig, determine_valid_cam_coords


def test_valid_point_cloud_projections() -> None:
    """Test to ensure determine_valid_cam_coords() returns valid points that are less than the image width and height."""
    # create a fake camera config
    camera_config = CameraConfig(
        extrinsic=np.eye(4), intrinsic=np.eye(3, 4), img_width=1920, img_height=1200, distortion_coeffs=np.zeros(3)
    )

    # create a test case of projected lidar points in the image space
    uv = np.array(
        [
            [0, 0],
            [camera_config.img_width - 0.3, 0],
            [0, camera_config.img_height - 0.3],
            [camera_config.img_width - 0.3, camera_config.img_height - 0.3],
        ]
    )
    uv_cam = np.ones((4, uv.shape[0]))

    # use the determine_valid_cam_coords() method
    valid_pts_bool = determine_valid_cam_coords(uv, uv_cam, camera_config)

    # as done in draw_ground_pts_in_image() in ground_visualization.py
    uv = np.round(uv[valid_pts_bool]).astype(np.int32)

    assert np.all(uv[:, 0] < camera_config.img_width) and np.all(uv[:, 1] < camera_config.img_height)
