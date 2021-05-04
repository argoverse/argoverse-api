# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Ground visualization utilities."""

import copy
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import imageio
import numpy as np
from colour import Color

from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.calibration import (
    point_cloud_to_homogeneous,
    project_lidar_to_img,
    project_lidar_to_img_motion_compensated,
)
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.cv2_plotting_utils import draw_point_cloud_in_img_cv2
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.se3 import SE3

__all__ = ["draw_ground_pts_in_image"]

NUM_RANGE_BINS = 50


def draw_ground_pts_in_image(
    sdb: SynchronizationDB,
    lidar_points: np.ndarray,
    city_SE3_egovehicle: SE3,
    dataset_map: ArgoverseMap,
    log_id: str,
    lidar_timestamp: int,
    city_name: str,
    dataset_dir: str,
    experiment_prefix: str,
    plot_ground: bool = True,
    motion_compensate: bool = False,
    camera: Optional[str] = None,
) -> Union[None, np.ndarray]:
    """Write an image to disk with rendered ground points for every camera.

    Args:
        sdb: instance of SynchronizationDB
        lidar_points: Numpy array of shape (N,3) in egovehicle frame
        city_SE3_egovehicle: SE3 instance which takes a point in egovehicle frame and brings it into city frame
        dataset_map: Map dataset instance
        log_id: ID of the log
        city_name: A city's name (e.g. 'MIA' or 'PIT')
        motion_compensate: Whether to bring lidar points from world frame @ lidar timestamp, to world frame @ camera
                           timestamp
        camera: camera name, if specified will return image of that specific camera, if None, will save all camera to
                disk and return None

    """
    # put into city coords, then prune away ground and non-RoI points
    lidar_points = city_SE3_egovehicle.transform_point_cloud(lidar_points)
    lidar_points = dataset_map.remove_non_driveable_area_points(lidar_points, city_name)
    _, not_ground_logicals = dataset_map.remove_ground_surface(
        copy.deepcopy(lidar_points), city_name, return_logicals=True
    )
    lidar_points = lidar_points[np.logical_not(not_ground_logicals) if plot_ground else not_ground_logicals]

    # put back into ego-vehicle coords
    lidar_points = city_SE3_egovehicle.inverse_transform_point_cloud(lidar_points)

    calib_fpath = f"{dataset_dir}/{log_id}/vehicle_calibration_info.json"
    calib_data = read_json_file(calib_fpath)

    # repeat green to red colormap every 50 m.
    colors_arr = np.array(
        [[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), NUM_RANGE_BINS)]
    ).squeeze()
    np.fliplr(colors_arr)

    for cam_idx, camera_name in enumerate(RING_CAMERA_LIST + STEREO_CAMERA_LIST):
        im_dir = f"{dataset_dir}/{log_id}/{camera_name}"

        # load images, e.g. 'image_raw_ring_front_center_000000486.jpg'
        cam_timestamp = sdb.get_closest_cam_channel_timestamp(lidar_timestamp, camera_name, log_id)
        if cam_timestamp is None:
            continue

        im_fname = f"{camera_name}_{cam_timestamp}.jpg"
        im_fpath = f"{im_dir}/{im_fname}"

        # Swap channel order as OpenCV expects it -- BGR not RGB
        # must make a copy to make memory contiguous
        img = imageio.imread(im_fpath)[:, :, ::-1].copy()
        points_h = point_cloud_to_homogeneous(copy.deepcopy(lidar_points)).T

        if motion_compensate:
            uv, uv_cam, valid_pts_bool = project_lidar_to_img_motion_compensated(
                points_h,  # these are recorded at lidar_time
                copy.deepcopy(calib_data),
                camera_name,
                cam_timestamp,
                lidar_timestamp,
                dataset_dir,
                log_id,
                False,
            )
        else:
            uv, uv_cam, valid_pts_bool = project_lidar_to_img(points_h, copy.deepcopy(calib_data), camera_name, False)

        if valid_pts_bool is None or uv is None or uv_cam is None:
            continue

        if valid_pts_bool.sum() == 0:
            continue

        uv = np.round(uv[valid_pts_bool]).astype(np.int32)
        uv_cam = uv_cam.T[valid_pts_bool]
        pt_ranges = np.linalg.norm(uv_cam[:, :3], axis=1)
        rgb_bins = np.round(pt_ranges).astype(np.int32)
        # account for moving past 100 meters, loop around again
        rgb_bins = rgb_bins % NUM_RANGE_BINS
        uv_colors = (255 * colors_arr[rgb_bins]).astype(np.int32)

        img = draw_point_cloud_in_img_cv2(img, uv, np.fliplr(uv_colors))

        if not Path(f"{experiment_prefix}_ground_viz/{log_id}/{camera_name}").exists():
            os.makedirs(f"{experiment_prefix}_ground_viz/{log_id}/{camera_name}")

        save_dir = f"{experiment_prefix}_ground_viz/{log_id}/{camera_name}"
        cv2.imwrite(f"{save_dir}/{camera_name}_{lidar_timestamp}.jpg", img)
        if camera == camera_name:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None
