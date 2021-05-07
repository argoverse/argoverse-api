# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>

import glob
from typing import Any, List, Mapping

import cv2
import numpy as np

from argoverse.utils.json_utils import read_json_file

DISPARITY_NORMALIZATION: float = 2.0 ** 8  # 256.0


class ArgoverseStereoDataLoader:
    """Stereo data loader for retrieving log data, given a path to the dataset."""

    def __init__(self, data_dir: str, split_name: str) -> None:
        """
        Args:
            data_dir: Path to the Argoverse stereo data.
            split_name: Data split, one of train, val, or test.
        """
        self.data_dir = data_dir
        self.split_name = split_name

    def get_log_calibration_data(self, log_id: str) -> Mapping[str, Any]:
        """Get calibration data.

        Args:
            log_id: Unique ID of vehicle log.

        Returns:
            log_calib_data: calibration dictionary.
        """
        calib_fpath = (
            f"{self.data_dir}/rectified_stereo_images_v1.1/{self.split_name}/{log_id}/"
            f"vehicle_calibration_stereo_info.json"
        )

        log_calib_data = read_json_file(calib_fpath)
        assert isinstance(log_calib_data, dict)
        return log_calib_data

    def get_ordered_log_stereo_image_fpaths(self, log_id: str, camera_name: str) -> List[str]:
        """Get list of paths to chronologically ordered rectified stereo images in this log.

        Args:
            log_id: Unique ID of vehicle log.
            camera: camera based on camera_stats.RECTIFIED_STEREO_CAMERA_LIST.

        Returns:
            stereo_img_fpaths: List of paths to chronologically ordered rectified stereo images in this log.
        """
        stereo_img_fpaths = sorted(
            glob.glob(f"{self.data_dir}/rectified_stereo_images_v1.1/{self.split_name}/{log_id}/{camera_name}/*.jpg")
        )

        return stereo_img_fpaths

    def get_ordered_log_disparity_map_fpaths(self, log_id: str, disparity_name: str) -> List[str]:
        """Get list of paths to chronologically ordered disparity maps in this log.

        Args:
            log_id: Unique ID of vehicle log.
            disparity_name: disparity map name, one of:

                ["stereo_front_left_rect_disparity",
                 "stereo_front_left_rect_objects_disparity"].

        Returns:
            disparity_map_fpaths: List of paths to chronologically ordered disparity maps in this log.
        """
        disparity_map_fpaths = sorted(
            glob.glob(f"{self.data_dir}/disparity_maps_v1.1/{self.split_name}/{log_id}/{disparity_name}/*.png")
        )

        return disparity_map_fpaths

    def get_rectified_stereo_image(self, stereo_img_path: str) -> np.ndarray:
        """Get the rectified stereo image.

        Args:
            stereo_img_path: Path to the rectified stereo image file.

        Returns:
            Array of shape (M, N, 3) representing an RGB rectified stereo image.
        """
        return cv2.cvtColor(cv2.imread(stereo_img_path), cv2.COLOR_BGR2RGB)

    def get_disparity_map(self, disparity_map_path: str) -> np.ndarray:
        """Get the disparity map.

        The disparity maps are saved as uint16 PNG images. A zero-value ("0") indicates that no ground truth exists
        for that pixel. The true disparity for a pixel can be recovered by first converting the uint16 value to float
        and then dividing it by 256.

        Args:
            disparity_map_path: Path to the disparity map file.

        Returns:
            disparity_map: Array of shape (M, N) representing a float32 single-channel disparity map.
        """
        disparity_map = cv2.imread(disparity_map_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return np.float32(disparity_map) / DISPARITY_NORMALIZATION
