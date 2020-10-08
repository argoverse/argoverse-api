# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import glob
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat


class SimpleArgoverseTrackingDataLoader:
    """
    Simple abstraction for retrieving log data, given a path to the dataset.
    """

    def __init__(self, data_dir: str, labels_dir: str) -> None:
        """
        Args:
            data_dir: str, representing path to raw Argoverse data
            labels_dir: strrepresenting path to Argoverse data labels
        """
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.sdb = SynchronizationDB(data_dir)

    def get_city_name(self, log_id: str) -> str:
        """
        Args:
            log_id: str

        Returns:
            city_name: str
        """
        city_info_fpath = f"{self.data_dir}/{log_id}/city_info.json"
        city_info = read_json_file(city_info_fpath)
        city_name = city_info["city_name"]
        assert isinstance(city_name, str)
        return city_name

    def get_log_calibration_data(self, log_id: str) -> Mapping[str, Any]:
        """
        Args:
            log_id: str

        Returns:
            log_calib_data: dictionary
        """
        calib_fpath = f"{self.data_dir}/{log_id}/vehicle_calibration_info.json"
        log_calib_data = read_json_file(calib_fpath)
        assert isinstance(log_calib_data, dict)
        return log_calib_data

    def get_city_to_egovehicle_se3(self, log_id: str, timestamp: int) -> Optional[SE3]:
        """
        Args:
            log_id: str, unique ID of vehicle log
            timestamp: int, timestamp of sensor observation, in nanoseconds

        Returns:
            city_to_egovehicle_se3: SE3 transformation to bring egovehicle frame point into city frame.
        """
        pose_fpath = f"{self.data_dir}/{log_id}/poses/city_SE3_egovehicle_{timestamp}.json"
        if not Path(pose_fpath).exists():
            return None
        pose_data = read_json_file(pose_fpath)
        rotation = np.array(pose_data["rotation"])
        translation = np.array(pose_data["translation"])
        city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(rotation), translation=translation)
        return city_to_egovehicle_se3

    def get_closest_im_fpath(self, log_id: str, camera_name: str, lidar_timestamp: int) -> Optional[str]:
        """
        Args:
            log_id: str, unique ID of vehicle log
            camera_name: str
            lidar_timestamp: int, timestamp of LiDAR sweep capture, in nanoseconds

        Returns:
            im_fpath, string representing path to image, or else None.
        """
        cam_timestamp = self.sdb.get_closest_cam_channel_timestamp(lidar_timestamp, camera_name, log_id)
        if cam_timestamp is None:
            return None
        im_dir = f"{self.data_dir}/{log_id}/{camera_name}"
        im_fname = f"{camera_name}_{cam_timestamp}.jpg"
        im_fpath = f"{im_dir}/{im_fname}"
        return im_fpath

    def get_closest_lidar_fpath(self, log_id: str, cam_timestamp: int) -> Optional[str]:
        """
        Args:
            log_id: str, unique ID of vehicle log
            cam_timestamp: int, timestamp of image capture, in nanoseconds

        Returns:
            ply_fpath: str, string representing path to PLY file, or else None.
        """
        lidar_timestamp = self.sdb.get_closest_lidar_timestamp(cam_timestamp, log_id)
        if lidar_timestamp is None:
            return None
        lidar_dir = f"{self.data_dir}/{log_id}/lidar"
        ply_fname = f"PC_{lidar_timestamp}.ply"
        ply_fpath = f"{lidar_dir}/{ply_fname}"
        return ply_fpath

    def get_ordered_log_ply_fpaths(self, log_id: str) -> List[str]:
        """
        Args:
            log_id: str, unique ID of vehicle log
        Returns:
            ply_fpaths: List of strings, representing paths to ply files in this log
        """
        ply_fpaths = sorted(glob.glob(f"{self.data_dir}/{log_id}/lidar/PC_*.ply"))
        return ply_fpaths

    def get_ordered_log_cam_fpaths(self, log_id: str, camera_name: str) -> List[str]:
        """
        Args
            log_id: str, unique ID of vehicle log

        Returns
            cam_img_fpaths: List of strings, representing paths to JPEG files in this log,
                for a specific camera
        """
        cam_img_fpaths = sorted(glob.glob(f"{self.data_dir}/{log_id}/{camera_name}/{camera_name}_*.jpg"))
        return cam_img_fpaths

    def get_labels_at_lidar_timestamp(self, log_id: str, lidar_timestamp: int) -> Optional[List[Mapping[str, Any]]]:
        """
        Args:
            log_id: str, unique ID of vehicle log
            lidar_timestamp: int, timestamp of LiDAR sweep capture, in nanoseconds

        Returns:
            labels: dictionary
        """
        timestamp_track_label_fpath = (
            f"{self.labels_dir}/{log_id}/per_sweep_annotations_amodal/tracked_object_labels_{lidar_timestamp}.json"
        )
        if not Path(timestamp_track_label_fpath).exists():
            return None

        labels = read_json_file(timestamp_track_label_fpath)
        assert isinstance(labels, list), labels
        return labels
