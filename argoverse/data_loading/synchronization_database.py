#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import glob
import logging
import pdb
from typing import Dict, Iterable, List, Optional, cast

import numpy as np
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.json_utils import read_json_file

logger = logging.getLogger(__name__)


def get_timestamps_from_sensor_folder(sensor_folder_wildcard: str) -> np.ndarray:
    """ Timestamp always lies at end of filename

        Args:
            sensor_folder_wildcard: string to glob to find all filepaths for a particular
                        sensor files within a single log run

        Returns:
            Numpy array of integers, representing timestamps
    """

    path_generator = glob.glob(sensor_folder_wildcard)
    path_generator.sort()

    return np.array([int(jpg_fpath.split("/")[-1].split(".")[0].split("_")[-1]) for jpg_fpath in path_generator])


class SynchronizationDB:

    # Camera is 30 Hz (once per 33.3 milliseconds)
    # LiDAR is 10 Hz
    # Max we are halfway between 33.3 milliseconds on either side
    # then convert milliseconds to nanoseconds
    MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF = 33.3 * (1.0 / 2) * (1.0 / 1000) * 1e9
    # Stereo Camera is 5 Hz (once per 200 milliseconds)
    # LiDAR is 10 Hz
    # Since Stereo is more sparse, we look for 200 millisecond on either side
    # then convert milliseconds to nanoseconds
    MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF = 200 * 1.0 * (1.0 / 1000) * 1e9

    def __init__(self, dataset_dir: str, collect_single_log_id: Optional[str] = None) -> None:
        """ Build the SynchronizationDB.
            Note that the timestamps for each camera channel are not identical, but they are clustered together.

            Args:
                dataset_dir: path to dataset.
                collect_single_log_id: log id to process. (All if not set)

            Returns:
                None
        """
        logger.info("Building SynchronizationDB")

        if collect_single_log_id is None:
            log_fpaths = glob.glob(f"{dataset_dir}/*")
        else:
            log_fpaths = [f"{dataset_dir}/{collect_single_log_id}"]

        self.per_log_pose_index: Dict[str, Dict[str, np.ndarray]] = {}

        for log_fpath in log_fpaths:
            log_id = log_fpath.split("/")[-1]

            self.per_log_pose_index[log_id] = {}
            for camera_name in STEREO_CAMERA_LIST + RING_CAMERA_LIST:

                sensor_folder_wildcard = f"{dataset_dir}/{log_id}/{camera_name}/{camera_name}_*.jpg"
                tovs = get_timestamps_from_sensor_folder(sensor_folder_wildcard)

                self.per_log_pose_index[log_id][camera_name] = tovs

    def get_valid_logs(self) -> Iterable[str]:
        """ Return the log_ids for which the SynchronizationDatabase contains pose information.
        """
        return self.per_log_pose_index.keys()

    def get_closest_cam_channel_timestamp(self, lidar_timestamp: int, camera_name: str, log_id: str) -> Optional[int]:
        """ Grab the LiDAR pose file. Get its timestamp, and then find the pose message
            with the closest absolute timestamp.

            Instead of sorting the whole array of timestamp differences, we just
            take the minimum value (to speed up this function).

            Args:
                lidar_timestamp: integer
                camera_name: string, representing path to log directories
                log_id: string

            Returns:
                closest_cam_ch_timestamp: closest timestamp
        """
        if log_id not in self.per_log_pose_index or camera_name not in self.per_log_pose_index[log_id]:
            return None

        timestamps = self.per_log_pose_index[log_id][camera_name]
        # catch case if no files were loaded for a particular sensor
        if not timestamps.tolist():
            return None

        closest_ind = np.argmin(np.absolute(timestamps - lidar_timestamp))
        closest_cam_ch_timestamp = cast(int, timestamps[closest_ind])  # mypy does not understand numpy arrays
        timestamp_diff = np.absolute(lidar_timestamp - closest_cam_ch_timestamp)
        if timestamp_diff > self.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF and camera_name in RING_CAMERA_LIST:
            # convert to nanoseconds->milliseconds for readability
            logger.error(
                "No corresponding image: %s > %s ms", timestamp_diff / 1e6, self.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF / 1e6
            )
            return None
        elif timestamp_diff > self.MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF and camera_name in STEREO_CAMERA_LIST:
            # convert to nanoseconds->milliseconds for readability
            logger.error(
                "No corresponding image: %s > %s ms",
                timestamp_diff / 1e6,
                self.MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF / 1e6,
            )
            return None
        return closest_cam_ch_timestamp


if __name__ == "__main__":
    # example usage

    root_dir = "./"
    camera = "ring_front_center"
    id = "c6911883-1843-3727-8eaa-41dc8cda8993"

    db = SynchronizationDB(root_dir)
    get_timestamps_from_sensor_folder(root_dir)
    db.get_valid_logs()
    db.get_closest_cam_channel_timestamp(0, camera, id)
