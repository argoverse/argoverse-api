# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>

import glob
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import cv2
import numpy as np

from argoverse.utils.calibration import Calibration, load_calib
from argoverse.utils.camera_stats import STEREO_CAMERA_LIST

logger = logging.getLogger(__name__)

STEREO_FRONT_LEFT_RECT = STEREO_CAMERA_LIST[2]
STEREO_FRONT_RIGHT_RECT = STEREO_CAMERA_LIST[3]


class ArgoverseStereoLoader:
    def __init__(self, root_dir: str, split_name: str) -> None:
        # initialize class member
        self.STEREO_CAMERA_LIST = STEREO_CAMERA_LIST
        self._log_list: Optional[List[str]] = None
        self._image_list: Optional[Dict[str, Dict[str, List[str]]]] = None
        self._disparity_list: Optional[Dict[str, Dict[str, List[str]]]] = None
        self._image_timestamp_list: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._timestamp_image_dict: Optional[Dict[str, Dict[str, Dict[int, str]]]] = None
        self._calib: Optional[Dict[str, Dict[str, Calibration]]] = None
        self.counter: int = 0

        self.image_count: int = 0

        self.root_dir: str = root_dir
        self.split_name: str = split_name

        self.current_log = self.log_list[self.counter]

        assert self.image_list is not None

        # load calibration file
        self.calib_filename: str = os.path.join(
            self.root_dir,
            "rectified_stereo_images_v1.1",
            self.split_name,
            self.current_log,
            "vehicle_calibration_stereo_info.json",
        )

        # stereo camera @5hz
        self.num_stereo_camera_frame: int = len(self.image_timestamp_list[STEREO_FRONT_LEFT_RECT])

        assert self.calib is not None

    @property
    def calib(self) -> Dict[str, Calibration]:
        """get calibration dict for current log

        Returns:
            calib: Calibration object for the current log
        """
        self._ensure_calib_is_populated()
        assert self._calib is not None
        return self._calib[self.current_log]

    def _ensure_calib_is_populated(self) -> None:
        """load up calibration object for all logs

        Returns:
            None
        """
        if self._calib is None:
            self._calib = {}
            for log in self.log_list:
                calib_filename = os.path.join(
                    self.root_dir,
                    "rectified_stereo_images_v1.1",
                    self.split_name,
                    log,
                    "vehicle_calibration_stereo_info.json",
                )
                self._calib[log] = load_calib(calib_filename)

    @property
    def log_list(self) -> List[str]:
        """return list of log (str) in the current dataset directory

        Returns:
            log_list: list of string representing log id
        """
        if self._log_list is None:

            def valid_log(log: str) -> bool:
                return os.path.exists(
                    os.path.join(
                        self.root_dir,
                        "rectified_stereo_images_v1.1",
                        self.split_name,
                        log,
                        "vehicle_calibration_stereo_info.json",
                    )
                )

            self._log_list = [
                x
                for x in os.listdir(os.path.join(self.root_dir, "rectified_stereo_images_v1.1", self.split_name))
                if valid_log(x)
            ]

        return self._log_list

    @property
    def image_list(self) -> Dict[str, List[str]]:
        """return list of all image path (str) for the current log

        Returns:
            image_list: dictionary of list of image, with camera name as key
        """
        if self._image_list is None:
            self._image_list = {}
            for log in self.log_list:
                self._image_list[log] = {}
                for camera in STEREO_CAMERA_LIST:
                    self._image_list[log][camera] = sorted(
                        glob.glob(
                            (
                                os.path.join(
                                    self.root_dir, "rectified_stereo_images_v1.1", self.split_name, log, camera, "*.jpg"
                                )
                            )
                        )
                    )
                    self.image_count += len(self._image_list[log][camera])
        return self._image_list[self.current_log]

    @property
    def disparity_list(self) -> Dict[str, List[str]]:
        """return list of all image paths (str) for all cameras for the current log

        Returns:
            image_list: dictionary of list of image, with camera name as key
        """
        if self._disparity_list is None:
            self._disparity_list = {}
            for log in self.log_list:
                self._disparity_list[log] = {}
                STEREO_DISPARITY_LIST = ["stereo_front_left_rect_disparity", "stereo_front_left_rect_objects_disparity"]
                for disparity in STEREO_DISPARITY_LIST:
                    self._disparity_list[log][disparity] = sorted(
                        glob.glob(
                            (
                                os.path.join(
                                    self.root_dir, "disparity_maps_v1.1", self.split_name, log, disparity, "*.png"
                                )
                            )
                        )
                    )
                    self.image_count += len(self._disparity_list[log][disparity])
        return self._disparity_list[self.current_log]

    @property
    def image_timestamp_list(self) -> Dict[str, List[int]]:
        """return dict of list of image timestamp (str) for the current log.

        Returns:
            image_timestamp_list: dictionary of list of image timestamp for all cameras
        """
        assert self.image_list is not None
        assert self._image_list is not None

        if self._image_timestamp_list is None:
            self._image_timestamp_list = {}
            for log in self.log_list:
                self._image_timestamp_list[log] = {}
                for camera in STEREO_CAMERA_LIST:
                    self._image_timestamp_list[log][camera] = [
                        int(Path(x).stem.split("_")[-1]) for x in self._image_list[log][camera]
                    ]

        return self._image_timestamp_list[self.current_log]

    @property
    def timestamp_image_dict(self) -> Dict[str, Dict[int, str]]:
        """return dict of list of image path (str) for the current log, index by timestamp.

        Returns:
            timestamp_image_dict: dictionary of list of image path for all cameras, with timestamp as key
        """
        if self._timestamp_image_dict is None:
            assert self.image_timestamp_list is not None
            assert self._image_timestamp_list is not None
            assert self.image_list is not None
            assert self._image_list is not None

            self._timestamp_image_dict = {}

            for log in self.log_list:
                self._timestamp_image_dict[log] = {}
                for camera in STEREO_CAMERA_LIST:
                    self._timestamp_image_dict[log][camera] = {
                        self._image_timestamp_list[log][camera][i]: self._image_list[log][camera][i]
                        for i in range(len(self._image_timestamp_list[log][camera]))
                    }

        return self._timestamp_image_dict[self.current_log]

    def __iter__(self) -> Iterator["ArgoverseStereoLoader"]:
        self.counter = -1

        return self

    def __next__(self) -> "ArgoverseStereoLoader":
        self.counter += 1

        if self.counter >= len(self):
            raise StopIteration
        else:
            self.current_log = self.log_list[self.counter]
            self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_FRONT_LEFT_RECT])
            return self

    def __len__(self) -> int:
        return len(self.log_list)

    def __str__(self) -> str:
        frame_image_stereo = self.num_stereo_camera_frame

        num_images = [len(self.image_list[cam]) for cam in STEREO_CAMERA_LIST]

        start_time = self.image_timestamp_list[STEREO_FRONT_LEFT_RECT][0]
        end_time = self.image_timestamp_list[STEREO_FRONT_LEFT_RECT][-1]

        time_in_sec = (end_time - start_time) * 10 ** (-9)
        return f"""
--------------------------------------------------------------------
------Log id: {self.current_log}
--------------------------------------------------------------------
Length of the sequence: {time_in_sec:.2f} seconds
Number of stereo pair frames (@5 Hz): {frame_image_stereo}
        """

    def __getitem__(self, key: int) -> "ArgoverseStereoLoader":
        self.counter = key
        self.current_log = self.log_list[self.counter]
        self.num_disparity_frame = len(self.disparity_list)
        self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_FRONT_LEFT_RECT])
        return self

    def get(self, log_id: str) -> "ArgoverseStereoLoader":
        """get ArgoverseStereoLoader object with current_log set to specified log_id

        Args:
            log_id: log id
        Returns:
            ArgoverseStereoLoader: with current_log set to log_id
        """
        self.current_log = log_id
        self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_FRONT_LEFT_RECT])
        return self

    def get_image_list(self, camera: str, log_id: Optional[str] = None, load: bool = False) -> List[str]:
        """Get list of image/or image path

        Args:
            camera: camera based on camera_stats.STEREO_CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            List[str]: list of image paths (str).
        """
        assert self.image_list is not None
        assert self._image_list is not None

        if log_id is None:
            log_id = self.current_log
        if load:
            return [self.get_image(i, camera) for i in range(len(self._image_list[log_id][camera]))]

        return self._image_list[log_id][camera]

    def get_disparity_list(self, name: str, log_id: Optional[str] = None, load: bool = False) -> List[str]:
        """Get list of image/or image path

        Args:
            name: disparity map name, one of:

                ["stereo_front_left_rect_disparity",
                 "stereo_front_left_rect_objects_disparity"]

            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            List[str]: list of disparity image paths (str),
        """
        assert self.disparity_list is not None
        assert self._disparity_list is not None

        if log_id is None:
            log_id = self.current_log
        if load:
            return [self.get_disparity_map(i, name) for i in range(len(self._disparity_list[log_id][name]))]

        return self._disparity_list[log_id][name]

    def get_image_at_timestamp(
        self,
        timestamp: int,
        camera: str,
        log_id: Optional[str] = None,
        load: bool = True,
    ) -> Optional[Union[str, np.ndarray]]:
        """get image or image path at a specific timestamp

        Args:
            timestamp: timestamp
            camera: camera based on camera_stats.STEREO_CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.timestamp_image_dict is not None
        assert self._timestamp_image_dict is not None

        if log_id is None:
            log_id = self.current_log
        assert self.timestamp_image_dict is not None
        try:
            image_path = self._timestamp_image_dict[log_id][camera][timestamp]
        except KeyError:
            logging.error(f"Cannot find {camera} image at timestamp {timestamp} in log {log_id}")
            return None

        if load:
            return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return image_path

    def get_image(
        self, idx: int, camera: str, log_id: Optional[str] = None, load: bool = True
    ) -> Union[str, np.ndarray]:
        """get image or image path at a specific index (in image index)

        Args:
            idx: image based 0-index
            camera: camera based on camera_stats.STEREO_CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_timestamp_list is not None
        assert self._image_timestamp_list is not None
        assert self.image_list is not None
        assert self._image_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._image_timestamp_list[log_id][camera])
        image_path = self._image_list[log_id][camera][idx]

        if load:
            return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return image_path

    def get_disparity_map(
        self, idx: int, name: str, log_id: Optional[str] = None, load: bool = True
    ) -> Union[str, np.ndarray]:
        """get disparity image or disparity image path at a specific index (in image index)

        Args:
            idx: image based 0-index
            name: disparity map name, one of:

                ["stereo_front_left_rect_disparity",
                 "stereo_front_left_rect_objects_disparity"]

            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_timestamp_list is not None
        assert self._image_timestamp_list is not None
        assert self.disparity_list is not None
        assert self._disparity_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._image_timestamp_list[log_id][STEREO_FRONT_LEFT_RECT])
        disparity_path = self._disparity_list[log_id][name][idx]

        if load:
            return np.float32(cv2.imread(disparity_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)) / 256.0
        return disparity_path

    def get_calibration(self, camera: str, log_id: Optional[str] = None) -> Calibration:
        """Get calibration corresponding to the camera.

        Args:
            camera: name of the camera; one of:

               ["ring_front_center",
                "ring_front_left",
                "ring_front_right",
                "ring_rear_left",
                "ring_rear_right",
                "ring_side_left",
                "ring_side_right",
                "stereo_front_left",
                "stereo_front_right",
                "stereo_front_left_rect",
                "stereo_front_right_rect"]

            log_id: ID of log to search (default: current log)

        Returns:
            Calibration info for a particular index
        """
        self._ensure_calib_is_populated()
        assert self._calib is not None

        if log_id is None:
            log_id = self.current_log

        return self._calib[log_id][camera]
