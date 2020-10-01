#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import glob
import logging
import os
import pickle as pkl
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from argoverse.data_loading.frame_record import FrameRecord
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.data_loading.trajectory_loader import TrajectoryLabel, load_json_track_labels
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.pkl_utils import load_pkl_dictionary, save_pkl_dictionary
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

logger = logging.getLogger(__name__)


class PerFrameLabelAccumulator:
    """We will cache the accumulated track label trajectories per city, per log, and per frame.
    In order to plot each frame sequentially, one at a time, we need to aggregate beforehand
    the tracks and cuboids for each frame.

    Attributes:
        bboxes_3d (bool): to use 3d bounding boxes (True) or 2d bounding boxes (False).
        dataset_dir (str): Dataset directory.
        labels_dir (str): Labels directory.
        log_egopose_dict (dict): Egopose per log id and timestamp.
        log_timestamp_dict (dict): List of frame records per log id and timestamp.
        per_city_traj_dict (dict): Per city trajectory dictionary.
        sdb (SynchronizationDB): Synchronization DB.
    """

    def __init__(
        self,
        dataset_dir: str,
        labels_dir: str,
        experiment_prefix: str,
        bboxes_3d: bool = False,
        save: bool = True,
    ) -> None:
        """Initialize PerFrameLabelAccumulator object for use with tracking benchmark data.

        Args:
            dataset_dir (str): Dataset directory.
            labels_dir (str): Labels directory.
            experiment_prefix (str): Prefix for experimint to use.
            bboxes_3d (bool, optional): to use 3d bounding boxes (True) or 2d bounding boxes (False).
        """
        self.bboxes_3d = bboxes_3d

        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        tmp_dir = tempfile.gettempdir()
        per_city_traj_dict_fpath = f"{tmp_dir}/per_city_traj_dict_{experiment_prefix}.pkl"
        log_egopose_dict_fpath = f"{tmp_dir}/log_egopose_dict_{experiment_prefix}.pkl"
        log_timestamp_dict_fpath = f"{tmp_dir}/log_timestamp_dict_{experiment_prefix}.pkl"

        coordinate_system = "map_world_fr"
        self.per_city_traj_dict: Dict[str, List[Tuple[np.ndarray, str]]] = {
            "MIA": [],
            "PIT": [],
        }  # all the trajectories for these 2 cities
        self.log_egopose_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
        self.log_timestamp_dict: Dict[str, Dict[int, List[FrameRecord]]] = {}
        self.sdb = SynchronizationDB(self.dataset_dir)

        if save:
            self.accumulate_per_log_data()
            save_pkl_dictionary(per_city_traj_dict_fpath, self.per_city_traj_dict)
            save_pkl_dictionary(log_egopose_dict_fpath, self.log_egopose_dict)
            save_pkl_dictionary(log_timestamp_dict_fpath, self.log_timestamp_dict)

    def accumulate_per_log_data(self, log_id: Optional[str] = None) -> None:
        """Loop through all of the logs that we have. Get the labels that pertain to the
        benchmark (i.e. tracking or detection) that we are interested in.

        We use a unique color to describe each trajectory, and then we store the
        instance of the trajectory, along with its color, *PER FRAME* , per log.

        """
        MIAMI_CUBOID_COUNT = 0
        PITT_CUBOID_COUNT = 0

        log_fpaths = glob.glob(f"{self.dataset_dir}/*")
        log_fpaths = [f for f in log_fpaths if os.path.isdir(f)]
        num_benchmark_logs = len(log_fpaths)

        for log_idx, log_fpath in enumerate(log_fpaths):
            log_id_ = log_fpath.split("/")[-1]
            if log_id is not None:
                if log_id_ != log_id:
                    continue
            if log_id_ not in self.sdb.get_valid_logs():
                continue

            city_info_fpath = f"{self.dataset_dir}/{log_id_}/city_info.json"
            city_info = read_json_file(city_info_fpath)
            log_city_name = city_info["city_name"]
            if log_city_name not in self.per_city_traj_dict:
                logger.warning(f"{log_city_name} not listed city")
                continue

            self.log_egopose_dict[log_id_] = {}
            self.log_timestamp_dict[log_id_] = {}

            traj_labels = self.get_log_trajectory_labels(log_id_)
            if traj_labels is None:
                continue  # skip this log since no tracking data

            for traj_idx, traj_label in enumerate(traj_labels):
                if (traj_idx % 500) == 0:
                    logger.info(f"On traj index {traj_idx}")
                traj_city_fr = self.place_trajectory_in_city_frame(traj_label, log_id_)
                # we don't know the city name until here
                if traj_idx == 0:
                    logger.info(f"Log {log_id_} has {len(traj_labels)} trajectories in {log_city_name}")

                self.per_city_traj_dict[log_city_name].append((traj_city_fr, log_id_))

        logger.info(f"We looked at {num_benchmark_logs} tracking logs")
        logger.info(f"Miami has {MIAMI_CUBOID_COUNT} and Pittsburgh has {PITT_CUBOID_COUNT} cuboids")

    def get_log_trajectory_labels(self, log_id: str) -> Optional[List[TrajectoryLabel]]:
        """Create a very large list with all of the trajectory data.

        Treat a single object cuboid label as one step in a trajectory.
        Then we can share the same representation for both.

        Args:
            log_id (str): Log id to load.

        Returns:
            List[TrajectoryLabel]: List of trajectory labels.
        """
        path = f"{self.labels_dir}/{log_id}/track_labels_amodal"
        if Path(path).exists():
            return load_json_track_labels(f"{path}/*.json")
        else:
            return None

    def place_trajectory_in_city_frame(self, traj_label: TrajectoryLabel, log_id: str) -> np.ndarray:
        """Place trajectory in the city frame
        Args:
            traj_label (TrajectoryLabel): instance of the TrajectoryLabel class.
            log_id (str): Log id.

        Returns:
            -   traj_city_fr: trajectory length of NUM_CUBOID_VERTS (x,y,z) coords per cuboid.

        """
        seq_len = traj_label.timestamps.shape[0]

        if self.bboxes_3d:
            NUM_CUBOID_VERTS = 8
        else:
            NUM_CUBOID_VERTS = 4

        # store NUM_CUBOID_VERTS (x,y,z) coords per cuboid
        traj_city_fr = np.zeros((seq_len, NUM_CUBOID_VERTS, 3))
        rand_color = (
            float(np.random.rand()),
            float(np.random.rand()),
            float(np.random.rand()),
        )
        logger.info(f"On log {log_id} with {traj_label.track_uuid}")
        for t in range(seq_len):

            obj_label_rec = ObjectLabelRecord(
                quaternion=traj_label.quaternions[t],
                translation=traj_label.translations[t],
                length=traj_label.max_length,
                width=traj_label.max_width,
                height=traj_label.max_height,
                occlusion=traj_label.occlusion[t],
            )

            timestamp = int(traj_label.timestamps[t])

            if self.bboxes_3d:
                bbox_ego_frame = obj_label_rec.as_3d_bbox()
            else:
                bbox_ego_frame = obj_label_rec.as_2d_bbox()

            bbox_city_fr, pose_city_to_ego = self.convert_bbox_to_city_frame(
                timestamp, self.dataset_dir, log_id, bbox_ego_frame
            )
            if bbox_city_fr is None:
                logger.warning(f"\t {log_id}: Couldnt find the pose for {traj_label.track_uuid}!")
                continue

            self.log_egopose_dict[log_id][timestamp] = pose_city_to_ego

            frame_rec = FrameRecord(
                bbox_city_fr=bbox_city_fr,
                bbox_ego_frame=bbox_ego_frame,
                occlusion_val=obj_label_rec.occlusion,
                color=rand_color,
                track_uuid=traj_label.track_uuid,
                obj_class_str=traj_label.obj_class_str,
            )

            self.log_timestamp_dict[log_id].setdefault(timestamp, []).append(frame_rec)

            traj_city_fr[t] = bbox_city_fr

        return traj_city_fr

    def convert_bbox_to_city_frame(
        self,
        lidar_timestamp_ns: int,
        dataset_dir: str,
        log_id: str,
        bbox_ego_frame: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Convert bounding box to city frame.
        Args:
            lidar_timestamp_ns (int): Lidar timestamp.
            dataset_dir (str): representing full path to the log_ids.
            log_id (str): e.g. '3ced8dba-62d0-3930-8f60-ebeea2feabb8'.
            bbox_ego_frame (np.ndarray): Numpy array of shape (4,3), representing bounding box in egovehicle frame

        Returned:
            bbox_city_fr: Numpy array of shape (4,3), representing bounding box in CITY frame
            pose_city_to_ego: dictionary, has two fields: 'translation' and 'rotation'
                        describing the SE(3) for p_city = city_to_egovehicle_se3 * p_egovehicle
        """
        city_to_egovehicle_se3 = get_city_SE3_egovehicle_at_sensor_t(lidar_timestamp_ns, dataset_dir, log_id)
        if city_to_egovehicle_se3 is None:
            raise RuntimeError(
                f"Could not get city to egovehicle coordinate transformation at timestamp {lidar_timestamp_ns}"
            )

        bbox_city_fr = city_to_egovehicle_se3.transform_point_cloud(bbox_ego_frame)
        pose_city_to_ego = {
            "rotation": city_to_egovehicle_se3.rotation,
            "translation": city_to_egovehicle_se3.translation,
        }
        return bbox_city_fr, pose_city_to_ego
