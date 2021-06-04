#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import collections.abc
import glob
import json
import logging
from pathlib import Path
from typing import List, NamedTuple

import numpy as np

from argoverse.data_loading.object_classes import OBJ_CLASS_MAPPING_DICT

logger = logging.getLogger(__name__)


class TrajectoryLabel(NamedTuple):
    """Trajectory object.

    Args:
        timestamps (np.array): Array of timestamps for trajectory.
        quaternions (np.array): Array of quaternions for trajectory.
        translations (np.array): Array of translations of SE3 poses for trajectory.
        obj_class (int): Object class id.
        obj_class_str (str): Object class name.
        occlusions (np.array): Array of occlusions for trajectory.
        track_uuid (str): Track uuid.
        log_id (str): Log id.
        max_length (float): Maximum length for trajectory.
        max_width (float): Maximum width for trajectory.
        max_height (float): Maximum height for trajectory.
        lengths (np.array): Array of lengths for trajectory.
        widths (np.array): Array of widths for trajectory.
        heights (np.array): Array of heights for trajectory.
    """

    timestamps: np.ndarray
    quaternions: np.ndarray
    translations: np.ndarray
    obj_class: int
    obj_class_str: str
    occlusion: np.ndarray
    track_uuid: str
    log_id: str
    max_length: float
    max_width: float
    max_height: float
    lengths: np.ndarray
    widths: np.ndarray
    heights: np.ndarray


def load_json_track_labels(log_track_labels_dir: str) -> List[TrajectoryLabel]:
    """Trajectories are stored on disk as 1 track per 1 json file.
     We load all labeled tracks here from the JSON files.

    Args:
        log_track_labels_dir (str): Log track directory.

    Returns:
        List[TrajectoryLabel]: a Python list of TrajectoryLabels.
    """
    json_fpath_list = glob.glob(log_track_labels_dir, recursive=True)
    trajectories = []

    for file_idx, json_fpath in enumerate(json_fpath_list):
        with open(json_fpath, "r") as f:
            json_data = json.load(f)

        track_uuid = Path(json_fpath).stem
        obj_cls = json_data["label_class"]

        # recent MLDS change
        if isinstance(obj_cls, collections.abc.Mapping):
            obj_cls = obj_cls["name"]

        if obj_cls in OBJ_CLASS_MAPPING_DICT:
            obj_cls_idx = OBJ_CLASS_MAPPING_DICT[obj_cls]
        else:
            logger.error(f"Unrecognized class {obj_cls}")
            raise ValueError(f"Unrecognized class {obj_cls}")

        quaternions = []
        translations = []
        timestamps = []
        occlusions = []

        lengths = []
        widths = []
        heights = []

        for track_frame in json_data["track_label_frames"]:

            # cuboid not occluded if not explicitly indicated.
            occlusions.append(track_frame.get("occlusion", 0))
            tr_center = track_frame["center"]
            tr_x = tr_center["x"]
            tr_y = tr_center["y"]
            tr_z = tr_center["z"]

            translation = np.array([tr_x, tr_y, tr_z])
            translations.append(translation)

            tr_rot = track_frame["rotation"]
            rot_w = tr_rot["w"]
            rot_x = tr_rot["x"]
            rot_y = tr_rot["y"]
            rot_z = tr_rot["z"]

            quaternion = np.array([rot_w, rot_x, rot_y, rot_z])
            quaternions.append(quaternion)

            timestamps.append(track_frame["timestamp"])

            lengths.append(track_frame["length"])
            widths.append(track_frame["width"])
            heights.append(track_frame["height"])

        trajectory = TrajectoryLabel(
            timestamps=np.array(timestamps),
            quaternions=np.array(quaternions),
            translations=np.array(translations),
            obj_class=obj_cls_idx,
            obj_class_str=obj_cls,
            occlusion=np.array(occlusions),
            track_uuid=track_uuid,
            log_id=json_fpath.split("/")[-3],
            max_length=max(lengths),
            max_width=max(widths),
            max_height=max(heights),
            lengths=np.array(lengths),
            widths=np.array(widths),
            heights=np.array(heights),
        )
        trajectories.append(trajectory)

    return trajectories
