# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import argparse
import glob
import json
import logging
import os
import pathlib
from typing import Any, Dict, List, TextIO, Union

import motmetrics as mm
import numpy as np
from argoverse.evaluation.eval_utils import get_pc_inside_bbox, label_to_bbox, leave_only_roi_region
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

min_point_num = 50
mh = mm.metrics.create()


logger = logging.getLogger(__name__)


_PathLike = Union[str, "os.PathLike[str]"]


def in_distance_range_pose(ego_center: np.ndarray, pose: np.ndarray, d_min: float, d_max: float) -> bool:
    """Determine if a pose is within distance range or not.

    Args:
        ego_center: ego center pose (zero if bbox is in ego frame).
        pose:  pose to test.
        d_min: minimum distance range
        d_max: maximum distance range

    Returns:
        A boolean saying if input pose is with specified distance range.
    """

    dist = float(np.linalg.norm(pose[0:2] - ego_center[0:2]))

    return dist > d_min and dist < d_max


def get_distance(x1: np.ndarray, x2: np.ndarray, name: str) -> float:
    """Get the distance between two poses, returns nan if distance is larger than detection threshold.

    Args:
        x1: first pose
        x2: second pose
        name: name of the field to test

    Returns:
        A distance value or NaN
    """

    dist = float(np.linalg.norm(x1[name][0:2] - x2[name][0:2]))
    return dist if dist < 2.25 else float(np.nan)


def eval_tracks(
    path_tracker_output: str,
    path_dataset: _PathLike,
    d_min: float,
    d_max: float,
    out_file: TextIO,
    centroid_method: str,
) -> None:
    """Evaluate tracking output.

    Args:
        path_tracker_output: path to tracker output
        path_dataset: path to dataset
        d_min: minimum distance range
        d_max: maximum distance range
        out_file: output file object
        centroid_method: method for ground truth centroid estimation
    """
    acc = mm.MOTAccumulator(auto_id=True)

    path_track_data = sorted(glob.glob(os.fspath(path_tracker_output) + "/*"))

    log_id = pathlib.Path(path_dataset).name
    logger.info("log_id = %s", log_id)

    city_info_fpath = f"{path_dataset}/city_info.json"
    city_info = read_json_file(city_info_fpath)
    city_name = city_info["city_name"]
    logger.info("city name = %s", city_name)

    ID_gt_all: List[str] = []

    for ind_frame in range(len(path_track_data)):
        if ind_frame % 50 == 0:
            logger.info("%d/%d" % (ind_frame, len(path_track_data)))

        timestamp_lidar = int(path_track_data[ind_frame].split("/")[-1].split("_")[-1].split(".")[0])
        path_gt = os.path.join(
            path_dataset, "per_sweep_annotations_amodal", f"tracked_object_labels_{timestamp_lidar}.json"
        )

        if not os.path.exists(path_gt):
            logger.warning("Missing ", path_gt)
            continue

        gt_data = read_json_file(path_gt)

        pose_data = read_json_file(f"{path_dataset}/poses/city_SE3_egovehicle_{timestamp_lidar}.json")
        rotation = np.array(pose_data["rotation"])
        translation = np.array(pose_data["translation"])
        ego_R = quat2rotmat(rotation)
        ego_t = translation
        egovehicle_to_city_se3 = SE3(rotation=ego_R, translation=ego_t)

        pc_raw0 = load_ply(os.path.join(path_dataset, f"lidar/PC_{timestamp_lidar}.ply"))
        pc_raw_roi = leave_only_roi_region(
            pc_raw0, egovehicle_to_city_se3, ground_removal_method="no", city_name=city_name
        )

        gt: Dict[str, Dict[str, Any]] = {}
        id_gts = []
        for i in range(len(gt_data)):

            if gt_data[i]["label_class"] != "VEHICLE":
                continue

            bbox, orientation = label_to_bbox(gt_data[i])
            pc_segment = get_pc_inside_bbox(pc_raw_roi, bbox)

            center = np.array([gt_data[i]["center"]["x"], gt_data[i]["center"]["y"], gt_data[i]["center"]["z"]])
            if (
                len(pc_segment) >= min_point_num
                and bbox[3] > 0
                and in_distance_range_pose(np.zeros(3), center, d_min, d_max)
            ):
                track_label_uuid = gt_data[i]["track_label_uuid"]
                gt[track_label_uuid] = {}
                if centroid_method == "average":
                    gt[track_label_uuid]["centroid"] = pc_segment.sum(axis=0) / len(pc_segment)
                elif centroid_method == "label_center":
                    gt[track_label_uuid]["centroid"] = center

                else:
                    logger.warning("Not implemented")

                gt[track_label_uuid]["bbox"] = bbox
                gt[track_label_uuid]["orientation"] = orientation

                if track_label_uuid not in ID_gt_all:
                    ID_gt_all.append(track_label_uuid)

                id_gts.append(track_label_uuid)

        tracks: Dict[str, Dict[str, Any]] = {}
        id_tracks: List[str] = []

        track_data = read_json_file(path_track_data[ind_frame])

        for track in track_data:
            key = track["track_label_uuid"]

            if track["label_class"] != "VEHICLE" or track["height"] == 0:
                continue

            center = np.array([track["center"]["x"], track["center"]["y"], track["center"]["z"]])

            if in_distance_range_pose(np.zeros(3), center, d_min, d_max):
                tracks[key] = {}
                tracks[key]["centroid"] = center

                id_tracks.append(key)

        dists: List[List[float]] = []
        for gt_key, gt_value in gt.items():
            gt_track_data: List[float] = []
            dists.append(gt_track_data)
            for track_key, track_value in tracks.items():
                gt_track_data.append(get_distance(gt_value, track_value, "centroid"))

        acc.update(id_gts, id_tracks, dists)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
        ],
        name="acc",
    )
    logger.info("summary = %s", summary)
    num_tracks = len(ID_gt_all)

    fn = os.path.basename(path_tracker_output)
    num_frames = summary["num_frames"][0]
    mota = summary["mota"][0] * 100
    motp = summary["motp"][0]
    idf1 = summary["idf1"][0]
    most_track = summary["mostly_tracked"][0] / num_tracks
    most_lost = summary["mostly_lost"][0] / num_tracks
    num_fp = summary["num_false_positives"][0]
    num_miss = summary["num_misses"][0]
    num_switch = summary["num_switches"][0]
    num_flag = summary["num_fragmentations"][0]

    out_string = (
        f"{fn} {num_frames} {mota:.2f} {motp:.2f} {idf1:.2f} {most_track:.2f} "
        f"{most_lost:.2f} {num_fp} {num_miss} {num_switch} {num_flag} \n"
    )
    out_file.write(out_string)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path_tracker_output",
        type=str,
        default="../../argodataset_30Hz/test_label/028d5cb1-f74d-366c-85ad-84fde69b0fd3",
    )
    parser.add_argument(
        "--path_labels", type=str, default="../../argodataset_30Hz/labels_v32/028d5cb1-f74d-366c-85ad-84fde69b0fd3"
    )
    parser.add_argument("--path_dataset", type=str, default="../../argodataset_30Hz/cvpr_test_set")
    parser.add_argument("--centroid_method", type=str, default="average", choices=["label_center", "average"])
    parser.add_argument("--flag", type=str, default="")
    parser.add_argument("--d_min", type=float, default=0)
    parser.add_argument("--d_max", type=float, default=100, required=True)

    args = parser.parse_args()
    logger.info("args = %s", args)

    tracker_basename = os.path.basename(args.path_tracker_output)

    out_filename = (f"{tracker_basename}_{args.flag}_{int(args.d_min)}_{int(args.d_max)}_{args.centroid_method}.txt")
    logger.info("output file name = %s", out_filename)

    with open(out_filename, "w") as out_file:
        eval_tracks(args.path_tracker_output, args.path_dataset, args.d_min, args.d_max, out_file, args.centroid_method)
