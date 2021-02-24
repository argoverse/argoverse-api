# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import os
import pathlib

import numpy as np

from argoverse.evaluation import eval_tracking, eval_utils
from argoverse.utils import ply_loader
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

TEST_DATA_LOC = pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "tracking"
D_MIN = 0
D_MAX = 100


def test_in_distance_range() -> None:
    """
    test for in_distance_range_pose()
    """

    assert eval_tracking.in_distance_range_pose(np.array([0, 0]), np.array([1, 1]), 0, 2)
    assert not eval_tracking.in_distance_range_pose(np.array([0, 0]), np.array([1, 1]), 0, 0.5)

    city_center = np.array((-100, 100))
    assert not eval_tracking.in_distance_range_pose(city_center, (0, 0), 0, 50)


def test_get_pc_inside_box() -> None:
    """
    test get_pc_inside_box(pc_raw,bbox) *bbox format is [p0,p1,p2,h]

                - -------- -
               /|         /|
              - -------- - . h
              | |        | |
              . p2 --------
              |/         |/
             p0 -------- p1


    :test lidar data:
    x    y    z  intensity  laser_number
    0  0.0  0.0  5.0        4.0          31.0
    1  1.0  0.0  5.0        1.0          14.0
    2  2.0  0.0  5.0        0.0          16.0
    3  3.0  0.0  5.0       20.0          30.0
    4  4.0  0.0  5.0        3.0          29.0
    5  5.0  0.0  5.0        1.0          11.0
    6  6.0  0.0  5.0       31.0          13.0
    7  7.0  0.0  5.0        2.0          28.0
    8  8.0  0.0  5.0        5.0          27.0
    9  9.0  0.0  5.0        6.0          10.0
    """
    bbox = np.array(
        [
            np.array([[0], [0], [0]]),
            np.array([[2], [0], [0]]),
            np.array([[0], [5], [0]]),
            np.array(10),
        ]
    )

    pc = ply_loader.load_ply(str(TEST_DATA_LOC / "1/lidar/PC_0.ply"))

    pc_inside = eval_utils.get_pc_inside_bbox(pc, bbox)

    assert len(pc_inside) == 3


# def test_leave_only_roi_region() -> None:
#     """
#         test leave_only_roi_region function
#         (lidar_pts,egovehicle_to_city_se3,ground_removal_method, city_name='MIA')
#         """
#     pc = ply_loader.load_ply(TEST_DATA_LOC / "1/lidar/PC_0.ply")
#     pose_data = read_json_file(TEST_DATA_LOC / "1/poses/city_SE3_egovehicle_0.json")
#     rotation = np.array(pose_data["rotation"])
#     translation = np.array(pose_data["translation"])
#     ego_R = quat2rotmat(rotation)
#     ego_t = translation
#     egovehicle_to_city_se3 = SE3(rotation=ego_R, translation=ego_t)
#     pts = eval_utils.leave_only_roi_region(pc, egovehicle_to_city_se3, ground_removal_method="map")
#     # might need toy data for map files


# def test_evaluation_track() -> None:
#     """
#         test eval_tracks function
#         """

#     # for testing, consider all point
#     eval_tracking.min_point_num = 0
#     centroid_methods = ["label_center", "average"]
#     temp_output = TEST_DATA_LOC / "tmp"
#     out_file = open(temp_output, "w+")

#     log = "1"
#     log_location = TEST_DATA_LOC / log

#     # sanity check, gt and results exactly the same
#     track_results_location = log_location / "per_sweep_annotations_amodal"
#     cm = centroid_methods[1]
#     eval_tracking.eval_tracks(track_results_location, os.fspath(log_location), D_MIN, D_MAX, out_file, cm)
#     out_file.close()
