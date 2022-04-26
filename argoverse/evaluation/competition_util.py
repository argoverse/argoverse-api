"""Utilities for submitting to Argoverse tracking and forecasting competitions"""

import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import yaw_to_quaternion3d

TYPE_LIST = Union[List[np.ndarray], np.ndarray]


def generate_forecasting_h5(
    data: Dict[int, TYPE_LIST],
    output_path: str,
    filename: str = "argoverse_forecasting_baseline",
    probabilities: Optional[Dict[int, List[float]]] = None,
) -> None:
    """
    Helper function to generate the result h5 file for argoverse forecasting challenge

    Args:
        data: a dictionary of trajectories, with the key being the sequence ID, and value being
              predicted trajectories for the sequence, stored in a (n,30,2) np.ndarray.
              "n" can be any number >=1. If probabilities are provided, the evaluation server
              will use the top-K most likely forecasts for any top-K metric. If probabilities
              are unavailable, the first-K trajectories will be evaluated instead. Each
              predicted trajectory should consist of 30 waypoints.
        output_path: path to the output directory to store the output h5 file
        filename: to be used as the name of the file
        probabilities (optional) : normalized probability for each trajectory
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf = h5py.File(os.path.join(output_path, filename + ".h5"), "w")
    future_frames = 30
    d_all: List[np.ndarray] = []
    counter = 0
    for key, value in data.items():
        print("\r" + str(counter + 1) + "/" + str(len(data)), end="")

        if isinstance(value, List):
            value = np.array(value)
        assert value.shape[1:3] == (
            future_frames,
            2,
        ), f"ERROR: the data should be of shape (n,30,2), currently getting {value.shape}"

        n = value.shape[0]
        len_val = len(value)
        value = value.reshape(n * future_frames, 2)
        if probabilities is not None:
            assert key in probabilities.keys(), f"missing probabilities for sequence {key}"
            assert (
                len(probabilities[key]) == len_val
            ), f"mismatch sequence and probabilities len for {key}: {len(probabilities[key])} !== {len_val}"
            # assert np.isclose(np.sum(probabilities[key]), 1), "probabilities are not normalized"

            d = np.array(
                [
                    [
                        key,
                        np.float32(x),
                        np.float32(y),
                        probabilities[key][int(np.floor(i / future_frames))],
                    ]
                    for i, (x, y) in enumerate(value)
                ]
            )
        else:
            d = np.array([[key, np.float32(x), np.float32(y)] for x, y in value])

        d_all.append(d)
        counter += 1

    d_all = np.concatenate(d_all, 0)
    hf.create_dataset("argoverse_forecasting", data=d_all, compression="gzip", compression_opts=9)
    hf.close()


def generate_tracking_zip(input_path: str, output_path: str, filename: str = "argoverse_tracking") -> None:
    """
    Helper function to generate the result zip file for argoverse tracking challenge

    Args:
        input path: path to the input directory which contain per_sweep_annotations_amodal/
        output_path: path to the output directory to store the output zip file
        filename: to be used as the name of the file

    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dirpath = tempfile.mkdtemp()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for log_id in os.listdir(input_path):
        if log_id.startswith("."):
            continue
        shutil.copytree(
            os.path.join(input_path, log_id, "per_sweep_annotations_amodal"),
            os.path.join(dirpath, log_id, "per_sweep_annotations_amodal"),
        )

    shutil.make_archive(os.path.join(output_path, "argoverse_tracking"), "zip", dirpath)
    shutil.rmtree(dirpath)


def generate_stereo_zip(data_dir: Path, output_dir: Path) -> None:
    """
    Helper function to generate the result zip file for the argoverse stereo challenge.

    Args:
        data_dir: Path to the directory containing the disparity predictions.
        output_dir: Path to the output directory to store the output zip file.
    """

    output_dir.mkdir(exist_ok=True, parents=True)

    num_test_logs = 15
    num_pred_logs = len([path for path in data_dir.iterdir() if path.is_dir()])

    assert (
        num_test_logs == num_pred_logs
    ), f"ERROR: Found {num_pred_logs} logs in the input dir {data_dir}. It must have {num_test_logs}."

    for log_path in tqdm(list(data_dir.iterdir())):
        if not log_path.is_dir():
            continue

        disparity_map_fpaths = log_path.glob("*.png")
        for disparity_map_fpath in disparity_map_fpaths:
            disparity_map_pred = cv2.imread(str(disparity_map_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            assert disparity_map_pred.shape == (
                2056,
                2464,
            ), f"ERROR: The predicted disparity map should be of shape (2056, 2464) but got {disparity_map_pred.shape}."

            assert (
                disparity_map_pred.dtype == "uint16"
            ), f"ERROR: The predicted disparity map should be of type uint16 but got {disparity_map_pred.dtype}."

    report_fpath = data_dir / "model_analysis_report.txt"
    if not report_fpath.is_file():
        print(f"ERROR: Report file {report_fpath} not found! Please add it to the input folder.")

    print("Creating zip file for submission...")
    shutil.make_archive(str(output_dir / "stereo_output"), "zip", data_dir)

    print(f"Zip file ({output_dir}/stereo_output.zip) created succesfully. Please submit it to EvalAI for evaluation.")


def get_polygon_from_points(points: np.ndarray) -> Polygon:
    """
    function to generate (convex hull) shapely polygon from set of points

    Args:
        points: list of 2d coordinate points

    Returns:
        polygon: shapely polygon representing the results
    """
    points = points
    hull = ConvexHull(points)

    poly = []

    for simplex in hull.simplices:
        poly.append([points[simplex, 0][0], points[simplex, 1][0], points[simplex, 2][0]])
        poly.append([points[simplex, 0][1], points[simplex, 1][1], points[simplex, 2][1]])

        # plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    return Polygon(poly)


def get_rotated_bbox_from_points(points: np.ndarray) -> Polygon:
    """
    function to generate mininum_rotated_rectangle from list of point coordinate

    Args:
        points: list of 2d coordinate points

    Returns:
        polygon: shapely polygon representing the results
    """
    return get_polygon_from_points(points).minimum_rotated_rectangle


def unit_vector(pt0: Tuple[float, float], pt1: Tuple[float, float]) -> Tuple[float, float]:
    # returns an unit vector that points in the direction of pt0 to pt1
    dis_0_to_1 = math.sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, (pt1[1] - pt0[1]) / dis_0_to_1


def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def poly_to_label(poly: Polygon, category: str = "VEHICLE", track_id: str = "") -> ObjectLabelRecord:
    """Convert a Shapely Polygon to a 3d cuboid by estimating the minimum-bounding rectangle.

    Args:
        poly: Shapely polygon object representing a convex hull of an object
        category: object category to which object belongs, e.g. VEHICLE, PEDESTRIAN, etc
        track_id: unique identifier
    Returns:
        object representing a 3d cuboid
    """

    bbox = poly.minimum_rotated_rectangle

    x = bbox.exterior.xy[0]
    y = bbox.exterior.xy[1]
    z = np.array([z for _, _, z in poly.exterior.coords])

    # z = poly.exterior.xy[2]

    d1 = dist((x[0], y[0]), (x[1], y[1]))
    d2 = dist((x[1], y[1]), (x[2], y[2]))

    # assign orientation so that the rectangle's longest side represents the object's length
    width = min(d1, d2)
    length = max(d1, d2)

    if max(d1, d2) == d2:
        unit_v = unit_vector((x[1], y[1]), (x[2], y[2]))
    else:
        unit_v = unit_vector((x[0], y[0]), (x[1], y[1]))

    angle_rad = np.arctan2(unit_v[1], unit_v[0])
    q = yaw_to_quaternion3d(angle_rad)

    height = max(z) - min(z)

    # location of object in egovehicle coordinates
    center = np.array([bbox.centroid.xy[0][0], bbox.centroid.xy[1][0], min(z) + height / 2])

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )

    return ObjectLabelRecord(
        quaternion=q,
        translation=center,
        length=length,
        width=width,
        height=height,
        occlusion=0,
        label_class=category,
        track_id=track_id,
    )


def get_objects(clustering: DBSCAN, pts: np.ndarray, category: str = "VEHICLE") -> List[Tuple[np.ndarray, uuid.UUID]]:

    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    objects = []

    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            # noise
            continue
        class_member_mask = labels == k
        xyz = pts[class_member_mask & core_samples_mask]
        if category == "VEHICLE":
            if len(xyz) >= 20:
                poly = get_polygon_from_points(xyz)
                label = poly_to_label(poly, category="VEHICLE")
                if label.length < 7 and label.length > 1 and label.height < 2.5:
                    objects.append((xyz, uuid.uuid4()))
        elif category == "PEDESTRIAN":
            if len(xyz) >= 20:
                poly = get_polygon_from_points(xyz)
                label = poly_to_label(poly, category="PEDESTRIAN")
                if label.width < 1 and label.length < 1 and label.height > 1 and label.height < 2.5:
                    objects.append((xyz, uuid.uuid4()))
    return objects


def save_label(argoverse_data: ArgoverseTrackingLoader, labels: List[ObjectLabelRecord], idx: int) -> None:
    # save label data at index idx

    data_dir = argoverse_data.root_dir

    log = argoverse_data.current_log

    label_dir = os.path.join(data_dir, log, "per_sweep_annotations_amodal")

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    labels_json_data = []

    timestamp = argoverse_data.lidar_timestamp_list[idx]

    for label in labels:
        qw, qx, qy, qz = label.quaternion
        json_data = {
            "center": {
                "x": label.translation[0],
                "y": label.translation[1],
                "z": label.translation[2],
            },
            "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "length": label.length,
            "width": label.width,
            "height": label.height,
            "occlusion": 0,
            "tracked": True,
            "timestamp": timestamp,
            "label_class": label.label_class,
            "track_label_uuid": label.track_id,
        }
        labels_json_data.append(json_data)

    fn = f"tracked_object_labels_{timestamp}.json"
    with open(os.path.join(label_dir, fn), "w") as json_file:
        json.dump(labels_json_data, json_file)


def transform_xyz(xyz: np.ndarray, pose1: SE3, pose2: SE3) -> np.ndarray:
    # transform xyz from pose1 to pose2

    # convert to city coordinate
    city_coord = pose1.transform_point_cloud(xyz)

    return pose2.inverse_transform_point_cloud(city_coord)
