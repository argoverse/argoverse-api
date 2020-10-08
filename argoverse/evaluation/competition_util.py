"""Utilities for submitting to Argoverse tracking and forecasting competitions"""

import json
import math
import os
import shutil
import tempfile
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import quaternion
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.cluster.dbscan_ import DBSCAN

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.utils.se3 import SE3

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
        data: a dictionary of trajectory, with the key being the sequence ID. For each sequence, the
              trajectory should be stored in a (9,30,2) np.ndarray
        output_path: path to the output directory to store the output h5 file
        filename: to be used as the name of the file
        probabilities (optional) : normalized probability for each trajectory

    Returns:

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

    Returns:

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
    # poly in polygon format

    bbox = poly.minimum_rotated_rectangle

    x = bbox.exterior.xy[0]
    y = bbox.exterior.xy[1]
    z = np.array([z for _, _, z in poly.exterior.coords])

    # z = poly.exterior.xy[2]

    d1 = dist((x[0], y[0]), (x[1], y[1]))
    d2 = dist((x[1], y[1]), (x[2], y[2]))

    width = min(d1, d2)
    length = max(d1, d2)

    if max(d1, d2) == d2:
        unit_v = unit_vector((x[1], y[1]), (x[2], y[2]))
    else:
        unit_v = unit_vector((x[0], y[0]), (x[1], y[1]))

    angle = math.atan2(unit_v[1], unit_v[0])

    height = max(z) - min(z)

    # translation = center
    center = np.array([bbox.centroid.xy[0][0], bbox.centroid.xy[1][0], min(z) + height / 2])

    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    q = quaternion.from_rotation_matrix(R)

    return ObjectLabelRecord(
        quaternion=quaternion.as_float_array(q),
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
        json_data = {
            "center": {
                "x": label.translation[0],
                "y": label.translation[1],
                "z": label.translation[2],
            },
            "rotation": {
                "x": label.quaternion[0],
                "y": label.quaternion[1],
                "z": label.quaternion[2],
                "w": label.quaternion[3],
            },
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
