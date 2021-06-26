"""Utilities for submitting to Argoverse tracking and forecasting competitions"""

import json
import math
import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

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


def get_polygon_from_points(points: np.ndarray) -> Polygon:
    """Convert a 3d point set to a Shapely polygon representing its convex hull.

    Args:
        points: list of 3d coordinate points

    Returns:
        polygon: shapely Polygon representing the points along the convex hull's boundary
    """
    points = points
    hull = ConvexHull(points)

    # `simplices` contains indices of points forming the simplical facets of the convex hull.
    poly_pts = hull.points[np.unique(hull.simplices)]
    return Polygon(poly_pts)


def get_rotated_bbox_from_points(points: np.ndarray) -> Polygon:
    """
    function to generate mininum_rotated_rectangle from list of point coordinate

    Args:
        points: list of 2d coordinate points

    Returns:
        polygon: shapely polygon representing the results
    """
    return get_polygon_from_points(points).minimum_rotated_rectangle


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
    centroid = bbox.centroid.coords[0]

    # exterior consists of of x and y values for bbox vertices [0,1,2,3,0], i.e. the first vertex is repeated as last
    x = np.array(bbox.exterior.xy[0]).reshape(5, 1)
    y = np.array(bbox.exterior.xy[1]).reshape(5, 1)

    v0, v1, v2, v3, _ = np.hstack([x, y])

    z = np.array([z for _, _, z in poly.exterior.coords])
    height = max(z) - min(z)

    d1 = np.linalg.norm(v0 - v1)
    d2 = np.linalg.norm(v1 - v2)

    # assign orientation so that the rectangle's longest side represents the object's length
    width = min(d1, d2)
    length = max(d1, d2)

    if d2 == length:
        # vector points from v1 -> v2
        v = v2 - v1
    else:
        # vector points from v0 -> v1
        v = v0 - v1

    # vector need not be unit length
    angle_rad = np.arctan2(v[1], v[0])
    q = yaw_to_quaternion3d(angle_rad)

    # location of object in egovehicle coordinates
    center = np.array([centroid[0], centroid[1], min(z) + height / 2])

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
    """ """
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
