# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Utilities used in evaluation of performance."""

from typing import Any, Dict, List, Tuple

import numpy as np

from argoverse.utils.transform import quat2rotmat

# Label dictionary should be of the form {"center": {"x": 0.0, "y": 0.0, "z": 0.0},
#                                         "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
#                                         "height": 0.0, "width": 0.0, "depth": 0.0}
_LabelType = Dict[str, Any]


def get_pc_inside_bbox(pc_raw: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Get part of raw point cloud inside a given bounding box.

    Args:
        pc_raw: The raw point cloud
        bbox: The bounding box to restrict into

    Returns:
        The part of the point cloud inside the bounding box

    """

    U_lst = []
    V_lst = []
    W_lst = []
    P1_lst = []
    P2_lst = []
    P4_lst = []
    P5_lst = []

    u = bbox[1] - bbox[0]
    v = bbox[2] - bbox[0]
    w = np.zeros((3, 1))
    w[2, 0] += bbox[3]

    p5 = w + bbox[0]

    U_lst.append(u[0:3, 0])
    if len(U_lst) == 0:
        return np.array([])

    V_lst.append(v[0:3, 0])
    W_lst.append(w[0:3, 0])
    P1_lst.append(bbox[0][0:3, 0])
    P2_lst.append(bbox[1][0:3, 0])
    P4_lst.append(bbox[2][0:3, 0])
    P5_lst.append(p5[0:3, 0])

    U = np.array(U_lst)
    W = np.array(W_lst)
    V = np.array(V_lst)
    P1 = np.array(P1_lst)
    P2 = np.array(P2_lst)
    P4 = np.array(P4_lst)
    P5 = np.array(P5_lst)

    dot1 = np.matmul(U, pc_raw.transpose(1, 0))
    dot2 = np.matmul(V, pc_raw.transpose(1, 0))
    dot3 = np.matmul(W, pc_raw.transpose(1, 0))
    u_p1 = np.tile((U * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    v_p1 = np.tile((V * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    w_p1 = np.tile((W * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    u_p2 = np.tile((U * P2).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    v_p4 = np.tile((V * P4).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    w_p5 = np.tile((W * P5).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)

    flag = np.logical_and(
        np.logical_and(in_between_matrix(dot1, u_p1, u_p2), in_between_matrix(dot2, v_p1, v_p4)),
        in_between_matrix(dot3, w_p1, w_p5),
    )

    return pc_raw[flag[0, :]]


def label_to_bbox(label: _LabelType) -> Tuple[np.ndarray, float]:
    """Convert a label into a parameterized bounding box that lives in the ego-vehicle
        coordinate frame.

    Args:
        label: _LabelType

    Returns:
        bbox: Numpmy array for bounding box itself
        orientation: angle in radians, representing bounding box yaw

    """
    length = label["length"]
    width = label["width"]
    height = label["height"]

    p0 = np.array([-length / 2, -width / 2, -height / 2])[:, np.newaxis]
    p1 = np.array([+length / 2, -width / 2, -height / 2])[:, np.newaxis]
    p2 = np.array([-length / 2, +width / 2, -height / 2])[:, np.newaxis]

    bbox = np.array([p0, p1, p2, height], dtype=object)

    R = quat2rotmat(
        (
            label["rotation"]["w"],
            label["rotation"]["x"],
            label["rotation"]["y"],
            label["rotation"]["z"],
        )
    )
    t = np.array([label["center"]["x"], label["center"]["y"], label["center"]["z"]])[:, np.newaxis]

    v = np.array([1, 0, 0])[:, np.newaxis]
    orientation = np.matmul(R, v)
    orientation = np.arctan2(orientation[1, 0], orientation[0, 0])

    return transform_bounding_box_3d(bbox, R, t), orientation


def transform_bounding_box_3d(bbox: np.ndarray, R: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
    """Transform bounding box with rotation and translation.

    Args:
        bbox: The bounding box
        R: The rotation transformation
        t: The translation transformation

    Returns:
        The transformed bounding box

    """

    p0 = np.matmul(R, bbox[0]) + t
    p1 = np.matmul(R, bbox[1]) + t
    p2 = np.matmul(R, bbox[2]) + t

    return [p0, p1, p2, bbox[3]]


def in_between_matrix(x: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Element-by-element check to see if x_ij is between v1_ij and v2_ij, without knowing v1 > v2 order.

    Args:
        x: matrix to check if is in bounds
        v1: elements comprising one side of range check
        v2: elements comprising other side of range check

    Returns:
        Matrix of whether x_ij is between v1_ij and v2_ij

    """

    return np.logical_or(np.logical_and(x <= v1, x >= v2), np.logical_and(x <= v2, x >= v1))
