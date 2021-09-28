from typing import Final, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from argoverse.typing.numpy import NDArray

CUBOID_VERTICES: Final[NDArray] = np.array(
    [
        [-1, -1, -1],  # 0
        [+1, -1, -1],  # 1
        [+1, +1, -1],  # 2
        [-1, +1, -1],  # 3
        [-1, +1, +1],  # 4
        [+1, +1, +1],  # 5
        [+1, -1, +1],  # 6
        [-1, -1, +1],  # 7
    ],
    dtype=np.double,
)


def cuboid2poly(cuboid: NDArray) -> NDArray:
    nvert = CUBOID_VERTICES.shape[0]
    polygon = CUBOID_VERTICES[None]

    t = cuboid[..., :3]
    dims = cuboid[..., 3:6]

    polygon = polygon * dims[:, None] / 2

    quat = cuboid[..., [-3, -2, -1, -4]].repeat(nvert, axis=0)
    rot = R.from_quat(quat)

    polygon = rot.apply(polygon.reshape(-1, 3)).reshape(-1, nvert, 3) + t[:, None]
    return polygon


def filter_point_cloud_to_bbox_3D_vectorized(
    bbox: NDArray, pc_raw: NDArray
) -> Tuple[NDArray, NDArray]:
    r"""
    Args:
       bbox: Numpy array pf shape (8,3) representing 3d cuboid vertices, ordered
                as shown below.
       pc_raw: Numpy array of shape (N,3), representing a point cloud
    Returns:
       segment: Numpy array of shape (K,3) representing 3d points that fell
                within 3d cuboid volume.
       is_valid: Numpy array of shape (N,) of type bool
    https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
    ::
            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.
    """
    # get 3 principal directions (edges) of the cuboid
    u = bbox[2] - bbox[6]
    v = bbox[2] - bbox[3]
    w = bbox[2] - bbox[1]

    valid_u1 = np.logical_and(
        u.dot(bbox[2]) <= pc_raw.dot(u), pc_raw.dot(u) <= u.dot(bbox[6])
    )
    valid_v1 = np.logical_and(
        v.dot(bbox[2]) <= pc_raw.dot(v), pc_raw.dot(v) <= v.dot(bbox[3])
    )
    valid_w1 = np.logical_and(
        w.dot(bbox[2]) <= pc_raw.dot(w), pc_raw.dot(w) <= w.dot(bbox[1])
    )

    valid_u2 = np.logical_and(
        u.dot(bbox[2]) >= pc_raw.dot(u), pc_raw.dot(u) >= u.dot(bbox[6])
    )
    valid_v2 = np.logical_and(
        v.dot(bbox[2]) >= pc_raw.dot(v), pc_raw.dot(v) >= v.dot(bbox[3])
    )
    valid_w2 = np.logical_and(
        w.dot(bbox[2]) >= pc_raw.dot(w), pc_raw.dot(w) >= w.dot(bbox[1])
    )

    valid_u = np.logical_or(valid_u1, valid_u2)
    valid_v = np.logical_or(valid_v1, valid_v2)
    valid_w = np.logical_or(valid_w1, valid_w2)

    is_valid = np.logical_and(np.logical_and(valid_u, valid_v), valid_w)
    segment_pc = pc_raw[is_valid]
    return segment_pc, is_valid
