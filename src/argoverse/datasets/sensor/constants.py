from typing import Final, List, Tuple

import numpy as np

RING_CAMS: Final[Tuple[str, ...]] = (
    "ring_rear_left",
    "ring_side_left",
    "ring_front_left",
    "ring_front_center",
    "ring_front_right",
    "ring_side_right",
    "ring_rear_right",
)

STEREO_CAMS: Final[Tuple[str, ...]] = (
    "stereo_front_left",
    "stereo_front_right",
)

CAMS: Final[Tuple[str, ...]] = RING_CAMS + STEREO_CAMS

INDEX_KEYS: Final[Tuple[str, str, str, str]] = (
    "split",
    "log_id",
    "record_type",
    "tov_ns",
)

ATTR_PATTERNS: Final[Tuple[str, ...]] = (
    "sensors/lidar/*.feather",
    "sensors/cameras/*/*.jpg",
)

CUBOID_COLS: Final[List[str]] = [
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "qw",
    "qx",
    "qy",
    "qz",
]

FOV: Final[np.ndarray] = np.array([-25.0 / 180.0 * np.pi, 15 / 180.0 * np.pi])

EGO_SE3_LIDAR_UP: Final[np.ndarray] = np.array(
    [[1, 0, 0, 1.35018], [0, 1, 0, 0], [0, 0, 1, 1.64042], [0, 0, 0, 1]]
)
