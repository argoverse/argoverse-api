from typing import Final, Tuple

import numpy as np
import numpy.typing as npt

INDEX_KEYS: Final[Tuple[str, str, str, str]] = (
    "split",
    "log_id",
    "record_type",
    "tov_ns",
)

FOV: Final[npt.NDArray[float]] = np.array([-25.0 / 180.0 * np.pi, 15 / 180.0 * np.pi])

EGO_SE3_LIDAR_UP: Final[np.ndarray] = np.array(
    [[1, 0, 0, 1.35018], [0, 1, 0, 0], [0, 0, 1, 1.64042], [0, 0, 0, 1]]
)
