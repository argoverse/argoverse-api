from typing import Tuple

import numpy as np

from argoverse.typing.numpy import NDArray


def cart2sph(x: NDArray, y: NDArray, z: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    h = np.hypot(x, y)
    r = np.hypot(h, z)
    el = np.arctan2(z, h)
    az = np.arctan2(y, x)
    return az, el, r
