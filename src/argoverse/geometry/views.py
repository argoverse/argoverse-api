from typing import Tuple

import numpy as np

from argoverse.geometry.conversions import cart2sph
from argoverse.typing.numpy import NDArray, NDArrayInt
from argoverse.utils.constants import NAN, PI


def pos2range(
    pos: NDArray,
    fov: NDArray,
    dims: NDArrayInt = np.array([64, 1024]),
) -> Tuple[NDArray, NDArray]:
    fov_bottom, fov_top = np.abs(fov).transpose()

    az, el, r = cart2sph(*pos.transpose())

    v = 0.5 * (-az / PI + 1.0)
    u = 1.0 - (el + fov_bottom) / (fov_bottom + fov_top)

    perm = np.argsort(r)

    uv = np.stack((u, v), axis=-1)[perm] * dims
    uv = np.clip(uv, 0, dims - 1).astype(int)

    range_im = np.full(dims, NAN)
    pos_im = np.full((dims.tolist() + [3]), NAN)

    u, v = uv.transpose()
    range_im[u, v] = r[perm]
    pos_im[u, v] = pos[perm]
    return range_im, pos_im
