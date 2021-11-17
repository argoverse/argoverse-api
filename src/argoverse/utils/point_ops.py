import numpy as np

from argoverse.typing.numpy import NDArray


def pos2hom(pos: NDArray) -> NDArray:
    pos_h: NDArray = np.ones((pos.shape[0], 4), dtype=np.float64)
    pos_h[:, :3] = pos
    return pos_h


def hom2pos(pos_h: NDArray) -> NDArray:
    return pos_h[:, :3]
