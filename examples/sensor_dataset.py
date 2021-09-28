"""Example script for loading data from the Argoverse 2.0 sensor dataset."""
from pathlib import Path

import imageio
import numpy as np

from argoverse.datasets.sensor.constants import EGO_SE3_LIDAR_UP, FOV
from argoverse.datasets.sensor.dataset import SensorDataset
from argoverse.geometry.views import pos2range
from argoverse.io.loading import read_feather
from argoverse.ops.points import hom2pos, pos2hom
from argoverse.rendering.color import TURBO
from argoverse.typing.numpy import NDArray, NDArrayInt
from argoverse.utils.pathlib import HOME


def main(dirname: Path) -> None:
    """Entry function for exploring the Argoverse 2.0 dataset.

    Args:
        dirname (Path): Directory name.
    """
    dataset = SensorDataset(dirname)
    dims: NDArrayInt = np.array([128, 1024])
    for path in dataset.get_lidar_paths():
        datum: NDArray = read_feather(path, dataset.index_names).to_numpy()

        pos_lidar = ego2lidar(datum[:, :3])
        range_im, _ = pos2range(pos_lidar, fov=FOV, dims=dims)

        write_range_im(range_im)
        break


def ego2lidar(pos_ego: NDArray) -> NDArray:
    pos_lidar_h: NDArray = (
        np.linalg.inv(EGO_SE3_LIDAR_UP) @ pos2hom(pos_ego).transpose()
    )
    return hom2pos(pos_lidar_h.transpose())


def write_range_im(range_im: NDArray) -> None:
    valid = ~np.isnan(range_im)
    INV_TURBO: NDArray = np.flipud(TURBO)

    range_im = INV_TURBO[range_im.astype(np.uint8)][..., :3]
    range_im = np.multiply(range_im, 255).astype(np.uint8)
    range_im[~valid] = 0

    imageio.imwrite("range_image.png", range_im)


if __name__ == "__main__":
    dirname = HOME / "data" / "datasets" / "argoverse-v2"
    main(dirname)
