"""Example script for loading data from the Argoverse 2.0 sensor dataset."""
from pathlib import Path

import imageio
import numpy as np
from polars import col

from argoverse.datasets.sensor.constants import (CUBOID_COLS, EGO_SE3_LIDAR_UP,
                                                 FOV, INDEX_KEYS)
from argoverse.datasets.sensor.dataset import SensorDataset
from argoverse.geometry.polygon import (
    cuboid2poly, filter_point_cloud_to_bbox_3D_vectorized)
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
    
    # Initialize sensor dataset.
    dataset = SensorDataset(dirname)

    # Range image dimensions.
    dims: NDArrayInt = np.array([64, 1024])

    # Iterate over lidar paths.
    for path in dataset.get_lidar_paths():

        # Test labels are hidden!
        if "test" in path:
            continue

        # Load feather file.
        datum: NDArray = read_feather(path, dataset.index_names).to_numpy()

        # (x,y,z) in ego frame -> (x,y,z) in lidar frame.
        pos_lidar = ego2lidar(datum[:, :3])

        # Construct range image.
        range_im, _ = pos2range(pos_lidar, fov=FOV, dims=dims)

        # Get labels path.
        labels_path = Path(path.replace("lidar", "labels"))

        # Ensure that labels exist for this sweep.
        if not labels_path.exists():
            continue

        # Read labels.
        labels = read_feather(labels_path, INDEX_KEYS)

        # Filter labels for REGULAR VEHICLEs.
        labels = labels.filter(col("label_class") == "REGULAR_VEHICLE")

        # Get translation + pose.
        cuboids = labels[CUBOID_COLS].to_numpy()

        # Transform cuboid into polygons.
        polys_ego = cuboid2poly(cuboids)

        # Move polygons from ego frame to lidar frame.
        polys = ego2lidar(polys_ego.reshape(-1, 3)).reshape(-1, 8, 3)

        # Compute polygon interior points.
        interior_pts = np.concatenate(
            [
                filter_point_cloud_to_bbox_3D_vectorized(poly, pos_lidar)[0]
                for poly in polys
            ]
        )

        # Sweep to range image view.
        range_im, _ = pos2range(pos_lidar, fov=FOV, dims=dims)

        # Interior points to range image segmentation.
        range_im_seg, _ = pos2range(interior_pts, fov=FOV, dims=dims)

        # Hacky way to visualize the segments.
        mask = ~np.isnan(range_im_seg)
        range_im[mask] = 236.0

        # Write image.
        write_range_im(range_im, "range-image.png")
        break


def ego2lidar(pos_ego: NDArray) -> NDArray:
    pos_lidar_h: NDArray = (
        np.linalg.inv(EGO_SE3_LIDAR_UP) @ pos2hom(pos_ego).transpose()
    )
    return hom2pos(pos_lidar_h.transpose())


def write_range_im(range_im: NDArray, fname: str) -> None:
    valid = ~np.isnan(range_im)
    INV_TURBO: NDArray = np.flipud(TURBO)

    range_im = INV_TURBO[range_im.astype(np.uint8)][..., :3]
    range_im = np.multiply(range_im, 255).astype(np.uint8)
    range_im[~valid] = 0

    imageio.imwrite(fname, range_im)


if __name__ == "__main__":
    dirname = HOME / "data" / "datasets" / "argoverse-v2"
    main(dirname)
