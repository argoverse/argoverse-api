"""Raster visualization tools."""
from typing import Any
import numpy as np
from numba import njit


# TODO: add njit support.
def pc2im(
    lidar_xyz: np.ndarray,
    cmap: np.ndarray,
    voxel_resolution: np.ndarray,
    grid_size: np.ndarray,
    is_normalized: bool = False,
) -> np.ndarray:

    # Grab the Cartesian coordinates (xyz).
    cart = lidar_xyz[..., :-1].copy()

    # If only xyz are provided, then assume intensity is 1.0.
    # Otherwise, use the provided intensity.
    if lidar_xyz.shape[-1] == 3:
        intensity = np.ones_like(lidar_xyz.shape[0])
    else:
        intensity = lidar_xyz[..., -1].copy()

    # Move the origin to the center of the image.
    cart += np.divide(grid_size, 2)

    # Scale the Cartesian coordinates by the voxel resolution.
    indices = np.divide(cart, voxel_resolution).astype(int)

    # Compute the voxel grid size.
    voxel_grid_size = np.divide(grid_size, voxel_resolution).astype(int)

    # Crop point cloud to the region-of-interest.
    lower_boundary_condition = np.greater_equal(indices, 0)
    upper_boundary_condition = np.less(indices, voxel_grid_size)
    grid_boundary_reduction = np.logical_and(lower_boundary_condition, upper_boundary_condition).all(axis=-1)

    # Filter the indices and intensity values.
    indices = indices[grid_boundary_reduction]
    cmap = cmap[grid_boundary_reduction]
    intensity = intensity[grid_boundary_reduction]

    # Create the raster image.
    im_dims = np.concatenate((voxel_grid_size[:2], cmap.shape[1:2])) + 1
    im = np.zeros(im_dims)
    for c, xyz in enumerate(indices):
        u = voxel_grid_size[0] - xyz[0]
        v = voxel_grid_size[1] - xyz[1]

        im[u, v, :3] = cmap[c]
        im[u, v, 3:4] += intensity[c]

    im[..., -1] /= im[..., -1].max()
    im[..., -1] = np.power(im[..., -1], 0.2)
    return im


# def overlay_annotations()
