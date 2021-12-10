"""Raster visualization tools."""
import numpy as np
from numba import njit


# TODO: add njit support.
def pc2im(lidar: np.ndarray, voxel_resolution: np.ndarray, grid_size: np.ndarray) -> np.ndarray:

    # Grab the Cartesian coordinates (xyz).
    cart = lidar[..., :-1].copy()

    # If only xyz are provided, then assume intensity is 1.0.
    # Otherwise, use the provided intensity.
    if lidar.shape[-1] == 3:
        intensity = np.ones_like(lidar.shape[0])
    else:
        intensity = lidar[..., -1].copy()

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
    intensity = intensity[grid_boundary_reduction]

    # Create the raster image.
    im = np.zeros(voxel_grid_size + 1)
    for c, lidar_xyz in enumerate(indices):
        u = voxel_grid_size[0] - lidar_xyz[0]
        v = lidar_xyz[1]
        z = lidar_xyz[2]
        im[u, v, z] = intensity[c]

    # Take the mean intensity over the vertical axis.
    im = np.mean(im, axis=-1, keepdims=True) / im.max()

    # Invert the colors.
    im = (1 - im) * (im != 0)

    # Greyscale -> RGB
    return np.repeat(im, 3, axis=-1)
