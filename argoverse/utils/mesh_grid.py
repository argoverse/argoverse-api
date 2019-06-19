# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Mesh grid utility functions."""

import numpy as np


def get_mesh_grid_as_point_cloud(
    min_x: int, max_x: int, min_y: int, max_y: int, downsample_factor: float = 1.0
) -> np.ndarray:
    """Sample regular grid and return the (x, y) coordinates.

    Args:
        min_x: Minimum x-coordinate of 2D grid
        max_x: Maximum x-coordinate of 2D grid
        min_y: Minimum y-coordinate of 2D grid
        max_y: Maximum y-coordinate of 2D grid

    Returns:
        pts: Array of shape (N, 2)
    """
    nx = max_x - min_x
    ny = max_y - min_y
    x = np.linspace(min_x, max_x, int((nx + 1) / downsample_factor))
    y = np.linspace(min_y, max_y, int((ny + 1) / downsample_factor))
    x_grid, y_grid = np.meshgrid(x, y)

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    x_grid = x_grid[:, np.newaxis]
    y_grid = y_grid[:, np.newaxis]

    pts = np.hstack([x_grid, y_grid])
    return pts
