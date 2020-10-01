# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interp_square_grid(
    grid_data: np.ndarray,
    in_dim: int = 200,
    out_dim: int = 30,
    interp_type: str = "linear",
) -> np.ndarray:
    """
    Interpolate a square grid
    Thousands of times faster than scipy.interpolate.interp2d.

    Args:
        grid_data: Numpy array of shape (in_dim,in_dim)
        in_dim: integer, representing length of a side of the input square grid
        out_dim: integer, representing length of a side of the output square grid
        interp_type: string, e.g. 'linear' or 'nearest' for interpolation scheme

    Returns:
        interpolated_grid: Numpy array of shape (out_dim,out_dim)
    """
    x = np.linspace(0, in_dim - 1, in_dim)
    y = np.linspace(0, in_dim - 1, in_dim)

    # define an interpolating function from this data:
    interpolating_function = RegularGridInterpolator((y, x), grid_data, method=interp_type)

    # Evaluate the interpolating function on a regular grid
    X = np.linspace(0, in_dim - 1, out_dim)
    Y = np.linspace(0, in_dim - 1, out_dim)
    x_grid, y_grid = np.meshgrid(X, Y)

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    x_grid = x_grid[:, np.newaxis]
    y_grid = y_grid[:, np.newaxis]

    pts = np.hstack([y_grid, x_grid])
    interpolated_grid = interpolating_function(pts)
    return interpolated_grid.reshape(out_dim, out_dim)
