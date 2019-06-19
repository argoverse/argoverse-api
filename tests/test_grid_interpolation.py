# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np
from argoverse.utils.grid_interpolation import interp_square_grid


def test_interp_square_grid_nearest_3to4():
    """
    "nearest" interpolation ight make sense for something binary valued like ROI.
    """
    grid_data = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    in_dim = 3
    out_dim = 4
    interp_type = "nearest"
    out_grid = interp_square_grid(grid_data, in_dim=in_dim, out_dim=out_dim, interp_type=interp_type)
    gt_interp_grid = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 2.0, 1.0], [1.0, 2.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    assert np.allclose(out_grid, gt_interp_grid)


def test_interp_square_grid_nearest_2to3():
    """
    "nearest" interpolation ight make sense for something binary valued like ROI.
    """
    grid_data = np.array([[1, 1], [1, 2]])
    in_dim = 2
    out_dim = 3
    interp_type = "nearest"
    out_grid = interp_square_grid(grid_data, in_dim=in_dim, out_dim=out_dim, interp_type=interp_type)
    gt_interp_grid = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    assert np.allclose(out_grid, gt_interp_grid)


def test_interp_square_grid_linear_3to5():
    """
    "linear" interpolation ight make sense for real valued numbers, e.g. ground heights.
    """
    in_grid = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    in_dim = 3
    out_dim = 5
    interp_type = "linear"
    out_grid = interp_square_grid(in_grid, in_dim=in_dim, out_dim=out_dim, interp_type=interp_type)
    gt_interp_grid = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.25, 1.5, 1.25, 1.0],
            [1.0, 1.5, 2.0, 1.5, 1.0],
            [1.0, 1.25, 1.5, 1.25, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(out_grid, gt_interp_grid)


def test_interp_square_grid_linear_2to3():
    """
    "linear" interpolation ight make sense for real valued numbers, e.g. ground heights.
    """
    grid_data = np.array([[1, 1], [1, 2]])
    in_dim = 2
    out_dim = 3
    interp_type = "linear"
    out_grid = interp_square_grid(grid_data, in_dim=in_dim, out_dim=out_dim, interp_type=interp_type)
    gt_interp_grid = [[1.0, 1.0, 1.0], [1.0, 1.25, 1.5], [1.0, 1.5, 2.0]]
    assert np.allclose(out_grid, gt_interp_grid)


def test_interp_square_grid_linear_2to3_neg():
    """
    "linear" interpolation ight make sense for real valued numbers, e.g. ground heights.
    """
    grid_data = np.array([[-50.0, -25.0], [-25.0, -25.0]])
    in_dim = 2
    out_dim = 3
    interp_type = "linear"
    out_grid = interp_square_grid(grid_data, in_dim=in_dim, out_dim=out_dim, interp_type=interp_type)
    gt_interp_grid = np.array([[-50.0, -37.5, -25.0], [-37.5, -31.25, -25.0], [-25.0, -25.0, -25.0]])

    assert np.allclose(out_grid, gt_interp_grid)


def test_interp_square_grid_linear_2to5_zero():
    """
    "linear" interpolation ight make sense for real valued numbers, e.g. ground heights.
    """
    grid_data = np.zeros((2, 2))
    in_dim = 2
    out_dim = 5
    interp_type = "linear"
    out_grid = interp_square_grid(grid_data, in_dim=in_dim, out_dim=out_dim, interp_type=interp_type)
    gt_interp_grid = np.zeros((5, 5))

    assert np.allclose(out_grid, gt_interp_grid)
