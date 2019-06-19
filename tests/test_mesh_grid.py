import numpy as np
from argoverse.utils.mesh_grid import get_mesh_grid_as_point_cloud


def test_get_mesh_grid_as_point_cloud_3x3square():
    """
    Sample a regular grid and return the (x,y) coordinates
    of the sampled points.
    """

    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = -1  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 4  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = get_mesh_grid_as_point_cloud(min_x, max_x, min_y, max_y, downsample_factor=1.0)

    assert pts.shape == (9, 2)
    gt_pts = np.array(
        [
            [-3.0, 2.0],
            [-2.0, 2.0],
            [-1.0, 2.0],
            [-3.0, 3.0],
            [-2.0, 3.0],
            [-1.0, 3.0],
            [-3.0, 4.0],
            [-2.0, 4.0],
            [-1.0, 4.0],
        ]
    )

    assert np.allclose(gt_pts, pts)


def test_get_mesh_grid_as_point_cloud_3x2rect():
    """
    Sample a regular grid and return the (x,y) coordinates
    of the sampled points.
    """
    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = -1  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 3  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = get_mesh_grid_as_point_cloud(min_x, max_x, min_y, max_y, downsample_factor=1.0)

    assert pts.shape == (6, 2)
    gt_pts = np.array([[-3.0, 2.0], [-2.0, 2.0], [-1.0, 2.0], [-3.0, 3.0], [-2.0, 3.0], [-1.0, 3.0]])

    assert np.allclose(gt_pts, pts)


def test_get_mesh_grid_as_point_cloud_single_pt():
    """
    Sample a regular grid and return the (x,y) coordinates
    of the sampled points.
    """
    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = -3  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 2  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = get_mesh_grid_as_point_cloud(min_x, max_x, min_y, max_y, downsample_factor=1.0)

    assert pts.shape == (1, 2)
    gt_pts = np.array([[-3.0, 2.0]])

    assert np.allclose(gt_pts, pts)


def test_get_mesh_grid_as_point_cloud_downsample():
    """
    Sample a regular grid and return the (x,y) coordinates
    of the sampled points.
    """

    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = 0  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 5  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = get_mesh_grid_as_point_cloud(min_x, max_x, min_y, max_y, downsample_factor=2.0)

    assert pts.shape == (4, 2)

    gt_pts = [[-3.0, 2.0], [0.0, 2.0], [-3.0, 5.0], [0.0, 5.0]]
    assert np.allclose(gt_pts, pts)
