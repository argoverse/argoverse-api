# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import List, Optional

import numpy as np

from argoverse.utils import mayavi_wrapper
from argoverse.utils.mesh_grid import get_mesh_grid_as_point_cloud
from argoverse.visualization.mayavi_utils import (
    Figure,
    draw_mayavi_line_segment,
    plot_3d_clipped_bbox_mayavi,
    plot_points_3D_mayavi,
)


def populate_frustum_voxels(planes: List[np.ndarray], fig: Figure, axis_pair: str) -> Figure:
    """
    Generate grid in xy plane, and then treat it as grid in xz (ground) plane
    in camera coordinate system.

    Args:
        planes: list of length 5. Each list element is a Numpy array
            of shape (4,) representing the equation of a plane,
            e.g. (a,b,c,d) in ax+by+cz=d
        fig: Mayavi figure to draw on
        axis_pair: Either "xz" or "yz"

    Returns:
        Mayavi figure
    """
    sparse_xz_voxel_grid = get_mesh_grid_as_point_cloud(-20, 20, 0, 40, downsample_factor=0.1)
    sparse_voxel_grid = np.zeros((sparse_xz_voxel_grid.shape[0], 3))

    if axis_pair == "xz":
        sparse_voxel_grid[:, 0] = sparse_xz_voxel_grid[:, 0]
        sparse_voxel_grid[:, 2] = sparse_xz_voxel_grid[:, 1]
    elif axis_pair == "yz":
        sparse_voxel_grid[:, 1] = sparse_xz_voxel_grid[:, 0]
        sparse_voxel_grid[:, 2] = sparse_xz_voxel_grid[:, 1]

    # keep only the points that have signed distance > 0 (inside the frustum, plane
    # normals also point into the frustum)
    for plane in planes:
        signed_d = np.matmul(sparse_voxel_grid, plane[:3]) + plane[3]
        sparse_voxel_grid = sparse_voxel_grid[np.where(signed_d > 0)]

    plot_points_3D_mayavi(sparse_voxel_grid, fig, fixed_color=(1, 0, 0))
    return fig


def plot_frustum_planes_and_normals(
    planes: List[np.ndarray],
    cuboid_verts: Optional[np.ndarray] = None,
    near_clip_dist: float = 0.5,
) -> None:
    """
    Args:
        planes: list of length 5. Each list element is a Numpy array
            of shape (4,) representing the equation of a plane,
            e.g. (a,b,c,d) in ax+by+cz=d
        cuboid_verts: Numpy array of shape (N,3) representing
            cuboid vertices

    Returns:
        None
    """
    fig = mayavi_wrapper.mlab.figure(bgcolor=(1, 1, 1), size=(2000, 1000))  # type: ignore

    if cuboid_verts is not None:
        # fig = plot_bbox_3d_mayavi(fig, cuboid_verts)
        fig = plot_3d_clipped_bbox_mayavi(fig, planes, cuboid_verts)

    P = np.array([0.0, 0.0, 0.0])
    for i, plane in enumerate(planes):
        (a, b, c, d) = plane
        if i == 0:
            color = (1, 0, 0)  # red left
        elif i == 1:
            color = (0, 0, 1)  # blue right
        elif i == 2:
            color = (1, 1, 0)  # near yellow
            P = np.array([0.0, 0.0, near_clip_dist])
        elif i == 3:
            color = (0, 1, 0)  # low is green
        elif i == 4:
            color = (0, 1, 1)  # top is teal

        plane_pts = generate_grid_on_plane(a, b, c, d, P)
        fig = plot_points_3D_mayavi(plane_pts, fig, color)
        # plot the normals at (0,0,0.5) and normal vector (u,v,w) given by (a,b,c)
        mayavi_wrapper.mlab.quiver3d(  # type: ignore
            0,
            0,
            0.5,
            a * 1000,
            b * 1000,
            c * 1000,
            color=color,
            figure=fig,
            line_width=8,
        )

    # draw teal line at top below the camera
    pt1 = np.array([-5, 0, -5])
    pt2 = np.array([5, 0, -5])
    color = (0, 1, 1)

    draw_mayavi_line_segment(fig, [pt1, pt2], color=color, line_width=8)

    # draw blue line in middle
    pt1 = np.array([-5, 5, -5])
    pt2 = np.array([5, 5, -5])
    color = (0, 0, 1)
    draw_mayavi_line_segment(fig, [pt1, pt2], color=color, line_width=8)

    # draw yellow, lowest line (+y axis is down)
    pt1 = np.array([-5, 10, -5])
    pt2 = np.array([5, 10, -5])
    color = (1, 1, 0)
    draw_mayavi_line_segment(fig, [pt1, pt2], color=color, line_width=8)

    fig = populate_frustum_voxels(planes, fig, "xz")
    fig = populate_frustum_voxels(planes, fig, "yz")

    mayavi_wrapper.mlab.view(distance=200)  # type: ignore
    mayavi_wrapper.mlab.show()  # type: ignore


def get_perpendicular(n: np.ndarray) -> np.ndarray:
    """
    n guarantees that dot(n, getPerpendicular(n)) is zero, which is the
    orthogonality condition, while also keeping the magnitude of the vector
    as high as possible. Note that setting the component with the smallest
    magnitude to 0 also guarantees that you don't get a 0,0,0 vector as a
    result, unless that is already your input.

    Args:
        n: Numpy array of shape (3,)

    Returns:
        result: Numpy array of shape (3,)
    """
    # find smallest component
    i = np.argmin(n)

    # get the other two indices
    a = (i + 1) % 3
    b = (i + 2) % 3

    result = np.zeros(3)
    result[i] = 0.0
    result[a] = n[b]
    result[b] = -n[a]
    return result


def generate_grid_on_plane(a: float, b: float, c: float, d: float, P: np.ndarray, radius: float = 15) -> np.ndarray:
    """
    Args:
        a,b,c,d: Coefficients of ``ax + by + cz = d`` defining plane
        P: Numpy array of shape (3,) representing point on the plane
        radius: Radius (default 15)

    Returns:
        plane_pts: Numpy array of shape (N,3) with points on the input plane
    """
    n = np.array([a, b, c])  # a,b,c from your equation
    perp = get_perpendicular(n)
    u = perp / np.linalg.norm(perp)
    v = np.cross(u, n)

    N = 100
    # delta and epsilon are floats:
    delta = radius / N  # N is how many points you want max in one direction
    epsilon = delta * 0.5

    n_pts = int((2 * radius + epsilon) / delta)
    pts = np.linspace(-radius, radius + epsilon, n_pts)

    plane_pts: List[float] = []
    for y in pts:
        for x in pts:
            # if (x*x+y*y < radius*radius): # only in the circle:
            plane_pts += [P + x * u + y * v]  # P is the point on the plane

    return np.array(plane_pts)
