"""Raster visualization tools."""
from typing import Dict, Final, List, Optional

import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial.transform import Rotation as R

from argoverse.utils.geometry import crop_points

AV2_CATEGORY_CMAP: Final[Dict[str, np.ndarray]] = {
    "REGULAR_VEHICLE": np.array([0.0, 255.0, 0.0]),
    "PEDESTRIAN": np.array([192.0, 255.0, 0.0]),
}


# Unit polygon (vertices in {-1, 0, 1}) with counter-clockwise (CCW) winding order.
# +---+---+
# | 1 | 0 |
# +---+---+
# | 2 | 3 |
# +---+---+
UNIT_POLYGON_2D: Final[np.ndarray] = np.array(
    [[+1.0, +1.0, 0.0], [-1.0, +1.0, 0.0], [-1.0, -1.0, 0.0], [+1.0, -1.0, 0.0]]
)

UNIT_POLYGON_2D_EDGES: Final[np.ndarray] = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


# TODO: add njit support.
def pc2im(
    lidar_xyz: np.ndarray,
    voxel_resolution: np.ndarray,
    grid_size: np.ndarray,
    cmap: Optional[np.ndarray] = None,
) -> np.ndarray:

    # Default colormap to gray.
    if cmap is None:
        cmap = np.full((lidar_xyz.shape[0], 3), 128.0)
    cmap /= 255.0

    # If only xyz are provided, then assume intensity is 1.0.
    # Otherwise, use the provided intensity.
    if lidar_xyz.shape[-1] == 3:
        intensity = np.ones_like(lidar_xyz.shape[0])
    else:
        intensity = lidar_xyz[..., -1].copy()

    # Grab the Cartesian coordinates (xyz).
    cart = lidar_xyz[..., :-1].copy()

    # Move the origin to the center of the image.
    cart += np.divide(grid_size, 2)

    # Scale the Cartesian coordinates by the voxel resolution.
    indices = np.divide(cart, voxel_resolution).astype(int)

    # Compute the voxel grid size.
    voxel_grid_size = np.divide(grid_size, voxel_resolution).astype(int)

    # Crop point cloud to the region-of-interest.
    indices, grid_boundary_reduction = crop_points(indices, np.array([0, 0, 0]), voxel_grid_size)

    # Filter the indices and intensity values.
    indices = indices[grid_boundary_reduction]
    cmap = cmap[grid_boundary_reduction]
    intensity = intensity[grid_boundary_reduction]

    # Create the raster image.
    im_dims = np.concatenate((voxel_grid_size[:2] + 1, cmap.shape[1:2])) + 1
    im = np.zeros(im_dims)

    # Construct uv coordinates.
    u = voxel_grid_size[0] - indices[:, 0]
    v = voxel_grid_size[1] - indices[:, 1]

    npoints = indices.shape[0]
    for i in range(npoints):
        im[u[i], v[i], :3] = cmap[i]
        im[u[i], v[i], 3:4] += intensity[i]

    # Normalize the intensity.
    im[..., -1] /= im[..., -1].max()

    # Gamma correction.
    im[..., -1] = np.power(im[..., -1], 0.05)

    # Scale RGB by intensity.
    im[..., :3] *= im[..., -1:]

    # Map RGB in [0, 1] -> [0, 255].
    return np.multiply(im[..., :3], 255.0)


def overlay_annotations(
    im: np.ndarray,
    annotations: pd.DataFrame,
    voxel_resolution: np.ndarray,
    category_cmap: Dict[str, np.ndarray] = {},
) -> np.ndarray:

    # Return original image if no annotations exist.
    if annotations.shape[0] == 0:
        return im

    categories = annotations[["category"]].to_numpy().flatten().tolist()

    # Grab centers (xyz) of the annotations.
    center_xyz = annotations[["x", "y", "z"]].to_numpy()

    # Grab dimensions (length, width, height) of the annotations.
    dims_lwh = annotations[["length", "width", "height"]].to_numpy()

    # Construct unit polygons.
    scaled_polygons = UNIT_POLYGON_2D[None] * np.divide(dims_lwh[:, None], 2.0)

    # Get scalar last quaternions.
    # NOTE: SciPy follows scaler *last* while AV2 uses scaler *first* ordering.
    quat_xyzw = annotations[["qx", "qy", "qz", "qw"]].to_numpy()

    # Repeat transformations by number of polygon vertices to vectorize SO3 transformation in SciPy.
    quat_xyzw = np.repeat(quat_xyzw[:, None], scaled_polygons.shape[1], axis=1).reshape(-1, 4)

    # Get SO3 transformation.
    ego_SO3_obj = R.from_quat(quat_xyzw)

    # Apply ego_SO3_obj to the scaled polygons.
    polygons_xyz = ego_SO3_obj.apply(scaled_polygons.reshape(-1, 3)).reshape(-1, 4, 3)

    # Translate by the annotations centers.
    polygons_xyz += center_xyz[:, None]

    alpha = np.linspace(0, 1, 1000)[None, None, :, None]

    polygons_xyz = polygons_xyz[:, UNIT_POLYGON_2D_EDGES]
    polygons_xyz = polygons_xyz[..., 0:1, :] * alpha + polygons_xyz[..., 1:2, :] * (1 - alpha)

    polygons_xy = polygons_xyz[..., :2]
    polygons_xy /= voxel_resolution[..., :2]
    polygons_xy += np.divide(im.shape[:2], 2.0)
    polygons_xy = polygons_xy.astype(int)

    colors = np.zeros_like(polygons_xyz)
    for i, category in enumerate(categories):
        if category not in category_cmap:
            colors[i] = np.array([0.0, 0.0, 255.0])
        else:
            colors[i] = category_cmap[category]

    polygons_xy, grid_boundary_reduction = crop_points(polygons_xy, np.array([0, 0]), np.array(im.shape[:2]))
    polygons_xy = polygons_xy[grid_boundary_reduction]
    colors = colors[grid_boundary_reduction]

    polygons_xy = polygons_xy.reshape(-1, 2)
    u = im.shape[0] - polygons_xy[..., 0] - 1
    v = im.shape[1] - polygons_xy[..., 1] - 1
    im[u, v] = colors
    return im
