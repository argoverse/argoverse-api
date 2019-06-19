# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import math
from typing import List

import numpy as np

LIDAR_RANGE = 250
GRID_DIST = 0.4
HEIGHT_VARIANCE_THRESHOLD = 0.001
HEIGHT_MEAN_THRESHOLD = -1
NUM_ANGLE_BINS = 2000


def filter_ground_pts_polar_grid_mean_var(lidar_pts: np.ndarray) -> np.ndarray:
    """
    We divide the world into polar voxels.
    We aggregate the height statistics of all of the points that fall into each polar voxel.
    If the mean is below a threshold, we call it a ground voxel.
    If the z-axis variance is very low, we also call it a ground voxel.

                Var(X) = E[X^2] - E[X]^2
    Args:
        lidar_pts: NumPy n-d array of shape (n,3)
    Returns:
        non_ground_lidar_pts: NumPy n-d array of shape (n,3)
    """
    print("Total number of points (before filtering): ", lidar_pts.shape)
    non_ground_lidar_pts: List[List[List[np.ndarray]]] = []

    xyz_mean = np.mean(lidar_pts, axis=0)

    # Zero-center the point cloud because we compute statistics around (0,0,0)-centered polar grid
    lidar_pts -= xyz_mean

    num_radial_bins = int(LIDAR_RANGE / GRID_DIST)
    angle_increment = 2 * math.pi / NUM_ANGLE_BINS

    ang_voxel_mean = np.zeros((NUM_ANGLE_BINS, num_radial_bins))
    ang_voxel_variance = np.zeros((NUM_ANGLE_BINS, num_radial_bins))
    num_elements_per_bin = np.zeros((NUM_ANGLE_BINS, num_radial_bins))
    pts_per_bin: List[List[List[np.ndarray]]] = [[[] for _ in range(num_radial_bins)] for _ in range(NUM_ANGLE_BINS)]

    for i in range(lidar_pts.shape[0]):
        x = lidar_pts[i, 0]
        y = lidar_pts[i, 1]
        z = lidar_pts[i, 2]
        dist_away = math.sqrt(x ** 2 + y ** 2)
        angle_rad = np.arctan2(y, x)
        if angle_rad <= 0:
            angle_rad += 2 * math.pi
        radial_bin = int(math.floor(dist_away / GRID_DIST))
        angle_bin = int(math.floor(angle_rad / angle_increment))

        ang_voxel_mean[angle_bin, radial_bin] += z
        ang_voxel_variance[angle_bin, radial_bin] += z ** 2
        num_elements_per_bin[angle_bin, radial_bin] += 1.0
        pts_per_bin[angle_bin][radial_bin].append(lidar_pts[i, :])

    for i in range(NUM_ANGLE_BINS):
        for j in range(num_radial_bins):
            if len(pts_per_bin[i][j]) > 0:
                ang_voxel_mean[i, j] /= num_elements_per_bin[i, j]
                ang_voxel_variance[i, j] = (ang_voxel_variance[i, j] / num_elements_per_bin[i, j]) - ang_voxel_mean[
                    i, j
                ] ** 2

                if (ang_voxel_mean[i, j] > HEIGHT_MEAN_THRESHOLD) or (
                    ang_voxel_variance[i, j] > HEIGHT_VARIANCE_THRESHOLD
                ):
                    non_ground_lidar_pts += pts_per_bin[i][j]

    non_ground_lidar_pts_np = np.array(non_ground_lidar_pts)
    print("Number of non-ground points: ", non_ground_lidar_pts_np.shape)

    non_ground_lidar_pts += xyz_mean

    return non_ground_lidar_pts
