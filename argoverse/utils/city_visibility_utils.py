# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import matplotlib.pyplot as plt
import numpy as np


def clip_point_cloud_to_visible_region(
    egovehicle_pts: np.ndarray, lidar_pts: np.ndarray, n_polar_bins: int = 100
) -> np.ndarray:
    """
    LiDAR points provide visibility information for a map point cloud in city coordinates.
    We bin the world into polar bins, and then if you've gone past the farthest LiDAR point
    in that orientation bin, we ignore such city points. Those city points could be lane
    centerlines or ROI boundary.

    Other options would be to fix a skinny rectangle around the egovehicle (long along road,
    and skinny along sidewalks), or to combine cells with <1.41 distance to each other into
    a polygon, growing the region with DBSCAN. Could also vectorize this.

    We loop through 2 pi radians in n polar bins to find the closest lidar return.
    Arctan has range [-pi, pi], so we start our loop there.

    Args:
       egovehicle_pts: 3d points in city coordinate fram
       lidar_pts: Array of LiDAR returns
       n_polar_bins: number of bins to discretize the unit circle with

    Returns:
       egovehicle_pts
    """
    angle_values = np.linspace(-np.pi, np.pi, n_polar_bins)
    for i, _ in enumerate(angle_values):
        min_angle = angle_values[i]
        max_angle = angle_values[(i + 1) % n_polar_bins]

        # find all roi points in this bin
        egovehicle_pt_angles = np.arctan2(egovehicle_pts[:, 1], egovehicle_pts[:, 0])
        egovehicle_pt_bools = np.logical_and(egovehicle_pt_angles >= min_angle, egovehicle_pt_angles < max_angle)

        # find all lidar points in this bin
        lidar_pt_angles = np.arctan2(lidar_pts[:, 1], lidar_pts[:, 0])
        lidar_pt_bools = np.logical_and(lidar_pt_angles >= min_angle, lidar_pt_angles < max_angle)
        bin_lidar_pts = lidar_pts[lidar_pt_bools]

        if len(bin_lidar_pts) == 0:
            continue

        # dist to farthest lidar point
        max_visible_dist = np.amax(np.linalg.norm(bin_lidar_pts[:, :2], axis=1))

        # if the roi point is farther than the farthest
        invalid_egovehicle_bools = np.linalg.norm(egovehicle_pts[:, :2], axis=1) > max_visible_dist
        invalid_egovehicle_bools = np.logical_and(egovehicle_pt_bools, invalid_egovehicle_bools)

        visualize = False
        if visualize:
            viz_polar_bin_contents(
                bin_lidar_pts,
                egovehicle_pts[invalid_egovehicle_bools],
                filename=f"polar_bin_{i}.jpg",
            )

        egovehicle_pts = egovehicle_pts[np.logical_not(invalid_egovehicle_bools)]

    return egovehicle_pts


def viz_polar_bin_contents(bin_lidar_pts: np.ndarray, invalid_egovehicle_pts: np.ndarray, filename: str) -> None:
    """
    Visualize what the utility is doing within each polar bin.

    Args:
       bin_lidar_pts: array
       invalid_egovehicle_pts: array
    """
    # visualize the contents of this polar bin
    plt.scatter(bin_lidar_pts[:, 0], bin_lidar_pts[:, 1], 10, marker=".", color="b")
    plt.scatter(
        invalid_egovehicle_pts[:, 0],
        invalid_egovehicle_pts[:, 1],
        10,
        marker=".",
        color="r",
    )
    plt.axis("equal")
    plt.savefig(filename)
    plt.close("all")
