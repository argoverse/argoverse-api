# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import datetime
import math
from typing import Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LinearRing, LineString, Point, Polygon

from argoverse.map_representation.lane_segment import LaneSegment

from . import mpl_plotting_utils
from .interpolate import interp_arc


def swap_left_and_right(
    condition: np.ndarray, left_centerline: np.ndarray, right_centerline: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Swap points in left and right centerline according to condition.

    Args:
       condition: Numpy array of shape (N,) of type boolean. Where true, swap the values in the left and
                   right centerlines.
       left_centerline: The left centerline, whose points should be swapped with the right centerline.
       right_centerline: The right centerline.

    Returns:
       left_centerline
       right_centerline
    """

    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline


def centerline_to_polygon(
    centerline: np.ndarray, width_scaling_factor: float = 1.0, visualize: bool = False
) -> np.ndarray:
    """
    Convert a lane centerline polyline into a rough polygon of the lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each side.
    We use this as the length of the hypotenuse of a right triangle, and compute the
    other two legs to find the scaled x and y displacement.

    Args:
       centerline: Numpy array of shape (N,2).
       width_scaling_factor: Multiplier that scales 3.8 meters to get the lane width.
       visualize: Save a figure showing the the output polygon.

    Returns:
       polygon: Numpy array of shape (2N+1,2), with duplicate first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3.8 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3.8 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    add_cond1 = np.logical_and(dx < 0, dy < 0)
    add_cond2 = np.logical_and(dx < 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    add_cond = np.logical_or(add_cond1, add_cond2)
    left_centerline, right_centerline = swap_left_and_right(subtract_cond, left_centerline, right_centerline)

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(neg_disp_cond, left_centerline, right_centerline)

    if visualize:
        plt.scatter(centerline[:, 0], centerline[:, 1], 20, marker=".", color="b")
        plt.scatter(right_centerline[:, 0], right_centerline[:, 1], 20, marker=".", color="r")
        plt.scatter(left_centerline[:, 0], left_centerline[:, 1], 20, marker=".", color="g")
        fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
        plt.savefig(f"polygon_unit_tests/{fname}.png")
        plt.close("all")

    # return the polygon
    return convert_lane_boundaries_to_polygon(right_centerline, left_centerline)


def convert_lane_boundaries_to_polygon(right_lane_bounds: np.ndarray, left_lane_bounds: np.ndarray) -> np.ndarray:
    """
    Take a left and right lane boundary and make a polygon of the lane segment, closing both ends of the segment.

    These polygons have the last vertex repeated (that is, first vertex == last vertex).

    Args:
       right_lane_bounds: Right lane boundary points. Shape is (N, 2).
       left_lane_bounds: Left lane boundary points.

    Returns:
       polygon: Numpy array of shape (2N+1,2)
    """
    assert right_lane_bounds.shape[0] == left_lane_bounds.shape[0]
    polygon = np.vstack([right_lane_bounds, left_lane_bounds[::-1]])
    polygon = np.vstack([polygon, right_lane_bounds[0]])
    return polygon


def filter_candidate_centerlines(
    xy: np.ndarray,
    candidate_cl: List[np.ndarray],
    stationary_threshold: float = 2.0,
    max_dist_margin: float = 2.0,
) -> List[np.ndarray]:
    """Filter candidate centerlines based on the distance travelled along the centerline.

    Args:
        xy: Trajectory coordinates.
        candidate_cl: List of candidate centerlines.
        stationary_threshold: minimum displacement to be called as non-stationary.
        max_dist_margin:

    Returns:
        filtered_candidate_centerlines: filtered list of candidate centerlines

    """

    # Check if stationary
    if math.sqrt((xy[0, 0] - xy[-1, 0]) ** 2 + (xy[0, 1] - xy[-1, 1]) ** 2) < stationary_threshold:
        stationary = True
    else:
        stationary = False

    # Filtering candidates to retain only those with distance along centerline close to traj length
    # Fit a second order polynomial and find trajectory length
    POLY_ORDER = 2
    poly = np.poly1d(np.polyfit(xy[:, 0], xy[:, 1], POLY_ORDER))
    obs_y_smooth = [poly(x) for x in xy[:, 0]]
    xy_smooth = [(xy[i, 0], obs_y_smooth[i]) for i in range(xy.shape[0])]
    traj_len = LineString(xy_smooth).length

    filtered_candidate_centerlines = []
    for centerline in candidate_cl:

        if stationary:
            filtered_candidate_centerlines.append(centerline)
        else:
            centerLine = LineString(centerline)
            start_dist = centerLine.project(Point(xy[0, 0], xy[0, 1]))
            end_dist = centerLine.project(Point(xy[-1, 0], xy[-1, 1]))

            dist_along_cl = end_dist - start_dist
            if dist_along_cl > traj_len - max_dist_margin and dist_along_cl < traj_len + max_dist_margin:
                filtered_candidate_centerlines.append(centerline)
    return filtered_candidate_centerlines


def is_overlapping_lane_seq(lane_seq1: Sequence[int], lane_seq2: Sequence[int]) -> bool:
    """
    Check if the 2 lane sequences are overlapping.
    Overlapping is defined as::

        s1------s2-----------------e1--------e2

    Here lane2 starts somewhere on lane 1 and ends after it, OR::

        s1------s2-----------------e2--------e1

    Here lane2 starts somewhere on lane 1 and ends before it

    Args:
        lane_seq1: list of lane ids
        lane_seq2: list of lane ids

    Returns:
        bool, True if the lane sequences overlap
    """

    if lane_seq2[0] in lane_seq1[1:] and lane_seq1[-1] in lane_seq2[:-1]:
        return True
    elif set(lane_seq2) <= set(lane_seq1):
        return True
    return False


def get_normal_and_tangential_distance_point(
    x: float, y: float, centerline: np.ndarray, delta: float = 0.01, last: bool = False
) -> Tuple[float, float]:
    """Get normal (offset from centerline) and tangential (distance along centerline) for the given point,
    along the given centerline

    Args:
        x: x-coordinate in map frame
        y: y-coordinate in map frame
        centerline: centerline along which n-t is to be computed
        delta: Used in computing offset direction
        last: True if point is the last coordinate of the trajectory

    Return:
        (tang_dist, norm_dist): tangential and normal distances
    """
    point = Point(x, y)
    centerline_ls = LineString(centerline)

    tang_dist = centerline_ls.project(point)
    norm_dist = point.distance(centerline_ls)
    point_on_cl = centerline_ls.interpolate(tang_dist)

    # Deal with last coordinate differently. Helped in dealing with floating point precision errors.
    if not last:
        pt1 = point_on_cl.coords[0]
        pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
        pt3 = point.coords[0]

    else:
        pt1 = centerline_ls.interpolate(tang_dist - delta).coords[0]
        pt2 = point_on_cl.coords[0]
        pt3 = point.coords[0]

    lr_coords = []
    lr_coords.extend([pt1, pt2, pt3])
    lr = LinearRing(lr_coords)

    # Left positive, right negative
    if lr.is_ccw:
        return (tang_dist, norm_dist)
    return (tang_dist, -norm_dist)


def get_nt_distance(xy: np.ndarray, centerline: np.ndarray, viz: bool = False) -> np.ndarray:
    """Get normal (offset from centerline) and tangential (distance along centerline) distances for the given xy trajectory,
    along the given centerline.

    Args:
        xy: Sequence of x,y,z coordinates.
        centerline: centerline along which n-t is to be computed
        viz: True if you want to visualize the computed centerlines.

    Returns:
        nt_distance: normal (offset from centerline) and tangential (distance along centerline) distances.
    """
    traj_len = xy.shape[0]
    nt_distance = np.zeros((traj_len, 2))

    delta_offset = 0.01
    last = 0
    max_dist: float = -1

    for i in range(traj_len):
        tang_dist, norm_dist = get_normal_and_tangential_distance_point(xy[i][0], xy[i][1], centerline, last=False)

        # Keep track of the last coordinate
        if tang_dist > max_dist:
            max_dist = tang_dist
            last_x = xy[i][0]
            last_y = xy[i][1]
            last_idx = i
        nt_distance[i, 0] = norm_dist
        nt_distance[i, 1] = tang_dist

    tang_dist, norm_dist = get_normal_and_tangential_distance_point(last_x, last_y, centerline, last=True)
    nt_distance[last_idx, 0] = norm_dist

    if viz:
        mpl_plotting_utils.visualize_centerline(centerline)

    return nt_distance


def get_oracle_from_candidate_centerlines(candidate_centerlines: List[np.ndarray], xy: np.ndarray) -> LineString:
    """Get oracle centerline from candidate centerlines. Chose based on direction of travel and maximum offset.
    First find the centerlines along which the distance travelled is close to maximum.
    If there are multiple candidates, then chose the one which has minimum max offset

    Args:
        candidate_centerlines: List of candidate centerlines
        xy: Trajectory coordinates

    Returns:
        oracle_centerline: Oracle centerline

    """

    max_offset = float("inf")
    max_dist_along_cl = -float("inf")

    # Chose based on distance travelled along centerline
    oracle_centerlines = []
    for centerline in candidate_centerlines:
        centerLine = LineString(centerline)
        start_dist = centerLine.project(Point(xy[0, 0], xy[0, 1]))
        end_dist = centerLine.project(Point(xy[-1, 0], xy[-1, 1]))
        dist_along_cl = end_dist - start_dist
        if dist_along_cl > max_dist_along_cl - 1:
            max_dist_along_cl = dist_along_cl
            oracle_centerlines.append(centerline)

    # Chose based on maximum offset
    min_of_max_offset = float("inf")
    for centerline in oracle_centerlines:
        max_offset = 0.0
        for i in range(xy.shape[0]):
            offset = Point(xy[i]).distance(LineString(centerline))
            max_offset = max(offset, max_offset)
        if max_offset < min_of_max_offset:
            min_of_max_offset = max_offset
            oracle_centerline = centerline

    return oracle_centerline


def get_centerlines_most_aligned_with_trajectory(xy: np.ndarray, candidate_cl: List[np.ndarray]) -> List[np.ndarray]:
    """Get the centerline from candidate_cl along which the trajectory travelled maximum distance

    Args:
        xy: Trajectory coordinates
        candidate_cl: List of candidate centerlines

    Returns:
        candidate_centerlines: centerlines along which distance travelled is maximum
    """

    max_dist_along_cl = -float("inf")

    for centerline in candidate_cl:
        centerline_linestring = LineString(centerline)
        start_dist = centerline_linestring.project(Point(xy[0, 0], xy[0, 1]))
        end_dist = centerline_linestring.project(Point(xy[-1, 0], xy[-1, 1]))
        dist_along_cl = end_dist - start_dist
        if max_dist_along_cl < -100 or dist_along_cl > max_dist_along_cl + 1:
            max_dist_along_cl = dist_along_cl
            candidate_centerlines = [centerline]
        elif dist_along_cl > max_dist_along_cl - 1:
            candidate_centerlines.append(centerline)
            max_dist_along_cl = max(max_dist_along_cl, dist_along_cl)

    return candidate_centerlines


def remove_overlapping_lane_seq(lane_seqs: List[List[int]]) -> List[List[int]]:
    """
    Remove lane sequences which are overlapping to some extent

    Args:
        lane_seqs (list of list of integers): List of list of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])

    Returns:
        List of sequence of lane ids (e.g. ``[[12345, 12346, 12347], [12345, 12348]]``)
    """
    redundant_lane_idx: Set[int] = set()
    for i in range(len(lane_seqs)):
        for j in range(len(lane_seqs)):
            if i in redundant_lane_idx or i == j:
                continue
            if is_overlapping_lane_seq(lane_seqs[i], lane_seqs[j]):
                redundant_lane_idx.add(j)

    unique_lane_seqs = [lane_seqs[i] for i in range(len(lane_seqs)) if i not in redundant_lane_idx]
    return unique_lane_seqs


def lane_waypt_to_query_dist(
    query_xy_city_coords: np.ndarray, nearby_lane_objs: List[LaneSegment]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the distance from a query to the closest waypoint in nearby lanes.

    Args:
       query_xy_city_coords: Numpy array of shape (2,)
       nearby_lane_objs: list of LaneSegment objects

    Returns:
       Tuple of (per_lane_dists, min_dist_nn_indices, dense_centerlines); all numpy arrays
    """
    per_lane_dists: List[float] = []
    dense_centerlines: List[np.ndarray] = []
    for nn_idx, lane_obj in enumerate(nearby_lane_objs):
        centerline = lane_obj.centerline
        # densely sample more points
        sample_num = 50
        centerline = interp_arc(sample_num, centerline[:, 0], centerline[:, 1])
        dense_centerlines += [centerline]
        # compute norms to waypoints
        waypoint_dist = np.linalg.norm(centerline - query_xy_city_coords, axis=1).min()
        per_lane_dists += [waypoint_dist]
    per_lane_dists = np.array(per_lane_dists)
    min_dist_nn_indices = np.argsort(per_lane_dists)
    return per_lane_dists, min_dist_nn_indices, dense_centerlines
