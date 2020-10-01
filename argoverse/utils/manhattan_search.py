# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Fast search functions of nearest neighbor based on Manhattan distance."""

import logging
from typing import List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_polygon_bboxes(polygons: np.ndarray) -> np.ndarray:
    """Compute the minimum size enclosing xy bounding box for each polygon that is provided as input.
    Args:
        polygons: an array of type 'O' (object) with shape (n,). Each object has shape (m, 3+).

    Returns:
        polygon_bboxes: a float array with shape (n, 4).
    """
    bboxes: List[np.ndarray] = []

    for polygon in polygons:
        bbox = compute_point_cloud_bbox(polygon)
        bboxes.append(bbox)

    polygon_bboxes = np.array(bboxes)
    return polygon_bboxes


def compute_point_cloud_bbox(point_cloud: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Given a set of 2D or 3D points, find the minimum size axis-aligned bounding box in the xy plane (ground plane).

    Args:
        point_cloud: an array of dim (N,3) or (N,2).
        verbose: False by default, if set to True, it prints the bounding box dimensions.

    Returns:
        bbox: an array of dim (4,) representing x_min, y_min, x_max, y_max.
    """
    x_min = np.amin(point_cloud[:, 0])
    x_max = np.amax(point_cloud[:, 0])

    y_min = np.amin(point_cloud[:, 1])
    y_max = np.amax(point_cloud[:, 1])

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    bbox = np.array([x_min, y_min, x_max, y_max])

    if verbose:
        logger.info(f"Point cloud bbox width = {bbox_width}, height = {bbox_height}")
    return bbox


def find_all_polygon_bboxes_overlapping_query_bbox(polygon_bboxes: np.ndarray, query_bbox: np.ndarray) -> np.ndarray:
    """Find all the overlapping polygon bounding boxes.

    Each bounding box has the following structure:
        bbox = np.array([x_min,y_min,x_max,y_max])

    In 3D space, if the coordinates are equal (polygon bboxes touch), then these are considered overlapping.
    We have a guarantee that the cropped image will have any sort of overlap with the zero'th object bounding box
    inside of the image e.g. along the x-dimension, either the left or right side of the bounding box lies between the
    edges of the query bounding box, or the bounding box completely engulfs the query bounding box.

    Args:
        polygon_bboxes: An array of shape (K,), each array element is a NumPy array of shape (4,) representing
                        the bounding box for a polygon or point cloud.
        query_bbox: An array of shape (4,) representing a 2d axis-aligned bounding box, with order
                    [min_x,min_y,max_x,max_y].

    Returns:
        An integer array of shape (K,) representing indices where overlap occurs.
    """
    query_min_x = query_bbox[0]
    query_min_y = query_bbox[1]

    query_max_x = query_bbox[2]
    query_max_y = query_bbox[3]

    bboxes_x1 = polygon_bboxes[:, 0]
    bboxes_x2 = polygon_bboxes[:, 2]

    bboxes_y1 = polygon_bboxes[:, 1]
    bboxes_y2 = polygon_bboxes[:, 3]

    # check if falls within range
    overlaps_left = (query_min_x <= bboxes_x2) & (bboxes_x2 <= query_max_x)
    overlaps_right = (query_min_x <= bboxes_x1) & (bboxes_x1 <= query_max_x)

    x_check1 = bboxes_x1 <= query_min_x
    x_check2 = query_min_x <= query_max_x
    x_check3 = query_max_x <= bboxes_x2
    x_subsumed = x_check1 & x_check2 & x_check3

    x_in_range = overlaps_left | overlaps_right | x_subsumed

    overlaps_below = (query_min_y <= bboxes_y2) & (bboxes_y2 <= query_max_y)
    overlaps_above = (query_min_y <= bboxes_y1) & (bboxes_y1 <= query_max_y)

    y_check1 = bboxes_y1 <= query_min_y
    y_check2 = query_min_y <= query_max_y
    y_check3 = query_max_y <= bboxes_y2
    y_subsumed = y_check1 & y_check2 & y_check3
    y_in_range = overlaps_below | overlaps_above | y_subsumed

    overlap_indxs = np.where(x_in_range & y_in_range)[0]
    return overlap_indxs


def find_local_polygons(
    lane_polygons: np.ndarray,
    lane_bboxes: np.ndarray,
    query_min_x: float,
    query_max_x: float,
    query_min_y: float,
    query_max_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find local polygons. We always also return indices.

    Take a collection of precomputed polygon bounding boxes, and compare with a query bounding box then returns the
    polygons that overlap, along with their array indices.

    Args:
        lane_polygons: An array of polygons.
        lane_bboxes: An array of shape (K,), each array element is a NumPy array of shape (4,) representing
                    the bounding box for a polygon or point cloud.
        query_min_x: minimum x coordinate of the query bounding box.
        query_max_x: maximum x coordinate of the query bounding box.
        query_min_y: minimum y coordinate of the query bounding box.
        query_max_y: maximum y coordinate of the query bounding box.
        return_indices: False by default, if set to True, the overlapping indices are returned along with the
                        overlapping polygon.

    Returns:
        Overlapping polygon.
        Overlapping indices.
    """
    query_bbox = np.array([query_min_x, query_min_y, query_max_x, query_max_y])
    overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(lane_bboxes, query_bbox)

    pruned_lane_polygons = lane_polygons[overlap_indxs]
    return pruned_lane_polygons, overlap_indxs


def prune_polygons_manhattan_dist(
    query_pt: np.ndarray,
    points_xyz: np.ndarray,
    query_search_range_manhattan: float = 200.0,
) -> np.ndarray:
    """Prune polygon points based on a search area defined by the manhattan distance.

    Take a collection of small point clouds and return only point clouds that fall within a manhattan search radius of
    the 2D query point.

    Similar to the function above, except query bounding box and polygon bounding boxes are not pre-computed, meaning
    they must be computed on fly, which can be quite computationally expensive in a loop.

    Args:
        query_pt: Numpy n-d array with dimension (2,) representing xy query location.
        points_xyz: An array of shape (n,) of array objects. Each array object could be a 2D or 3D polygon, i.e. of
        shape (m,2) or (m,3) respectively.
        query_search_range_manhattan: Side length of query bounding box square which is set to 200 by default.

    Returns:
        An array pruned xyz point objects of shape (k,). Each array object could be a 2D or 3D polygon, i.e. of shape
        (m,2) or (m,3) respectively.
    """
    bboxes = compute_polygon_bboxes(points_xyz)

    query_min_x = query_pt[0] - query_search_range_manhattan
    query_max_x = query_pt[0] + query_search_range_manhattan
    query_min_y = query_pt[1] - query_search_range_manhattan
    query_max_y = query_pt[1] + query_search_range_manhattan

    query_bbox = np.array([query_min_x, query_min_y, query_max_x, query_max_y])
    overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(bboxes, query_bbox)

    pruned_points_xyz = points_xyz[overlap_indxs]
    return pruned_points_xyz
