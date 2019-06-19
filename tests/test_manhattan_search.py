# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import List, Sequence, Tuple

import numpy as np
import pytest
from argoverse.utils.manhattan_search import (
    compute_point_cloud_bbox,
    compute_polygon_bboxes,
    find_all_polygon_bboxes_overlapping_query_bbox,
    find_local_polygons,
    prune_polygons_manhattan_dist,
)


def assert_np_obj_arrs_eq(pruned_polygons: np.ndarray, gt_pruned_polygons: np.ndarray):
    """Test for equivalency of two pology representations."""
    assert pruned_polygons.shape == gt_pruned_polygons.shape
    assert pruned_polygons.dtype == gt_pruned_polygons.dtype == object
    for i in range(pruned_polygons.shape[0]):
        poly = pruned_polygons[i]
        poly_gt = gt_pruned_polygons[i]
        assert np.allclose(poly, poly_gt)


@pytest.mark.parametrize(
    "point_cloud, gt_bbox",
    [
        (np.array([[-0.3, 0.5], [0.2, 0.1], [-0.5, 1.9]]), np.array([-0.5, 0.1, 0.2, 1.9])),
        (np.array([[-0.3, 0.5], [-0.3, 0.5], [-0.3, 0.5]]), np.array([-0.3, 0.5, -0.3, 0.5])),
        (np.array([[-0.3, 0.5, 50.1], [0.2, 0.1, -100.3], [-0.5, 1.9, -0.01]]), np.array([-0.5, 0.1, 0.2, 1.9])),
    ],
)
def test_compute_point_cloud_bbox_2d(point_cloud: np.ndarray, gt_bbox: np.ndarray):
    """Test for bounding box from pointcloud functionality."""
    assert np.allclose(compute_point_cloud_bbox(point_cloud), gt_bbox)


@pytest.fixture
def polygons_and_gt_bboxes() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return a list of polygons and the corresponding polygon bounding boxes."""
    poly_1 = np.array([[-1.5, -0.5], [0.5, -0.5], [-0.5, 1.5]])
    poly_2 = np.array([[0.5, 1], [-0.5, 1], [-0.6, 1]])
    poly_3 = np.array([[1.5, 2.5], [1, 3.5], [0.5, 2.5]])
    poly_4 = np.array([[-2.5, 1.75], [1.5, 1.75], [1.5, -1.5], [-2.5, -1.5]])
    poly_5 = np.array([[1.5, 0.5], [1.5, 1], [1.5, 1.1]])
    polygons = [poly_1, poly_2, poly_3, poly_4, poly_5]

    gt_poly_1_bbox = np.array([-1.5, -0.5, 0.5, 1.5])
    gt_poly_2_bbox = np.array([-0.6, 1, 0.5, 1])
    gt_poly_3_bbox = np.array([0.5, 2.5, 1.5, 3.5])
    gt_poly_4_bbox = np.array([-2.5, -1.5, 1.5, 1.75])
    gt_poly_5_bbox = np.array([1.5, 0.5, 1.5, 1.1])
    gt_poly_bboxes = [gt_poly_1_bbox, gt_poly_2_bbox, gt_poly_3_bbox, gt_poly_4_bbox, gt_poly_5_bbox]

    return polygons, gt_poly_bboxes


def test_find_all_polygon_bboxes_overlapping_query_bbox(polygons_and_gt_bboxes):
    """Test for correctness of """
    poly_bboxes = np.array([compute_point_cloud_bbox(poly) for poly in polygons_and_gt_bboxes[0]])

    query_bbox = np.array([-1.5, 0.5, 1.5, 1.5])
    overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(poly_bboxes, query_bbox)
    gt_overlap_bool = np.array([True, True, False, True, True])
    gt_overlap_indxs = np.where(gt_overlap_bool)[0]
    assert np.allclose(overlap_indxs, gt_overlap_indxs)


def test_compute_polygon_bboxes(polygons_and_gt_bboxes):
    """Test for correctness of compute_polygon_bboxes."""
    polygon_bboxes = compute_polygon_bboxes(np.array(polygons_and_gt_bboxes[0]))
    gt_polygon_bboxes = np.array(polygons_and_gt_bboxes[1])
    assert np.allclose(polygon_bboxes, gt_polygon_bboxes)


@pytest.mark.parametrize(
    "query_pt, query_search_range_manhattan, gt_indices",
    [(np.array([-0.5, 1.5]), 0.5, [0, 1, 3]), (np.array([-0.5, 1.5]), 0.499, [0, 3]), (np.array([0, 2]), 0.24, [])],
)
def test_prune_polygons_manhattan_dist_find_nearby(
    query_pt: np.ndarray, query_search_range_manhattan: float, gt_indices: Sequence[int], polygons_and_gt_bboxes
):
    """Test for correctness of prune_polygons_manhattan_dist."""
    polygons = np.array(polygons_and_gt_bboxes[0])
    pruned_polygons = prune_polygons_manhattan_dist(query_pt, polygons.copy(), query_search_range_manhattan)
    gt_pruned_polygons = np.array([polygons[i] for i in gt_indices], dtype="O")
    assert_np_obj_arrs_eq(gt_pruned_polygons, pruned_polygons)


def test_find_local_polygons(polygons_and_gt_bboxes):
    """Test for correctness of find_local_polygons."""
    polygons = np.array(polygons_and_gt_bboxes[0])
    poly_bboxes = np.array(polygons_and_gt_bboxes[1])
    query_bbox = np.array([-1.5, 0.5, 1.5, 1.5])

    query_min_x, query_min_y, query_max_x, query_max_y = query_bbox
    local_polygons, overlap_indxs = find_local_polygons(
        polygons.copy(), poly_bboxes, query_min_x, query_max_x, query_min_y, query_max_y
    )

    gt_overlap_bool = np.array([True, True, False, True, True])
    gt_overlap_indxs = np.where(gt_overlap_bool)[0]

    assert np.allclose(overlap_indxs, gt_overlap_indxs)
    assert_np_obj_arrs_eq(polygons[gt_overlap_bool], local_polygons)
