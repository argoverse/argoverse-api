
"""
Tests to ensure Argoverse 2.0 vector map entities are loaded/converted correctly to Numpy arrays.
"""

import numpy as np

from argoverse.data_loading.vector_map_v2_loader import point_arr_from_points_list_dict

def test_point_arr_from_points_list_dict() -> None:
    """Ensure dictionary with lane boundary polyline parameterization is lifted to Numpy array correctly."""
    lane_segment = {
        "right_lane_boundary": {
            "points": [
                {"x": 919.19, "y": 150.0, "z": -21.42},
                {"x": 916.42, "y": 171.79, "z": -21.67},
            ]
        }
    }
    right_ln_bound = point_arr_from_points_list_dict(lane_segment["right_lane_boundary"]["points"])
    # fmt: off
    expected_right_ln_bound = np.array(
        [
            [919.19, 150.0, -21.42],
            [916.42, 171.79, -21.67]
        ]
    )
    # fmt: on
    np.testing.assert_allclose(right_ln_bound, expected_right_ln_bound)


def test_point_arr_from_points_list_dict() -> None:
    """Ensure dictionary with lane boundary polyline parameterization is lifted to Numpy array correctly."""
    # fmt: off
    points_list_dict = [
        {"x": 5504, "y": 3291, "z": -6},
        {"x": 5491, "y": 3263, "z": -5}
    ]
    # fmt: on
    arr = point_arr_from_points_list_dict(points_list_dict)
    # fmt: off
    gt_arr = np.array(
        [
            [5504, 3291, -6],
            [5491, 3263, -5]
        ])
    # fmt: on
    np.testing.assert_allclose(arr, gt_arr)
