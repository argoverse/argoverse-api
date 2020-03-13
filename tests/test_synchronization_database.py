# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
""" Test LiDAR / camera timestamp synchronization utilities"""

import numpy as np

from argoverse.data_loading.synchronization_database import find_closest_integer_in_ref_arr


def test_find_closest_integer_in_ref_arr_middle() -> None:
    """
    Verify that given a query integer, we can find the closest entry
    in an integer array (if lands in middle of array).
    """
    query_int = 4
    ref_arr = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)
    closest_int, int_diff = find_closest_integer_in_ref_arr(query_int, ref_arr)
    assert closest_int == 5
    assert int_diff == 1


def test_find_closest_integer_in_ref_arr_before() -> None:
    """
    Verify that given a query integer, we can find the closest entry
    in an integer array (if lands before start of array).
    """
    query_int = 0
    ref_arr = np.array([2, 5, 6, 7], dtype=np.int64)
    closest_int, int_diff = find_closest_integer_in_ref_arr(query_int, ref_arr)
    assert closest_int == 2
    assert int_diff == 2


def test_find_closest_integer_in_ref_arr_after() -> None:
    """
    Verify that given a query integer, we can find the closest entry
    in an integer array (if lands after the end of the array).
    """
    query_int = 9
    ref_arr = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)
    closest_int, int_diff = find_closest_integer_in_ref_arr(query_int, ref_arr)
    assert closest_int == 7
    assert int_diff == 2
