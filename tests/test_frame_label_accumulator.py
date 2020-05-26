# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Forecasting Loader unit tests"""

import glob
import pathlib
import tempfile

import numpy as np
import pytest

from argoverse.data_loading.frame_label_accumulator import PerFrameLabelAccumulator

TEST_DATA_LOC = pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "tracking"


@pytest.fixture
def frame_acc() -> PerFrameLabelAccumulator:
    pfa = PerFrameLabelAccumulator(TEST_DATA_LOC, TEST_DATA_LOC, "test", save=False)
    pfa.accumulate_per_log_data()
    pfa.accumulate_per_log_data(log_id="1")
    return pfa


def test_traj_label_place_in_city(frame_acc: PerFrameLabelAccumulator) -> None:
    traj_list = frame_acc.get_log_trajectory_labels("1")
    city_frame_1_gt = [
        [[2.0, -1.0, -1.0], [2.0, -3.0, -1.0], [4.0, -1.0, -1.0], [4.0, -3.0, -1.0]],
        [[3.0, 1.0, 1.0], [3.0, 3.0, 1.0], [5.0, 1.0, 1.0], [5.0, 3.0, 1.0]],
        [[1.0, 4.0, 1.0], [1.0, 2.0, 1.0], [-1.0, 4.0, 1.0], [-1.0, 2.0, 1.0]],
    ]
    city_frame_0_gt = [
        [[2.0, 1.0, 1.0], [2.0, -1.0, 1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]],
        [[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [3.0, 1.0, -1.0], [3.0, -1.0, -1.0]],
        [[1.0, 0.0, 1.0], [1.0, 2.0, 1.0], [3.0, 0.0, 1.0], [3.0, 2.0, 1.0]],
    ]
    for traj in traj_list:
        assert traj.obj_class_str == "VEHICLE"
        city_frame = frame_acc.place_trajectory_in_city_frame(traj, "1")
        if traj.track_uuid == "00000000-0000-0000-0000-000000000000":
            assert np.array_equal(city_frame_0_gt, city_frame)
        elif traj.track_uuid == "00000000-0000-0000-0000-000000000001":
            assert np.array_equal(city_frame_1_gt, city_frame)
