# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np
from argoverse.evaluation.eval_forecasting import compute_metric
from numpy.testing import assert_almost_equal


def test_compute_metric():
    """Test computation of ADE and FDE"""

    # Test Case:

    #   x: Ground Truth Trajectory
    #   *: Predicted Trajectory 1
    #   o: Predicted Trajectory 2

    #   0   1   2   3   4   5   6   7   8   9   10  11  12

    # 10

    # 9
    #       *   *   *   *   *   *   *   *   *   *   *   *
    # 8     x   x   x   x   x

    # 7     o   o   o   o   o   x   o

    # 6                             x   o

    # 5                                 x   o

    # 4                                 x   o

    # 3                                 x   o

    # 2                                 x   o

    # 1                                 x   o

    # 0

    target_1 = np.array(
        [
            [1.0, 8.0],
            [2.0, 8.0],
            [3.0, 8.0],
            [4.0, 8.0],
            [5.0, 8.0],
            [6.0, 7.0],
            [7.0, 6.0],
            [8.0, 5.0],
            [8.0, 4.0],
            [8.0, 3.0],
            [8.0, 2.0],
            [8.0, 1.0],
        ]
    )

    predicted_1_1 = np.array(
        [
            [1.0, 8.5],
            [2.0, 8.5],
            [3.0, 8.5],
            [4.0, 8.5],
            [5.0, 8.5],
            [6.0, 8.5],
            [7.0, 8.5],
            [8.0, 8.5],
            [9.0, 8.5],
            [10.0, 8.5],
            [11.0, 8.5],
            [12.0, 8.5],
        ]
    )

    predicted_1_2 = np.array(
        [
            [1.0, 7.0],
            [2.0, 7.0],
            [3.0, 7.0],
            [4.0, 7.0],
            [5.0, 7.0],
            [7.0, 7.0],
            [8.0, 6.0],
            [9.0, 5.0],
            [9.0, 4.0],
            [9.0, 3.0],
            [9.0, 2.0],
            [9.0, 1.0],
        ]
    )

    output_1 = [predicted_1_1, predicted_1_2]

    target_2 = target_1

    predicted_2_1 = predicted_1_1

    output_2 = [predicted_2_1]

    output = np.array([output_1, output_2])
    target = np.array([target_1, target_2])
    ade, fde, min_idx = compute_metric(output, target)

    expected_ade = 2.006
    expected_fde = 4.75
    expected_min_idx = [1, 0]

    assert_almost_equal(ade, expected_ade, 3)
    assert_almost_equal(fde, expected_fde, 3)
    np.array_equal(min_idx, expected_min_idx)
