# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module tests the motion forecasting metric evaluation."""

import numpy as np
from numpy.testing import assert_almost_equal

from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics


def test_compute_forecasting_metric() -> None:
    """Test computation of motion forecasting metrics."""
    # Test Case:

    #   x: Ground Truth Trajectory
    #   *: Predicted Trajectory 1
    #   o: Predicted Trajectory 2

    #   90   91   92   93   94   95   96   97   98   99   100  101  102

    # 3980       |       x   *   o            |
    #            |       x   *   o            |
    #            |       x   *   o            |
    #            |       x   *   o            |\
    # 3975       |       x   *   o            | \
    #            |       x   *   o            |  \
    #            |       x   *   o            |   ---------------------------
    #            |       x   *   o            |
    # 3970       |       x   *   o            |
    #            |         x   *   o          |
    #            |           x   *   o        |
    #            |           x   *   o        |
    # 3965       |           x   *   o        |
    #            |           x   *      o     |
    #            |           x   *          o |  o   o   o   o   o   o   o   o
    #            |           x   *            |
    # 3960       |             x   *          |\
    #            |               x   *        | \
    #            |               x   *        |  \
    #            |               x   *        |   -----------------------------
    # 3955       |               x   *        |
    #            |               x   *        |
    #            |               x   *        |
    #            |               x   *        |
    # 3950       |               x   *        |

    ground_truth = np.array(
        [
            [93, 3979],
            [93, 3978],
            [93, 3977],
            [93, 3976],
            [93, 3976],
            [93, 3975],
            [93, 3974],
            [93, 3973],
            [93, 3972],
            [93, 3971],
            [93, 3970],
            [94, 3969],
            [94, 3969],
            [94, 3968],
            [94, 3967],
            [94, 3966],
            [94, 3966],
            [94, 3965],
            [94, 3964],
            [94, 3963],
            [94, 3962],
            [95, 3961],
            [95, 3960],
            [95, 3959],
            [94, 3957],
            [94, 3957],
            [94, 3956],
            [94, 3955],
            [95, 3953],
            [95, 3952],
        ]
    )

    forecasted_1 = np.array(
        [
            [94, 3979],
            [94, 3978],
            [94, 3977],
            [94, 3976],
            [94, 3976],
            [94, 3975],
            [94, 3974],
            [94, 3973],
            [94, 3972],
            [94, 3971],
            [94, 3970],
            [95, 3969],
            [95, 3969],
            [95, 3968],
            [95, 3967],
            [95, 3966],
            [95, 3966],
            [95, 3965],
            [95, 3964],
            [95, 3963],
            [95, 3962],
            [96, 3961],
            [96, 3960],
            [96, 3959],
            [95, 3957],
            [95, 3957],
            [95, 3956],
            [95, 3955],
            [96, 3953],
            [96, 3952],
        ]
    )

    forecasted_2 = np.array(
        [
            [95, 3979],
            [95, 3978],
            [95, 3977],
            [95, 3976],
            [95, 3976],
            [95, 3975],
            [95, 3974],
            [95, 3973],
            [95, 3972],
            [95, 3971],
            [95, 3970],
            [96, 3969],
            [96, 3969],
            [96, 3968],
            [96, 3967],
            [96, 3966],
            [96, 3966],
            [96, 3965],
            [96, 3964],
            [96, 3963],
            [96, 3962],
            [97, 3961],
            [97, 3960],
            [98, 3960],
            [98, 3960],
            [99, 3960],
            [100, 3960],
            [101, 3960],
            [102, 3960],
            [103, 3960],
        ]
    )

    city_names = {1: "MIA"}
    max_n_guesses = 2
    horizon = 30
    miss_threshold = 1.0

    # Case 1
    forecasted_trajectories = {1: [forecasted_1, forecasted_2]}
    forecasted_probabilities = {1: [0.80, 0.20]}
    ground_truth_trajectories = {1: ground_truth}

    metrics = compute_forecasting_metrics(
        forecasted_trajectories,
        ground_truth_trajectories,
        city_names,
        max_n_guesses,
        horizon,
        miss_threshold,
        forecasted_probabilities,
    )

    expected_min_ade = 1.0
    expected_min_fde = 1.0
    expected_dac = 1.0
    expected_miss_rate = 0.0
    expected_p_min_ade = 1.22
    expected_p_min_fde = 1.22
    expected_p_miss_rate = 0.20
    assert_almost_equal(expected_min_ade, round(metrics["minADE"], 2), 2)
    assert_almost_equal(expected_min_fde, round(metrics["minFDE"], 2), 2)
    assert_almost_equal(expected_dac, round(metrics["DAC"], 2), 2)
    assert_almost_equal(expected_miss_rate, round(metrics["MR"], 2), 2)
    assert_almost_equal(expected_p_min_ade, round(metrics["p-minADE"], 2), 2)
    assert_almost_equal(expected_p_min_fde, round(metrics["p-minFDE"], 2), 2)
    assert_almost_equal(expected_p_miss_rate, round(metrics["p-MR"], 2), 2)

    # Case 2
    forecasted_trajectories = {1: [forecasted_2]}
    forecasted_probabilities = {1: [1.0]}
    ground_truth_trajectories = {1: ground_truth}

    metrics = compute_forecasting_metrics(
        forecasted_trajectories,
        ground_truth_trajectories,
        city_names,
        max_n_guesses,
        horizon,
        miss_threshold,
        forecasted_probabilities,
    )

    expected_min_ade = 3.23
    expected_min_fde = 11.31
    expected_dac = 1.0
    expected_miss_rate = 1.0
    expected_p_min_ade = 3.23
    expected_p_min_fde = 11.31
    expected_p_miss_rate = 1.0

    assert_almost_equal(expected_min_ade, round(metrics["minADE"], 2), 2)
    assert_almost_equal(expected_min_fde, round(metrics["minFDE"], 2), 2)
    assert_almost_equal(expected_dac, round(metrics["DAC"], 2), 2)
    assert_almost_equal(expected_miss_rate, round(metrics["MR"], 2), 2)
    assert_almost_equal(expected_p_min_ade, round(metrics["p-minADE"], 2), 2)
    assert_almost_equal(expected_p_min_fde, round(metrics["p-minFDE"], 2), 2)
    assert_almost_equal(expected_p_miss_rate, round(metrics["p-MR"], 2), 2)
