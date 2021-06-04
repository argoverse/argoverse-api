# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module tests the motion forecasting metric computation."""

from typing import Mapping

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics


def assert_metrics_almost_equal(expected_metrics: Mapping[str, float], output_metrics: Mapping[str, float]) -> None:
    """Assert that the expected metrics match the computed metrics.

    Args:
        expected_metrics: Expected metrics for the test case.
        output_metrics: Metrics computed by the evaluation code.
    """

    assert set(expected_metrics.keys()).issubset(set(output_metrics.keys()))
    for metric in expected_metrics:
        assert_almost_equal(expected_metrics[metric], round(output_metrics[metric], 2), 2)


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

# Ground truth trajectory for the test case.
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

# Forecasted trajectory that is close to the ground truth.
good_forecast = np.array(
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

# Forecasted trajectory that is far from the ground truth.
bad_forecast = np.array(
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
horizon = 30
miss_threshold = 1.0

# Test cases for metric computation
test_metric_params = {
    "Case 1: Top-2 error, 2 forecasts, probabilities provided.": [
        {1: [good_forecast, bad_forecast]},
        {1: ground_truth},
        2,
        {1: [0.80, 0.20]},
        {
            "minADE": 1.0,
            "minFDE": 1.0,
            "DAC": 1.0,
            "MR": 0.0,
            "p-minADE": 1.22,
            "p-minFDE": 1.22,
            "p-MR": 0.20,
            "brier-minADE": 1.04,
            "brier-minFDE": 1.04,
        },
    ],
    "Case 2: Top-2, 1 forecast, probabilities provided.": [
        {1: [bad_forecast]},
        {1: ground_truth},
        2,
        {1: [1.0]},
        {
            "minADE": 3.23,
            "minFDE": 11.31,
            "DAC": 1.0,
            "MR": 1.0,
            "p-minADE": 3.23,
            "p-minFDE": 11.31,
            "p-MR": 1.0,
            "brier-minADE": 3.23,
            "brier-minFDE": 11.31,
        },
    ],
    "Case 3: Top-1, 2 forecast, probabilities provided.": [
        {1: [good_forecast, bad_forecast]},
        {1: ground_truth},
        1,
        {1: [0.2, 0.8]},
        {
            "minADE": 3.23,
            "minFDE": 11.31,
            "DAC": 1.0,
            "MR": 1.0,
            "p-minADE": 3.23,
            "p-minFDE": 11.31,
            "p-MR": 1.0,
            "brier-minADE": 3.23,
            "brier-minFDE": 11.31,
        },
    ],
    "Case 4: Top-2, 2 forecast, probabilities provided (not normalized)": [
        {1: [good_forecast, bad_forecast]},
        {1: ground_truth},
        2,
        {1: [0.3, 0.2]},
        {
            "minADE": 1.0,
            "minFDE": 1.0,
            "DAC": 1.0,
            "MR": 0.0,
            "p-minADE": 1.51,
            "p-minFDE": 1.51,
            "p-MR": 0.4,
            "brier-minADE": 1.16,
            "brier-minFDE": 1.16,
        },
    ],
    "Case 5: Top-2, 2 forecast, probabilities provided (uniform)": [
        {1: [good_forecast, bad_forecast]},
        {1: ground_truth},
        2,
        {1: [0.5, 0.5]},
        {
            "minADE": 1.0,
            "minFDE": 1.0,
            "DAC": 1.0,
            "MR": 0.0,
            "p-minADE": 1.69,
            "p-minFDE": 1.69,
            "p-MR": 0.5,
            "brier-minADE": 1.25,
            "brier-minFDE": 1.25,
        },
    ],
    "Case 6: Top-2, 2 forecast, No probabilities": [
        {1: [bad_forecast, bad_forecast, good_forecast]},
        {1: ground_truth},
        2,
        None,
        {"minADE": 3.23, "minFDE": 11.31, "DAC": 1.0, "MR": 1.0},
    ],
}


@pytest.mark.parametrize(
    "forecasted_trajectories, ground_truth_trajectories, max_n_guesses, forecasted_probabilities, expected_metrics",
    [v for _, v in test_metric_params.items()],
    ids=[k for k in test_metric_params],
)
def test_forecasting_metrics(
    forecasted_trajectories,
    ground_truth_trajectories,
    max_n_guesses,
    forecasted_probabilities,
    expected_metrics,
) -> None:
    metrics = compute_forecasting_metrics(
        forecasted_trajectories,
        ground_truth_trajectories,
        city_names,
        max_n_guesses,
        horizon,
        miss_threshold,
        forecasted_probabilities,
    )

    assert_metrics_almost_equal(expected_metrics, metrics)
