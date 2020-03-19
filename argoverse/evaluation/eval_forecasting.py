# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module evaluates the forecasted trajectories against the ground truth."""

import math
import pickle as pkl
from typing import Dict, List, Tuple

import numpy as np

from argoverse.map_representation.map_api import ArgoverseMap


def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Dict[int, List[float]],
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade = [], []
    min_fde, prob_min_fde = [], []
    n_misses, prob_n_misses = [], []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        min_idx = 0
        for j in range(0, min(max_guesses, len(forecasted_trajectories[k]))):
            fde = get_fde(forecasted_trajectories[k][j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        curr_min_ade = get_ade(forecasted_trajectories[k][min_idx][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)
        prob_n_misses.append(1.0 if curr_min_fde > miss_threshold else (1.0 - forecasted_probabilities[k][min_idx]))
        prob_min_ade.append(-np.log(forecasted_probabilities[k][min_idx]) + curr_min_ade)
        prob_min_fde.append(-np.log(forecasted_probabilities[k][min_idx]) + curr_min_fde)
    metric_results["minADE"] = sum(min_ade) / len(min_ade)
    metric_results["minFDE"] = sum(min_fde) / len(min_fde)
    metric_results["MR"] = sum(n_misses) / len(n_misses)
    metric_results["p-minADE"] = sum(prob_min_ade) / len(prob_min_ade)
    metric_results["p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde)
    metric_results["p-MR"] = sum(prob_n_misses) / len(prob_n_misses)
    return metric_results


def get_drivable_area_compliance(
    forecasted_trajectories: Dict[int, List[np.ndarray]], city_names: Dict[int, str], max_n_guesses: int
) -> float:
    """Compute drivable area compliance metric.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        city_names: Dict mapping sequence id to city name.
        max_n_guesses: Maximum number of guesses allowed.

    Returns:
        Mean drivable area compliance

    """
    avm = ArgoverseMap()

    dac_score = []

    for seq_id, trajectories in forecasted_trajectories.items():
        city_name = city_names[seq_id]
        num_dac_trajectories = 0
        n_guesses = min(max_n_guesses, len(trajectories))
        for trajectory in trajectories[:n_guesses]:
            raster_layer = avm.get_raster_layer_points_boolean(trajectory, city_name, "driveable_area")
            if np.sum(raster_layer) == raster_layer.shape[0]:
                num_dac_trajectories += 1

        dac_score.append(num_dac_trajectories / n_guesses)

    return sum(dac_score) / len(dac_score)


def compute_forecasting_metrics(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    city_names: Dict[int, str],
    max_n_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Dict[int, List[float]],
) -> Dict[str, float]:
    """Compute all the forecasting metrics.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        city_names: Dict mapping sequence id to city name.
        max_n_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Miss threshold
        forecasted_probabilities: Normalized Probabilities associated with each of the forecasted trajectories.

     Returns:
        metric_results: Dictionary containing values for all metrics.
    """
    metric_results = get_displacement_errors_and_miss_rate(
        forecasted_trajectories, gt_trajectories, max_n_guesses, horizon, miss_threshold, forecasted_probabilities
    )
    metric_results["DAC"] = get_drivable_area_compliance(forecasted_trajectories, city_names, max_n_guesses)

    print("------------------------------------------------")
    print(f"Prediction Horizon : {horizon}, Max #guesses (K): {max_n_guesses}")
    print("------------------------------------------------")
    print(metric_results)
    print("------------------------------------------------")

    return metric_results
