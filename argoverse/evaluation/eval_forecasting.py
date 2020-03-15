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
) -> Tuple[float, float, float]:
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

    Returns:
        mean_min_ade: Min ade averaged over all the samples
        mean_min_fde: Min fde averaged over all the samples
        miss_rate: Mean number of misses

    """
    min_ade = []
    min_fde = []
    n_misses = []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        for j in range(0, min(max_guesses, len(forecasted_trajectories[k]))):
            fde = get_fde(forecasted_trajectories[k][j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                curr_min_fde = fde
                curr_min_ade = get_ade(forecasted_trajectories[k][j][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)
    mean_min_ade = sum(min_ade) / len(min_ade)
    mean_min_fde = sum(min_fde) / len(min_fde)
    miss_rate = sum(n_misses) / len(n_misses)
    return mean_min_ade, mean_min_fde, miss_rate


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
) -> Tuple[float, float, float, float]:
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

     Returns:
        mean_min_ade: Mean of min_ade for all the trajectories.
        mean_min_fde: Mean of min_fde for all the trajectories.
        mean_dac: Mean drivable area compliance for all the trajectories.
        miss_rate: Miss rate.

    """
    mean_min_ade, mean_min_fde, miss_rate = get_displacement_errors_and_miss_rate(
        forecasted_trajectories, gt_trajectories, max_n_guesses, horizon, miss_threshold
    )

    mean_dac = get_drivable_area_compliance(forecasted_trajectories, city_names, max_n_guesses)

    print("------------------------------------------------")
    print(f"Prediction Horizon : {horizon}, Max #guesses (K): {max_n_guesses}")
    print("------------------------------------------------")
    print(f"minADE: {mean_min_ade}")
    print(f"minFDE: {mean_min_fde}")
    print(f"DAC: {mean_dac}")
    print(f"Miss Rate: {miss_rate}")
    print("------------------------------------------------")

    return mean_min_ade, mean_min_fde, mean_dac, miss_rate
