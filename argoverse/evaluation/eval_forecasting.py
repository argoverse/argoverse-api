# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Functions for evaluating error in forecasting."""
import math
from typing import List, Tuple

import numpy as np


def get_ade(output: np.ndarray, target: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        output: Predicted trajectory with shape (pred_len x 2)
        target: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = output.shape[0]
    ade = float(
        sum(math.sqrt((output[i, 0] - target[i, 0]) ** 2 + (output[i, 1] - target[i, 1]) ** 2) for i in range(pred_len))
        / pred_len
    )
    return ade


def get_fde(output: np.ndarray, target: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        output: Predicted trajectory with shape (pred_len x 2)
        target: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error
    """
    fde = math.sqrt((output[-1, 0] - target[-1, 0]) ** 2 + (output[-1, 1] - target[-1, 1]) ** 2)
    return fde


def compute_metric(output: np.ndarray, target: np.ndarray) -> Tuple[float, float, List[int]]:
    """Compute ADE and FDE

    Args:
        output: Predicted top-k trajectories with shape (num_tracks, 1), where each 
                element is a list. Each list has >= 1 predictions of shape (pred_len x 2).  
        target: Ground Truth Trajectory of shape (num_tracks x pred_len x 2)

    Returns:
        ade: Average Displacement Error
        fde: Final Displacement Error
        min_ade_idx: index corresponding to the trajectory in top-k which gave minimum ade

    """
    ade = []
    fde = []
    min_ade_idx = []
    for i in range(output.shape[0]):
        best_ade = float("inf")
        best_fde = float("inf")
        best_idx = -1
        for j in range(len(output[i])):
            curr_ade = get_ade(output[i][j], target[i])
            if curr_ade < best_ade:
                best_ade = curr_ade
                best_fde = get_fde(output[i][j], target[i])
                best_idx = j
        ade.append(best_ade)
        fde.append(best_fde)
        min_ade_idx.append(best_idx)
    mean_ade = sum(ade) / len(ade)
    mean_fde = sum(fde) / len(fde)
    return mean_ade, mean_fde, min_ade_idx
