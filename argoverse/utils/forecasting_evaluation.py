# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np


def compute_summed_distance_point_cloud2D(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Args:
        points_a: numpy n-d array with dims (N x 2)
        points_b: numpy n-d array with dims (N x 2)

    Returns:
        scalar (double): summed L2 norm between each pair of corresponding 2D points
    """
    return np.sum(np.sqrt(np.sum(np.square(points_a - points_b), axis=1)))


def evaluate_prediction(
    pred_traj: np.ndarray,
    ground_truth_traj: np.ndarray,
    eval_method: str = "EVAL_DESTINATION_ONLY",
) -> np.ndarray:
    """Compute the error as L2 norm in trajectories

    Args:
        pred_traj: numpy n-d array with dims (N x 2)
        ground_truth_traj: numpy n-d array with dims (N x 2)
        eval_method:
    """
    if eval_method == "EVAL_DESTINATION_ONLY":
        return np.linalg.norm(pred_traj[-1] - ground_truth_traj[-1])
    elif eval_method == "EVAL_AT_DISCRETIZED_STEPS":
        # eval many timesteps along the full discretized trajectory
        return compute_summed_distance_point_cloud2D(ground_truth_traj, pred_traj)
    else:
        print("Eval method unrecognized")
