# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
"""Detection utilities for the Argoverse detection leaderboard.

Accepts detections (in Argoverse ground truth format) and ground truth labels

"""

from enum import Enum, auto
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.utils.transform import quat2rotmat


class SimFnType(Enum):
    CENTER = auto()


class DistFnType(Enum):
    TRANSLATION = auto()
    SCALE = auto()
    ORIENTATION = auto()


class InterpType(Enum):
    ALL = auto()


def filter_instances(
    instances: np.ndarray, target_class: str, range_metric: str = "euclidean", max_dist: float = 50.0
) -> np.ndarray:
    """Filter the annotations based on a set of conditions.

    Args:
        annos: The instances to be filtered.
        target_class: The name of the class of interest.
        range_metric: The range metric used for filtering.
    
    Returns:
        The filtered annotations.
    """
    instances = np.array([instance for instance in instances if instance.label_class == target_class])

    if range_metric == "euclidean":
        centers = np.array([dt.translation for dt in instances])
        filtered_annos = np.array([])

        if centers.shape[0] > 0:
            dt_dists = np.linalg.norm(centers, axis=1)
            filtered_annos = instances[dt_dists < max_dist]

        return filtered_annos


def get_ranks(dts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the rankings for the detections.

    Args:
        dts: Detections.

    Returns:
        scores: The detection scores.
        ranks: The ranking for the detections.

    """
    scores = np.array([dt.score for dt in dts])
    ranks = scores.argsort()[::-1]
    return np.expand_dims(scores, 1)[ranks], ranks


def interp(prec: np.ndarray, method: InterpType = InterpType.ALL) -> np.ndarray:
    """Interpolate the precision over all recall levels.

    Args:
        prec: Precision at all recall levels.
        method: Accumulation method.
    
    Returns:
        Interpolated precision at all recall levels.
    """
    if method == InterpType.ALL:
        prec_interp = np.maximum.accumulate(prec[::-1])[::-1]
    else:
        raise NotImplemented("This interpolation method is not implemented!")
    return prec_interp


def compute_match_matrix(dts: np.ndarray, gts: np.ndarray, metric: SimFnType) -> np.ndarray:
    """Calculate the match matrix between detections and ground truth labels,
    using a specified similarity function.

    Args:
        dts: Detections.
        gts: Ground truth labels.
        metric: Similarity metric type.
    
    Returns:
        Interpolated precision at all recall levels.
    """
    if metric == SimFnType.CENTER:
        dt_centers = np.array([dt.translation for dt in dts])
        gt_centers = np.array([gt.translation for gt in gts])
        sims = -cdist(dt_centers, gt_centers)
    else:
        raise NotImplemented("This similarity metric is not implemented!")
    return sims


def get_error_types(match_scores: np.ndarray, thresh: float, metric: SimFnType) -> np.ndarray:
    # Euclidean distance represented as a "similarity" metric.
    if metric == SimFnType.CENTER:
        return match_scores > -thresh
    else:
        raise NotImplemented("This similarity metric is not implemented!")


def dist_fn(dts: pd.DataFrame, gts: pd.DataFrame, metric: DistFnType) -> np.ndarray:
    if metric == DistFnType.TRANSLATION:
        dt_centers = np.vstack(dts["translation"].array)
        gt_centers = np.vstack(gts["translation"].array)
        trans_errors = np.linalg.norm(dt_centers - gt_centers, axis=1)
        return trans_errors
    elif metric == DistFnType.SCALE:
        dt_dims = dts[["width", "length", "height"]]
        gt_dims = gts[["width", "length", "height"]]
        inter = np.minimum(dt_dims, gt_dims).prod(axis=1)
        union = np.maximum(dt_dims, gt_dims).prod(axis=1)
        scale_errors = 1 - (inter / union)
        return scale_errors
    elif metric == DistFnType.ORIENTATION:
        dt_yaws = R.from_quat(np.vstack(dts["quaternion"].array)[:, [3, 0, 1, 2]]).as_euler("xyz")[:, 2]
        gt_yaws = R.from_quat(np.vstack(gts["quaternion"].array)[:, [3, 0, 1, 2]]).as_euler("xyz")[:, 2]
        orientation_errors = np.abs((dt_yaws - gt_yaws + np.pi) % (2 * np.pi) - np.pi)
        return orientation_errors
    else:
        raise NotImplemented("This distance metric is not implemented!")


def get_label_orientations(labels: List[ObjectLabelRecord]) -> float:
    """Get the orientation (yaw) of a label.

    Args:
        label: The label

    Returns:
        The float orientation (yaw angle) of the label
    """
    R = [quat2rotmat(l.quaternion) for l in labels]
    v = np.array([1, 0, 0])[:, np.newaxis]
    orientations = np.matmul(R, v)
    return np.arctan2(orientations[:, 1, 0], orientations[:, 0, 0])
