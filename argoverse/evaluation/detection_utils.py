# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
"""Detection utilities for the Argoverse detection leaderboard.

Accepts detections (in Argoverse ground truth format) and ground truth labels
for computing evaluation metrics for 3d object detection. We have five different,
metrics: mAP, ATE, ASE, AOE, and DCS. A true positive for mAP is defined as the
highest confidence prediction within a specified Euclidean distance threshold
from a bird's-eye view. We prefer these metrics instead of IoU due to the
increased interpretability of the error modes in a set of detections.

"""

from enum import Enum, auto
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.evaluation.eval_tracking import get_orientation_error_deg
from argoverse.utils.transform import quat_argo2scipy_vectorized


class SimFnType(Enum):
    CENTER = auto()


class DistFnType(Enum):
    TRANSLATION = auto()
    SCALE = auto()
    ORIENTATION = auto()


class InterpType(Enum):
    ALL = auto()


class FilterMetric(Enum):
    EUCLIDEAN = auto()


def filter_instances(
    instances: List[ObjectLabelRecord], target_class_name: str, filter_metric: FilterMetric, max_detection_range: float
) -> np.ndarray:
    """Filter the GT annotations based on a set of conditions (classname and distance from egovehicle).

    Args:
        instances: The instances to be filtered (N,).
        target_class_name: The name of the class of interest.
        filter_metric: The range metric used for filtering.
        max_detection_range: The maximum distance for range filtering.

    Returns:
        The filtered annotations.
    """
    instances = np.array([instance for instance in instances if instance.label_class == target_class_name])

    if filter_metric == FilterMetric.EUCLIDEAN:
        centers = np.array([dt.translation for dt in instances])
        filtered_annos = np.array([])

        if centers.shape[0] > 0:
            dt_dists = np.linalg.norm(centers, axis=1)
            filtered_annos = instances[dt_dists < max_detection_range]
    else:
        raise NotImplementedError("This filter metric is not implemented!")
    return filtered_annos


def get_ranks(dts: List[ObjectLabelRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Get the rankings for the detections.

    Args:
        dts: Detections (N,).

    Returns:
        scores: The detection scores (N,).
        ranks: The ranking for the detections (N,).
    """
    scores = np.array([dt.score for dt in dts])
    ranks = scores.argsort()[::-1]
    return np.expand_dims(scores, 1)[ranks], ranks


def interp(prec: np.ndarray, method: InterpType = InterpType.ALL) -> np.ndarray:
    """Interpolate the precision over all recall levels.

    Args:
        prec: Precision at all recall levels (N,).
        method: Accumulation method.

    Returns:
        prec_interp: Interpolated precision at all recall levels (N,).
    """
    if method == InterpType.ALL:
        prec_interp = np.maximum.accumulate(prec[::-1])[::-1]
    else:
        raise NotImplemented("This interpolation method is not implemented!")
    return prec_interp


def compute_affinity_matrix(
    dts: List[ObjectLabelRecord], gts: List[ObjectLabelRecord], metric: SimFnType
) -> np.ndarray:
    """Calculate the match matrix between detections and ground truth labels,
    using a specified similarity function.

    Args:
        dts: Detections (N,).
        gts: Ground truth labels (M,).
        metric: Similarity metric type.

    Returns:
        sims: Similarity scores between detections and ground truth annotations (N, M).
    """
    if metric == SimFnType.CENTER:
        dt_centers = np.array([dt.translation for dt in dts])
        gt_centers = np.array([gt.translation for gt in gts])
        sims = -cdist(dt_centers, gt_centers)
    else:
        raise NotImplemented("This similarity metric is not implemented!")
    return sims


def dist_fn(dts: pd.DataFrame, gts: pd.DataFrame, metric: DistFnType) -> np.ndarray:
    """Distance functions between detections and ground truth.

    Args:
        dts: Detections (N,).
        gts: Ground truth labels (M,).
        metric: Distance function type.

    Returns:
        Distance between the detections and ground truth, using the provided metric.
    """
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
        # re-order quaternions to go from Argoverse format to scipy format, then the third euler angle (z) is yaw
        dt_quats = np.vstack(dts["quaternion"].array)
        dt_yaws = R.from_quat(quat_argo2scipy_vectorized(dt_quats)).as_euler("xyz")[:, 2]

        gt_quats = np.vstack(gts["quaternion"].array)
        gt_yaws = R.from_quat(quat_argo2scipy_vectorized(gt_quats)).as_euler("xyz")[:, 2]

        signed_orientation_errors = normalize_angle(dt_yaws - gt_yaws)
        orientation_errors = np.abs(signed_orientation_errors)
        return orientation_errors
    else:
        raise NotImplemented("This distance metric is not implemented!")


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    """Map angle (in radians) from domain [-π, π] to [0, π).

    Returns:
        The angle (in radians) mapped to the interval [0, π].
    """
    period = 2 * np.pi
    phase_shift = np.pi
    return (angle + np.pi) % period - phase_shift
