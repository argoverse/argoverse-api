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
from argoverse.utils.transform import quat_argo2scipy_vectorized


class AffFnType(Enum):
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
    """Filter the GT annotations based on a set of conditions (class name and distance from egovehicle).

    Args:
        instances: The instances to be filtered (N, ).
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


def rank(dts: List[ObjectLabelRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Get the rankings for the detections, according to detector confidence.

    Args:
        dts: Detections (N,).

    Returns:
        ranks: The ranking for the detections (N, ).
        scores: The detection scores (N, ).
    """
    scores = np.array([dt.score for dt in dts])
    ranks = scores.argsort()[::-1]
    ranked_detections = dts[ranks]
    return ranked_detections, scores[:, np.newaxis]


def interp(prec: np.ndarray, method: InterpType = InterpType.ALL) -> np.ndarray:
    """Interpolate the precision over all recall levels.

    Args:
        prec: Precision at all recall levels (N, ).
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
    dts: List[ObjectLabelRecord], gts: List[ObjectLabelRecord], metric: AffFnType
) -> np.ndarray:
    """Calculate the match matrix between detections and ground truth labels,
    using a specified affinity function.

    Args:
        dts: Detections (N, ).
        gts: Ground truth labels (M, ).
        metric: Similarity metric type.

    Returns:
        sims: Similarity scores between detections and ground truth annotations (N, M).
    """
    if metric == AffFnType.CENTER:
        dt_centers = np.array([dt.translation for dt in dts])
        gt_centers = np.array([gt.translation for gt in gts])
        sims = -cdist(dt_centers, gt_centers)
    else:
        raise NotImplemented("This similarity metric is not implemented!")
    return sims


def dist_fn(dts: pd.DataFrame, gts: pd.DataFrame, metric: DistFnType) -> np.ndarray:
    """Distance functions between detections and ground truth.

    Args:
        dts: Detections (N, ).
        gts: Ground truth labels (N, ).
        metric: Distance function type.

    Returns:
        Distance between the detections and ground truth, using the provided metric (N, ).
    """
    if metric == DistFnType.TRANSLATION:
        dt_centers = np.vstack(dts["translation"].array)
        gt_centers = np.vstack(gts["translation"].array)
        trans_errors = np.linalg.norm(dt_centers - gt_centers, axis=1)
        return trans_errors
    elif metric == DistFnType.SCALE:
        dt_dims = dts[["width", "length", "height"]]
        gt_dims = gts[["width", "length", "height"]]
        scale_errors = 1 - iou_aligned_3d(dt_dims, gt_dims)
        return scale_errors
    elif metric == DistFnType.ORIENTATION:
        # Re-order quaternions to go from Argoverse format to scipy format, then the third euler angle (z) is yaw.
        dt_quats = np.vstack(dts["quaternion"].array)
        dt_yaws = R.from_quat(quat_argo2scipy_vectorized(dt_quats)).as_euler("xyz")[:, 2]

        gt_quats = np.vstack(gts["quaternion"].array)
        gt_yaws = R.from_quat(quat_argo2scipy_vectorized(gt_quats)).as_euler("xyz")[:, 2]

        orientation_errors = wrap(dt_yaws - gt_yaws)
        return orientation_errors
    else:
        raise NotImplemented("This distance metric is not implemented!")


def iou_aligned_3d(dt_dims: pd.DataFrame, gt_dims: pd.DataFrame) -> np.ndarray:
    """Calculate the 3d, axis-aligned (vertical axis alignment) intersection-over-union (IoU)
    between the detections and the ground truth labels. Both objects are aligned to their
    +x axis and their centroids are placed at the origin before computation of the IoU.

    Args:
        dt_dims: Detections (N, 3).
        gt_dims: Ground truth labels (N, 3).
    
    Returns:
        Intersection-over-union between the detections and their assigned ground
        truth labels (N, ).

    """
    inter = np.minimum(dt_dims, gt_dims).prod(axis=1)
    union = np.maximum(dt_dims, gt_dims).prod(axis=1)
    return (inter / union).values


def wrap(angles: np.ndarray, period: float = np.pi) -> np.ndarray:
    """Map angles (in radians) from domain [-∞, ∞] to [0, π). This function is
        the inverse of `np.unwrap`.

    Returns:
        The angles (in radians) mapped to the interval [0, π).
    """

    # Map angles to [0, ∞].
    angles = np.abs(angles)

    # Calculate floor division and remainder simultaneously.
    divs, mods = np.divmod(angles, period)

    # Select angles which exceed specified period.
    angle_complement_mask = np.nonzero(divs)

    # Take set complement of `mods` w.r.t. the set [0, π].
    # `mods` must be nonzero, thus the image is the interval [0, π).
    angles[angle_complement_mask] = period - mods[angle_complement_mask]
    return angles
