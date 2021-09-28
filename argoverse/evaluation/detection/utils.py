# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
"""Detection utilities for the Argoverse detection leaderboard.

Accepts detections (in Argoverse ground truth format) and ground truth labels
for computing evaluation metrics for 3d object detection. We have five different,
metrics: mAP, ATE, ASE, AOE, and DCS. A true positive for mAP is defined as the
highest confidence prediction within a specified Euclidean distance threshold
from a bird's-eye view. We prefer these metrics instead of IoU due to the
increased interpretability of the error modes in a set of detections.
"""

import copy
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import DefaultDict, List, NamedTuple, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from argoverse.data_loading.object_label_record import ObjectLabelRecord, read_label
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t, read_city_name
from argoverse.evaluation.detection.constants import (
    COMPETITION_CLASSES,
    MAX_NORMALIZED_AOE,
    MAX_NORMALIZED_ASE,
    MAX_NUM_BOXES,
    MAX_SCALE_ERROR,
    MAX_YAW_ERROR,
    MIN_AP,
    MIN_CDS,
    N_TP_ERRORS,
)
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat_argo2scipy_vectorized

matplotlib.use("Agg")  # isort:skip
import matplotlib.pyplot as plt  # isort:skip  # noqa: E402

logger = logging.getLogger(__name__)

_PathLike = Union[str, "os.PathLike[str]"]


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


class DetectionCfg(NamedTuple):
    """Instantiates a DetectionCfg object for configuring a DetectionEvaluator.

    Args:
        affinity_threshs: Affinity thresholds for determining a true positive.
        affinity_fn_type: Type of affinity function to be used for calculating average precision.
        n_rec_samples: Number of recall points to sample uniformly in [0, 1]. Default to 101 recall samples.
        tp_thresh: Center distance threshold for the true positive metrics (in meters).
        dt_classes: Detection classes for evaluation.
        dt_metric: Detection metric to use for filtering of both detections and ground truth annotations.
        max_dt_range: The max distance (under a specific metric in meters) for a detection or ground truth to be
            considered for evaluation.
        save_figs: Flag to save figures.
        tp_normalization_terms: Normalization constants for ATE, ASE, and AOE.
        summary_default_vals: Evaluation summary default values.
        eval_only_roi_instances: Only use dets and ground truth that lie within region of interest during eval.
        map_root: Root directory for map files.
    """

    affinity_threshs: List[float] = [0.5, 1.0, 2.0, 4.0]  # Meters
    affinity_fn_type: AffFnType = AffFnType.CENTER
    n_rec_samples: int = 101
    tp_thresh: float = 2.0  # Meters
    dt_classes: List[str] = COMPETITION_CLASSES
    dt_metric: FilterMetric = FilterMetric.EUCLIDEAN
    max_dt_range: float = 100.0  # Meters
    save_figs: bool = False
    tp_normalization_terms: np.ndarray = np.array([tp_thresh, MAX_SCALE_ERROR, MAX_YAW_ERROR])
    summary_default_vals: np.ndarray = np.array([MIN_AP, tp_thresh, MAX_NORMALIZED_ASE, MAX_NORMALIZED_AOE, MIN_CDS])
    eval_only_roi_instances: bool = True
    map_root: _PathLike = Path(__file__).parent.parent.parent.parent / "map_files"  # argoverse-api/map_files


@dataclass
class AccumulateJob:
    """Dataclass args for running `accumulate`.

    Args:
        dt_root_fpath: Detections root folder file path.
        gt_fpath: Ground truth file path.
        cfg: Detection configuration.
        avm: Argoverse map object.
    """

    dt_root_fpath: Path
    gt_fpath: Path
    cfg: DetectionCfg
    avm: Optional[ArgoverseMap]


def accumulate(job: AccumulateJob) -> Tuple[DefaultDict[str, np.ndarray], DefaultDict[str, int]]:
    """Accumulate the true/false positives (boolean flags) and true positive errors for each class.

    Args:
        job: Accumulate job.

    Returns:
        cls_to_accum: Class to accumulated statistics dictionary of shape |C| -> (N, K + S) where C
            is the number of detection classes, K is the number of true positive thresholds used for
            AP computation, and S is the number of true positive errors.
        cls_to_ninst: Mapping of shape |C| -> (1,) the class names to the number of instances in the ground
            truth dataset.
    """
    dt_root_fpath, gt_fpath, cfg, avm = job.dt_root_fpath, job.gt_fpath, job.cfg, job.avm

    log_id = gt_fpath.parents[1].stem
    logger.debug(f"log_id = {log_id}")
    ts = int(gt_fpath.stem.split("_")[-1])

    dt_fpath = dt_root_fpath / f"{log_id}/per_sweep_annotations_amodal/" f"tracked_object_labels_{ts}.json"

    dts = np.array(read_label(str(dt_fpath)))
    gts = np.array(read_label(str(gt_fpath)))

    if cfg.eval_only_roi_instances and avm is not None:
        # go up 3 levels, because hierarchy is as follows:
        # {gt_root_fpath}/{log_id}/per_sweep_annotations_amodal/{gt_root_fname}
        gt_root_fpath = Path(gt_fpath).parents[2]
        city_SE3_egovehicle = get_city_SE3_egovehicle_at_sensor_t(ts, str(gt_root_fpath), log_id)
        if city_SE3_egovehicle is not None:
            log_city_name = read_city_name(os.path.join(gt_root_fpath, log_id, "city_info.json"))
            if dts.shape[0] > 0:
                dts = filter_objs_to_roi(dts, avm, city_SE3_egovehicle, log_city_name)
            if gts.shape[0] > 0:
                gts = filter_objs_to_roi(gts, avm, city_SE3_egovehicle, log_city_name)

    cls_to_accum = defaultdict(list)
    cls_to_ninst = defaultdict(int)
    for class_name in cfg.dt_classes:
        dt_filtered = filter_instances(
            dts,
            class_name,
            filter_metric=cfg.dt_metric,
            max_detection_range=cfg.max_dt_range,
        )
        gt_filtered = filter_instances(
            gts,
            class_name,
            filter_metric=cfg.dt_metric,
            max_detection_range=cfg.max_dt_range,
        )
        gt_filtered = remove_duplicate_instances(gt_filtered, cfg)

        logger.debug(f"{dt_filtered.shape[0]} detections")
        logger.debug(f"{gt_filtered.shape[0]} ground truth")
        if dt_filtered.shape[0] > 0:
            ranked_dts, ranked_scores = rank(dt_filtered)

            metrics = assign(ranked_dts, gt_filtered, cfg)
            cls_to_accum[class_name] = np.hstack((metrics, ranked_scores[:, None]))

        cls_to_ninst[class_name] = gt_filtered.shape[0]
    return cls_to_accum, cls_to_ninst


def remove_duplicate_instances(instances: np.ndarray, cfg: DetectionCfg) -> np.ndarray:
    """Remove any duplicate cuboids in ground truth.

    Any ground truth cuboid of the same object class that shares the same centroid
    with another is considered a duplicate instance.

    We first form an (N,N) affinity matrix with entries equal to negative distance.
    We then find rows in the affinity matrix with more than one zero, and
    then for each such row, we choose only the first column index with value zero.

    Args:
       instances: array of length (M,), each entry is an ObjectLabelRecord
       cfg: Detection configuration.

    Returns:
       array of length (N,) where N <= M, each entry is a unique ObjectLabelRecord
    """
    if len(instances) == 0:
        return instances
    assert isinstance(instances, np.ndarray)

    # create affinity matrix as inverse distance to other objects
    affinity_matrix = compute_affinity_matrix(copy.deepcopy(instances), copy.deepcopy(instances), cfg.affinity_fn_type)

    row_idxs, col_idxs = np.where(affinity_matrix == 0)
    # find the indices where each row index appears for the first time
    unique_row_idxs, unique_element_idxs = np.unique(row_idxs, return_index=True)
    # choose the first instance in each column where repeat occurs
    first_col_idxs = col_idxs[unique_element_idxs]
    # eliminate redundant column indices
    unique_ids = np.unique(first_col_idxs)
    return instances[unique_ids]


def assign(dts: np.ndarray, gts: np.ndarray, cfg: DetectionCfg) -> np.ndarray:
    """Attempt assignment of each detection to a ground truth label.

    Args:
        dts: Detections of shape (N,).
        gts: Ground truth labels of shape (M,).
        cfg: Detection configuration.

    Returns:
        metrics: Matrix of true/false positive concatenated with true positive errors (N, K + S) where K is the number
            of true positive thresholds used for AP computation and S is the number of true positive errors.
    """

    n_threshs = len(cfg.affinity_threshs)
    metrics = np.zeros((dts.shape[0], n_threshs + N_TP_ERRORS))

    # Set the true positive metrics to np.nan since error is undefined on false positives.
    metrics[:, n_threshs : n_threshs + N_TP_ERRORS] = np.nan
    if gts.shape[0] == 0:
        return metrics

    affinity_matrix = compute_affinity_matrix(dts, gts, cfg.affinity_fn_type)

    # Get the GT label for each max-affinity GT label, detection pair.
    gt_matches = affinity_matrix.argmax(axis=1)[np.newaxis, :]

    # The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
    # We want to take the corresponding affinity for each of the initial assignments using `gt_matches`.
    # The following line grabs the max affinity for each detection to a ground truth label.
    affinities = np.take_along_axis(affinity_matrix.T, gt_matches, axis=0).squeeze(0)

    # Find the indices of the "first" detection assigned to each GT.
    unique_gt_matches, unique_dt_matches = np.unique(gt_matches, return_index=True)
    for i, thresh in enumerate(cfg.affinity_threshs):

        # `tp_mask` may need to be defined differently with other affinities.
        tp_mask = affinities[unique_dt_matches] > -thresh
        metrics[unique_dt_matches, i] = tp_mask

        # Only compute true positive error when `thresh` is equal to the tp threshold.
        is_tp_thresh = thresh == cfg.tp_thresh
        # Ensure that there are true positives of the respective class in the frame.
        has_true_positives = np.count_nonzero(tp_mask) > 0

        if is_tp_thresh and has_true_positives:
            dt_tp_indices = unique_dt_matches[tp_mask]
            gt_tp_indices = unique_gt_matches[tp_mask]

            # Form DataFrame of shape (N, D) where D is the number of attributes in `ObjectLabelRecord`.
            dt_df = pd.DataFrame([dt.__dict__ for dt in dts[dt_tp_indices]])
            gt_df = pd.DataFrame([gt.__dict__ for gt in gts[gt_tp_indices]])

            trans_error = dist_fn(dt_df, gt_df, DistFnType.TRANSLATION)
            scale_error = dist_fn(dt_df, gt_df, DistFnType.SCALE)
            orient_error = dist_fn(dt_df, gt_df, DistFnType.ORIENTATION)

            metrics[dt_tp_indices, n_threshs : n_threshs + N_TP_ERRORS] = np.vstack(
                (trans_error, scale_error, orient_error)
            ).T
    return metrics


def filter_objs_to_roi(
    instances: np.ndarray, avm: ArgoverseMap, city_SE3_egovehicle: SE3, city_name: str
) -> np.ndarray:
    """Filter objects to the region of interest (5 meter dilation of driveable area).

    We ignore instances outside of region of interest (ROI) during evaluation.

    Args:
        instances: Numpy array of shape (N,) with ObjectLabelRecord entries
        avm: Argoverse map object
        city_SE3_egovehicle: pose of egovehicle within city map at time of sweep
        city_name: name of city where log was captured

    Returns:
        instances_roi: objects with any of 4 cuboid corners located within ROI
    """
    # for each cuboid, get its 4 corners in the egovehicle frame
    corners_egoframe = np.vstack([dt.as_2d_bbox() for dt in instances])
    corners_cityframe = city_SE3_egovehicle.transform_point_cloud(corners_egoframe)
    corner_within_roi = avm.get_raster_layer_points_boolean(corners_cityframe, city_name, "roi")
    # check for each cuboid if any of its 4 corners lies within the ROI
    is_within_roi = corner_within_roi.reshape(-1, 4).any(axis=1)
    instances_roi = instances[is_within_roi]
    return instances_roi


def filter_instances(
    instances: List[ObjectLabelRecord],
    target_class_name: str,
    filter_metric: FilterMetric,
    max_detection_range: float,
) -> np.ndarray:
    """Filter object instances based on a set of conditions (class name and distance from egovehicle).

    Args:
        instances: Instances to be filtered (N,), either detections or ground truth object labels
        target_class_name: Name of the class of interest.
        filter_metric: Range metric used for filtering.
        max_detection_range: Maximum distance for range filtering.

    Returns:
        Filtered annotations.
    """
    instances = np.array([instance for instance in instances if instance.label_class == target_class_name])

    if filter_metric == FilterMetric.EUCLIDEAN:
        centers = np.array([dt.translation for dt in instances])
        filtered_instances = np.array([])

        if centers.shape[0] > 0:
            dt_dists = np.linalg.norm(centers, axis=1)
            filtered_instances = instances[dt_dists < max_detection_range]
    else:
        raise NotImplementedError("This filter metric is not implemented!")

    return filtered_instances


def rank(dts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rank the detections in descending order according to score (detector confidence).


    Args:
        dts: Array of `ObjectLabelRecord` objects. (N,).

    Returns:
        ranked_dts: Array of `ObjectLabelRecord` objects ranked by score (N,) where N <= MAX_NUM_BOXES.
        ranked_scores: Array of floats sorted in descending order (N,) where N <= MAX_NUM_BOXES.
    """
    scores = np.array([dt.score for dt in dts.tolist()])
    ranks = scores.argsort()[::-1]

    ranked_dts = dts[ranks]
    ranked_scores = scores[ranks]

    # Ensure the number of boxes considered per class is at most `MAX_NUM_BOXES`.
    return ranked_dts[:MAX_NUM_BOXES], ranked_scores[:MAX_NUM_BOXES]


def interp(prec: np.ndarray, method: InterpType = InterpType.ALL) -> np.ndarray:
    """Interpolate the precision over all recall levels. See equation 2 in
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    for more information.

    Args:
        prec: Precision at all recall levels (N,).
        method: Accumulation method.

    Returns:
        prec_interp: Interpolated precision at all recall levels (N,).
    """
    if method == InterpType.ALL:
        prec_interp = np.maximum.accumulate(prec[::-1])[::-1]
    else:
        raise NotImplementedError("This interpolation method is not implemented!")
    return prec_interp


def compute_affinity_matrix(
    dts: List[ObjectLabelRecord], gts: List[ObjectLabelRecord], metric: AffFnType
) -> np.ndarray:
    """Calculate the affinity matrix between detections and ground truth labels,
    using a specified affinity function type.

    Args:
        dts: Detections (N,).
        gts: Ground truth labels (M,).
        metric: Affinity metric type.

    Returns:
        sims: Affinity scores between detections and ground truth annotations (N, M).
    """
    if metric == AffFnType.CENTER:
        dt_centers = np.array([dt.translation for dt in dts])
        gt_centers = np.array([gt.translation for gt in gts])
        sims = -cdist(dt_centers, gt_centers)
    else:
        raise NotImplementedError("This similarity metric is not implemented!")
    return sims


def calc_ap(gt_ranked: np.ndarray, recalls_interp: np.ndarray, ninst: int) -> Tuple[float, np.ndarray]:
    """Compute precision and recall, interpolated over n fixed recall points.

    Args:
        gt_ranked: Ground truths, ranked by confidence.
        recalls_interp: Interpolated recall values.
        ninst: Number of instances of this class.
    Returns:
        avg_precision: Average precision.
        precisions_interp: Interpolated precision values.
    """
    tp = gt_ranked

    cumulative_tp = np.cumsum(tp, dtype=int)
    cumulative_fp = np.cumsum(~tp, dtype=int)
    cumulative_fn = ninst - cumulative_tp

    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + np.finfo(float).eps)
    recalls = cumulative_tp / (cumulative_tp + cumulative_fn)
    precisions = interp(precisions)
    precisions_interp = np.interp(recalls_interp, recalls, precisions, right=0)
    avg_precision = precisions_interp.mean()
    return avg_precision, precisions_interp


def dist_fn(dts: pd.DataFrame, gts: pd.DataFrame, metric: DistFnType) -> np.ndarray:
    """Distance functions between detections and ground truth.

    Args:
        dts: Detections (N, D) where D is the number of attributes in `ObjectLabelRecord`.
        gts: Ground truth labels (N, D) where D is the number of attributes in `ObjectLabelRecord`.
        metric: Distance function type.

    Returns:
        Distance between the detections and ground truth, using the provided metric (N,).
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

        orientation_errors = wrap_angle(dt_yaws - gt_yaws)
        return orientation_errors
    else:
        raise NotImplementedError("This distance metric is not implemented!")


def iou_aligned_3d(dt_dims: pd.DataFrame, gt_dims: pd.DataFrame) -> np.ndarray:
    """Calculate the 3d, axis-aligned (vertical axis alignment) intersection-over-union (IoU)
    between the detections and the ground truth labels. Both objects are aligned to their
    +x axis and their centroids are placed at the origin before computation of the IoU.

    Args:
        dt_dims: Detections (N, 3).
        gt_dims: Ground truth labels (N, 3).

    Returns:
        Intersection-over-union between the detections and their assigned ground
        truth labels (N,).

    """
    inter = np.minimum(dt_dims, gt_dims).prod(axis=1)
    union = np.maximum(dt_dims, gt_dims).prod(axis=1)
    return (inter / union).values


def wrap_angle(angles: np.ndarray, period: float = np.pi) -> np.ndarray:
    """Map angles (in radians) from domain [-∞, ∞] to [0, π). This function is
        the inverse of `np.unwrap`.

    Returns:
        Angles (in radians) mapped to the interval [0, π).
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


def plot(rec_interp: np.ndarray, prec_interp: np.ndarray, cls_name: str, figs_fpath: Path) -> Path:
    """Plot and save the precision recall curve.

    Args:
        rec_interp: Interpolated recall data of shape (N,).
        prec_interp: Interpolated precision data of shape (N,).
        cls_name: Class name.
        figs_fpath: Path to the folder which will contain the output figures.
    Returns:
        dst_fpath: Plot file path.
    """
    plt.plot(rec_interp, prec_interp)
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    dst_fpath = Path(f"{figs_fpath}/{cls_name}.png")
    plt.savefig(dst_fpath)
    plt.close()
    return dst_fpath
