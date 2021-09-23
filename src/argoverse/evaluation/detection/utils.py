# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Detection utilities for the Argoverse detection leaderboard.

Accepts detections (in Argoverse ground truth format) and ground truth labels
for computing evaluation metrics for 3d object detection. We have five different,
metrics: mAP, ATE, ASE, AOE, and DCS. A true positive for mAP is defined as the
highest confidence prediction within a specified Euclidean distance threshold
from a bird's-eye view. We prefer these metrics instead of IoU due to the
increased interpretability of the error modes in a set of detections.
"""

import logging
import os
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Final, List, NamedTuple, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from argoverse.evaluation.detection.constants import (COMPETITION_CLASSES,
                                                      MAX_NORMALIZED_AOE,
                                                      MAX_NORMALIZED_ASE,
                                                      MAX_NUM_BOXES,
                                                      MAX_SCALE_ERROR,
                                                      MAX_YAW_ERROR, MIN_AP,
                                                      MIN_CDS)
from pandas import DataFrame
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")  # isort:skip
import matplotlib.pyplot as plt  # isort:skip  # noqa: E402

logger = logging.getLogger(__name__)

_PathLike = Union[str, "os.PathLike[str]"]

EPS: Final[float] = np.finfo(float).eps


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
    tp_normalization_terms: np.ndarray = np.array(
        [tp_thresh, MAX_SCALE_ERROR, MAX_YAW_ERROR]
    )
    summary_default_vals: np.ndarray = np.array(
        [MIN_AP, tp_thresh, MAX_NORMALIZED_ASE, MAX_NORMALIZED_AOE, MIN_CDS]
    )
    eval_only_roi_instances: bool = True
    map_root: _PathLike = (
        Path(__file__).parent.parent.parent.parent / "map_files"
    )  # argoverse-api/map_files
    splits: Tuple[str, ...] = ("val",)


def accumulate(
    job: Tuple[DataFrame, DataFrame, DataFrame, DetectionCfg]
) -> Tuple[DataFrame, Dict[str, int]]:
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
    dts, gts, poses, cfg, avm = job

    dts = dts.sort_values("tov_ns").reset_index(drop=True)
    gts = gts.sort_values("tov_ns").reset_index(drop=True)
    poses = poses.sort_values("tov_ns").reset_index(drop=True)
    if cfg.eval_only_roi_instances and avm is not None:
        dts = filter_objs_to_roi(dts, poses, avm)
        gts = filter_objs_to_roi(gts, poses, avm)

    dts_filtered = filter_instances(dts, cfg)
    gts_filtered = filter_instances(gts, cfg)

    metrics: List[DataFrame] = []
    cls_to_ninst: Dict[str, int] = {}
    for label_class in cfg.dt_classes:
        label_dts = dts_filtered[dts_filtered["label_class"] == label_class]
        label_gts = gts_filtered[gts_filtered["label_class"] == label_class]
        label_gts = remove_duplicate_instances(label_gts, cfg)
        cls_to_ninst[label_class] = len(label_gts)

        if dts_filtered.shape[0] > 0:
            ranked_dts = rank(label_dts)

            class_metrics = assign(ranked_dts, label_gts, cfg)
            class_metrics["label_class"] = label_class
            if class_metrics.shape[0] > 0:
                metrics.append(class_metrics)

    if len(metrics) > 0:
        return pd.concat(metrics), cls_to_ninst
    return DataFrame(metrics), cls_to_ninst


def remove_duplicate_instances(instances: DataFrame, cfg: DetectionCfg) -> DataFrame:
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

    # create affinity matrix as inverse distance to other objects
    affinity_matrix = compute_affinity_matrix(
        instances, instances, cfg.affinity_fn_type
    )

    row_idxs, col_idxs = np.where(affinity_matrix == 0)

    # find the indices where each row index appears for the first time
    _, unique_element_idxs = np.unique(row_idxs, return_index=True)

    # choose the first instance in each column where repeat occurs
    first_col_idxs = col_idxs[unique_element_idxs]

    # eliminate redundant column indices
    unique_ids = np.unique(first_col_idxs)

    return instances.iloc[unique_ids].reset_index(drop=True)


def assign(dts: DataFrame, gts: DataFrame, cfg: DetectionCfg) -> DataFrame:
    """Attempt assignment of each detection to a ground truth label.

    Args:
        dts: Detections of shape (N,).
        gts: Ground truth labels of shape (M,).
        cfg: Detection configuration.

    Returns:
        metrics: Matrix of true/false positive concatenated with true positive errors (N, K + S) where K is the number
            of true positive thresholds used for AP computation and S is the number of true positive errors.
    """
    tp_cols = ["ATE", "ASE", "AOE"]
    cols = list(map(str, cfg.affinity_threshs)) + tp_cols
    ncols = len(cols)
    nthreshs = len(cfg.affinity_threshs)

    data = np.zeros((dts.shape[0], ncols))

    metrics = DataFrame(data, columns=cols)

    aff_dtype = {str(x): bool for x in cfg.affinity_threshs}
    metrics = metrics.astype(aff_dtype)
    metrics.iloc[:, nthreshs : nthreshs + ncols] = np.nan
    metrics["score"] = dts["score"]

    # Set the true positive metrics to np.nan since error is undefined on false positives.
    # metrics[:, n_threshs : n_threshs + N_TP_ERRORS] = np.nan
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
        metrics.iloc[unique_dt_matches, i] = tp_mask

        # Only compute true positive error when `thresh` is equal to the tps threshold.
        is_tp_thresh = thresh == cfg.tp_thresh

        # Ensure that there are true positives of the respective class in the frame.
        has_true_positives = np.count_nonzero(tp_mask) > 0
        if is_tp_thresh and has_true_positives:
            dt_tp_indices = unique_dt_matches[tp_mask]
            gt_tp_indices = unique_gt_matches[tp_mask]

            dt_df = dts.iloc[dt_tp_indices]
            gt_df = gts.iloc[gt_tp_indices]

            trans_error = dist_fn(dt_df, gt_df, DistFnType.TRANSLATION)
            scale_error = dist_fn(dt_df, gt_df, DistFnType.SCALE)
            orient_error = dist_fn(dt_df, gt_df, DistFnType.ORIENTATION)

            tp_metrics = np.vstack((trans_error, scale_error, orient_error)).T
            metrics.loc[dt_tp_indices, tp_cols] = tp_metrics
    return metrics


def rank(dts: DataFrame) -> DataFrame:
    """Rank the detections in descending order according to score (detector confidence).


    Args:
        dts: Array of `ObjectLabelRecord` objects. (N,).

    Returns:
        ranked_dts: Array of `ObjectLabelRecord` objects ranked by score (N,) where N <= MAX_NUM_BOXES.
        ranked_scores: Array of floats sorted in descending order (N,) where N <= MAX_NUM_BOXES.
    """

    dts = dts.sort_values("score", ascending=False)
    return dts[:MAX_NUM_BOXES].reset_index(drop=True)


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
    dts: DataFrame, gts: DataFrame, metric: AffFnType
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
        cols = ["x", "y", "z"]
        dt_centers = dts[cols]
        gt_centers = gts[cols]
        sims = -cdist(dt_centers, gt_centers)
    else:
        raise NotImplementedError("This similarity metric is not implemented!")
    return sims


def calc_ap(
    tps: np.ndarray, recalls_interp: np.ndarray, ninst: int
) -> Tuple[float, np.ndarray]:
    """Compute precision and recall, interpolated over n fixed recall points.

    Args:
        tps: Ground truths, ranked by confidence.
        recalls_interp: Interpolated recall values.
        ninst: Number of instances of this class.
    Returns:
        avg_precision: Average precision.
        precisions_interp: Interpolated precision values.
    """
    cumulative_tps = tps.cumsum()
    cumulative_fps = (~tps).cumsum()
    cumulative_fns = ninst - cumulative_tps

    precisions = cumulative_tps / (cumulative_tps + cumulative_fps + EPS)
    recalls = cumulative_tps / (cumulative_tps + cumulative_fns)
    precisions = interp(precisions)

    precisions_interp = np.interp(recalls_interp, recalls, precisions, right=0)
    avg_precision = precisions_interp.mean()
    return avg_precision, precisions_interp


def dist_fn(dts: DataFrame, gts: DataFrame, metric: DistFnType) -> np.ndarray:
    """Distance functions between detections and ground truth.

    Args:
        dts: Detections (N, D) where D is the number of attributes in `ObjectLabelRecord`.
        gts: Ground truth labels (N, D) where D is the number of attributes in `ObjectLabelRecord`.
        metric: Distance function type.

    Returns:
        Distance between the detections and ground truth, using the provided metric (N,).
    """
    if metric == DistFnType.TRANSLATION:
        cols = ["x", "y", "z"]

        dt_centers = dts[cols].reset_index(drop=True)
        gt_centers = gts[cols].reset_index(drop=True)

        trans_errors = np.linalg.norm(dt_centers - gt_centers, axis=1)
        return trans_errors
    elif metric == DistFnType.SCALE:
        cols = ["length", "width", "height"]
        dt_dims = dts[cols].reset_index(drop=True)
        gt_dims = gts[cols].reset_index(drop=True)

        scale_errors = 1 - iou_aligned_3d(dt_dims, gt_dims)
        return scale_errors
    elif metric == DistFnType.ORIENTATION:
        # Re-order quaternions to go from Argoverse format to scipy format, then the third euler angle (z) is yaw.

        dt_quats = dts[["qx", "qy", "qz", "qw"]]
        gt_quats = gts[["qx", "qy", "qz", "qw"]]

        dt_yaws = R.from_quat(dt_quats).as_euler("xyz")[:, -1]
        gt_yaws = R.from_quat(gt_quats).as_euler("xyz")[:, -1]

        orientation_errors = wrap_angle(dt_yaws - gt_yaws)
        return orientation_errors
    else:
        raise NotImplementedError("This distance metric is not implemented!")


def iou_aligned_3d(dt_dims: DataFrame, gt_dims: DataFrame) -> DataFrame:
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
    return np.divide(inter, union)


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


def plot(
    rec_interp: np.ndarray, prec_interp: np.ndarray, cls_name: str, figs_fpath: Path
) -> Path:
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


def filter_objs_to_roi(
    cuboids: DataFrame, poses: DataFrame, avm: ArgoverseMap
) -> DataFrame:
    cuboids = cuboids.sort_values("tov_ns").reset_index(drop=True)
    poses = poses.sort_values("tov_ns").reset_index(drop=True)

    cuboids_with_poses = pd.merge_asof(
        cuboids, poses, by="uuid", on="tov_ns", direction="nearest"
    )
    ego_SO3_gts = cuboids_with_poses[["qx_x", "qy_x", "qz_x", "qw_x"]]
    ego_SO3_gts = R.from_quat(ego_SO3_gts.to_numpy().repeat(4, axis=0))

    city_SO3_ego = cuboids_with_poses[["qx_y", "qy_y", "qz_y", "qw_y"]]
    city_SO3_ego = R.from_quat(city_SO3_ego.to_numpy().repeat(4, axis=0))

    unit_box = np.array(
        [[+1.0, +1.0, +1.0], [+1.0, -1.0, +1.0], [-1.0, +1.0, +1.0], [-1.0, -1.0, +1.0]]
    )

    xyz_obj = cuboids_with_poses[["x", "y", "z"]].to_numpy()
    pts_obj = (
        unit_box[:, None] * cuboids_with_poses[["length", "width", "height"]].to_numpy()
    )
    pts_ego = ego_SO3_gts.apply(pts_obj.reshape(-1, 3)) + xyz_obj.repeat(
        unit_box.shape[0], axis=0
    ).reshape(-1, 3)

    pts_city = city_SO3_ego.apply(pts_ego) + cuboids_with_poses[
        ["tx", "ty", "tz"]
    ].to_numpy().repeat(unit_box.shape[0], axis=0).reshape(-1, 3)
    corner_within_roi = avm.get_raster_layer_points_boolean(
        pts_city[..., :2], "MIA", "roi"
    )

    corner_within_roi = corner_within_roi.reshape(-1, 4)
    is_within_roi = corner_within_roi.any(axis=1)
    return cuboids_with_poses[is_within_roi]


def filter_instances(cuboids: DataFrame, cfg: DetectionCfg) -> DataFrame:
    outputs = []
    for class_name in cfg.dt_classes:
        class_mask = cuboids.label_class == class_name
        classes = cuboids[class_mask]
        norm = classes[["x", "y", "z"]].pow(2).sum(axis=1).pow(0.5)
        mask = norm < cfg.max_dt_range
        outputs.append(classes[mask])
    return pd.concat(outputs)
