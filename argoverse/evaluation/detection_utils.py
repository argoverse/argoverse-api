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
    IOU_2D = auto()
    IOU_3D = auto()


class DistFnType(Enum):
    TRANSLATION = auto()
    SCALE = auto()
    ORIENTATION = auto()


def filter_annos(annos: np.ndarray, target_class: str, range_metric="euclidean", max_dist=50) -> np.ndarray:
    """Filter the annotations based on a set of conditions.

    Args:
        annos: The annotations to be filtered.
        target_class: The name of the class of interest.
        range_metric: The range metric used for filtering.
    
    Returns:
        The filtered annotations.
    """
    annos = np.array([dt_anno for dt_anno in annos if dt_anno.label_class == target_class])

    if range_metric == "euclidean":
        centers = np.array([dt.translation for dt in annos])
        filtered_annos = np.array([])

        if centers.shape[0] > 0:
            dt_dists = np.linalg.norm(centers, axis=1)
            filtered_annos = annos[dt_dists < max_dist]

        return filtered_annos


def get_ranks(dt_annos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the rankings for the detection annotations.

    Args:
        dt_annos: Detection annotations.

    Returns:
        scores: The detection scores.
        ranks: The ranking for the detections.

    """
    scores = np.array([dt_anno.score for dt_anno in dt_annos])
    ranks = scores.argsort()[::-1]
    return np.expand_dims(scores, 1)[ranks], ranks


def interp(prec: np.ndarray, method="all") -> np.ndarray:
    """Interpolate the precision over all recall levels.

    Args:
        prec: Precision at all recall levels.
        method: Accumulation method.
    
    Returns:
        Interpolated precision at all recall levels.
    """
    if method == "all":
        return np.maximum.accumulate(prec[::-1])[::-1]


def sim_fn(dt_annos: np.ndarray, gt_annos: np.ndarray, metric: SimFnType) -> np.ndarray:
    if metric == SimFnType.CENTER:
        dt_centers = np.array([dt.translation for dt in dt_annos])
        gt_centers = np.array([gt.translation for gt in gt_annos])
        sims = -cdist(dt_centers, gt_centers)
    elif metric == SimFnType.IOU_2D:
        raise NotImplemented("This similarity metric is not implemented!")
    elif metric == SimFnType.IOU_3D:
        raise NotImplemented("This similarity metric is not implemented!")
    else:
        raise NotImplemented("This similarity metric is not implemented!")
    return sims


def get_error_types(match_scores: np.ndarray, thresh: float, metric: SimFnType) -> np.ndarray:
    # Euclidean distance represented as a "similarity" metric.
    if metric == SimFnType.CENTER:
        return match_scores > -thresh
    else:
        raise NotImplemented("This similarity metric is not implemented!")


def dist_fn(dt_df: pd.DataFrame, gt_df: pd.DataFrame, metric: DistFnType) -> np.ndarray:
    if metric == DistFnType.TRANSLATION:
        dt_centers = np.vstack(dt_df["translation"].array)
        gt_centers = np.vstack(gt_df["translation"].array)
        trans_errors = np.linalg.norm(dt_centers - gt_centers, axis=1)
        return trans_errors
    elif metric == DistFnType.SCALE:
        dt_dims = dt_df[["width", "length", "height"]]
        gt_dims = gt_df[["width", "length", "height"]]
        inter = np.minimum(dt_dims, gt_dims).prod(axis=1)
        union = np.maximum(dt_dims, gt_dims).prod(axis=1)
        scale_errors = 1 - (inter / union)
        return scale_errors
    elif metric == DistFnType.ORIENTATION:
        dt_yaws = R.from_quat(np.vstack(dt_df["quaternion"].array)[:, [3, 0, 1, 2]]).as_euler("xyz")[:, 2]
        gt_yaws = R.from_quat(np.vstack(gt_df["quaternion"].array)[:, [3, 0, 1, 2]]).as_euler("xyz")[:, 2]
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
