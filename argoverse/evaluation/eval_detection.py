# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
"""Detection evaluation for the Argoverse detection leaderboard.

Evaluation:

    Precision/Recall

        1. Average Precision: Standard VOC-style average precision calculation
            except a true positive requires a bird's eye center distance of less
            than a predefined threshold.

    True Positive Errors

        All true positive errors -- as the name implies -- accumulate error solely
        when an object is a true positive match to a ground truth detection. The matching
        criterion is represented by `tp_thresh` in the DetectionCfg class. In our challege,
        we use a `tp_thresh` of 2 meters.

        1. Average Translation Error: The average Euclidean distance (center-based) between a
            detection and its ground truth assignment.
        2. Average Scale Error: The average intersection over union (IoU) after the prediction
            and assigned ground truth's pose has been aligned.
        3. Average Orientation Error: The average angular distance between the detection and
            the assigned ground truth. We choose the smallest angle between the two different
            headings when calculating the error.

    Composite Scores

        1. Composite Detection Score: The ranking metric for the detection leaderboard. This
            is computed as the product of mAP with the sum of the complements of the true positive
            errors (after normalization), i.e.,

            Average Translation Measure (ATM): ATE / TP_THRESHOLD; 0 <= 1 - ATE / TP_THRESHOLD <= 1
            Average Scaling Measure (ASM): 1 - ASE / 1;  0 <= 1 - ASE / 1 <= 1
            Average Orientation Measure (AOM): 1 - AOE / PI; 0 <= 1 - AOE / PI <= 1

            These (as well as AP) are averaged over each detection class to produce:

            mAP, mATM, mASM, mAOM.

            Lastly, the Composite Detection Score is computed as:

            CDS = mAP * (mATE + mASE + mAOE); 0 <= mAP * (mATE + mASE + mAOE) <= 1.

        ** In the case of no true positives under the specified threshold, the true positive measures
            will assume their upper bounds of 1.0. respectively.

Results:

    The results are represented as a (C + 1, P) table, where C + 1 represents the number of evaluation classes
    in addition to the mean statistics average across all classes, and P refers to the number of included statistics, 
    e.g. AP, ATE, ASE, AOE, CDS by default.

    Note: The `evaluate` function will use all available logical cores on the machine.

"""
import argparse
import logging
import os
from collections import defaultdict
from dataclasses import Field, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import DefaultDict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from pandas.core import frame

from argoverse.data_loading.object_classes import OBJ_CLASS_MAPPING_DICT
from argoverse.data_loading.object_label_record import read_label
from argoverse.evaluation.detection_utils import (
    AffFnType,
    DistFnType,
    FilterMetric,
    compute_affinity_matrix,
    dist_fn,
    filter_instances,
    interp,
    rank,
)

matplotlib.use("Agg")  # isort:skip


import matplotlib.pyplot as plt  # isort:skip


logger = logging.getLogger(__name__)


TP_ERROR_NAMES: List[str] = ["ATE", "ASE", "AOE"]
N_TP_ERRORS: int = len(TP_ERROR_NAMES)

STATISTIC_NAMES: List[str] = ["AP"] + TP_ERROR_NAMES + ["CDS"]

MAX_YAW_ERROR: float = np.pi

# Higher is better.
MIN_AP: float = 0.0
MIN_CDS: float = 0.0

# Lower is better.
MAX_NORMALIZED_ATE: float = 1.0
MAX_NORMALIZED_ASE: float = 1.0
MAX_NORMALIZED_AOE: float = 1.0

# Each measure is in [0, 1].
MEASURE_DEFAULT_VALUES: List[float] = [MIN_AP, MAX_NORMALIZED_ATE, MAX_NORMALIZED_ASE, MAX_NORMALIZED_AOE, MIN_CDS]

# Max number of boxes considered per class per scene.
MAX_NUM_BOXES: int = 500

SIGNIFICANT_DIGITS: float = 3


@dataclass
class DetectionCfg:
    """Instantiates a DetectionCfg object for configuring a DetectionEvaluator.

    Args:
        affinity_threshs: The affinity thresholds for determining a true positive.
        affinity_fn_type: The type of affinity function to be used for calculating average precision.
        n_rec_samples: Number of recall points to sample uniformly in [0, 1]. Default to 101 recall samples.
        tp_thresh: Center distance threshold for the true positive metrics (in meters).
        detection_classes: Detection classes for evaluation.
        detection_metric: The detection metric to use for filtering of both detections and ground truth annotations.
        max_detection_range: The max distance (under a specific metric in meters) for a detection or ground truth to be
            considered for evaluation.
        save_figs: Flag to save figures.
        tp_normalization_terms: Normalization constants for ATE, ASE, and AOE.
    """

    affinity_threshs: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])  # Meters
    affinity_fn_type: AffFnType = AffFnType.CENTER
    n_rec_samples: int = 101
    tp_thresh: float = 2.0  # Meters
    detection_classes: List[str] = field(default_factory=lambda: list(OBJ_CLASS_MAPPING_DICT.keys()))
    detection_metric: FilterMetric = FilterMetric.EUCLIDEAN
    max_detection_range: float = 100.0  # Meters
    save_figs: bool = False
    tp_normalization_terms: np.ndarray = field(init=False)

    def __post_init__(self):
        self.tp_normalization_terms: np.ndarray = np.array([self.tp_thresh, 1.0, MAX_YAW_ERROR])


@dataclass
class DetectionEvaluator:
    """Instantiates a DetectionEvaluator object for evaluation.

    Args:
        detection_fpath: The path to the folder which contains the detections.
        ground_truth_fpath: The path to the folder which contains all the logs.
        figures_fpath: The path to the folder which will contain the output figures.
        detection_cfg: The detection configuration settings.
    """

    detection_fpath: Path
    ground_truth_fpath: Path
    figures_fpath: Path
    detection_cfg: DetectionCfg = DetectionCfg()

    def evaluate(self) -> pd.DataFrame:
        """Evaluate detection output and return metrics. The multiprocessing
        library is used for parallel assignment between detections and ground truth
        annotations.

        Returns:
            Evaluation metrics of shape (C + 1, K) where C + 1 is the number of classes
            plus a row for their means. K refers to the number of evaluation metrics.
        """
        dt_fpaths = list(self.detection_fpath.glob("*/per_sweep_annotations_amodal/*.json"))
        gt_fpaths = list(self.ground_truth_fpath.glob("*/per_sweep_annotations_amodal/*.json"))

        assert len(dt_fpaths) == len(gt_fpaths)
        data: DefaultDict[str, np.ndarray] = defaultdict(list)
        cls_to_ninst: DefaultDict[str, int] = defaultdict(int)

        with Pool(os.cpu_count()) as p:
            accum = p.map(self.accumulate, gt_fpaths)

        for frame_stats, frame_cls_to_inst in accum:
            for cls_name, cls_stats in frame_stats.items():
                data[cls_name].append(cls_stats)
            for cls_name, num_inst in frame_cls_to_inst.items():
                cls_to_ninst[cls_name] += num_inst

        data = defaultdict(np.ndarray, {k: np.vstack(v) for k, v in data.items()})

        init_data = {dt_cls: MEASURE_DEFAULT_VALUES for dt_cls in self.detection_cfg.detection_classes}
        summary = pd.DataFrame.from_dict(init_data, orient="index", columns=STATISTIC_NAMES)
        summary_update = pd.DataFrame.from_dict(
            self.summarize(data, cls_to_ninst), orient="index", columns=STATISTIC_NAMES
        )

        summary.update(summary_update)
        summary = summary.round(SIGNIFICANT_DIGITS)
        summary.index = summary.index.str.title()

        summary.loc["Average Metrics"] = summary.mean().round(SIGNIFICANT_DIGITS)
        return summary

    def accumulate(self, gt_fpath: Path) -> Tuple[DefaultDict[str, np.ndarray], DefaultDict[str, int]]:
        """Accumulate the true/false positives (boolean flags) and true positive errors for each class.

        Args:
            gt_fpath: Ground truth file path.

        Returns:
            cls_to_accum: Class to accumulated statistics dictionary of shape |C| -> (N, K + S) where C
                is the number of detection classes, K is the number of true positive thresholds used for
                AP computation, and S is the number of true positive errors.
            cls_to_ninst: Mapping of shape |C| -> (1, ) the class names to the number of instances in the ground
                truth dataset.
        """
        log_id = gt_fpath.parents[1].stem
        logger.info(f"log_id = {log_id}")
        ts = gt_fpath.stem.split("_")[-1]

        dt_fpath = self.detection_fpath / f"{log_id}/per_sweep_annotations_amodal/" f"tracked_object_labels_{ts}.json"

        dts = np.array(read_label(str(dt_fpath)))
        gts = np.array(read_label(str(gt_fpath)))

        cls_to_accum = defaultdict(list)
        cls_to_ninst = defaultdict(int)
        for class_name in self.detection_cfg.detection_classes:
            dt_filtered = filter_instances(
                dts,
                class_name,
                filter_metric=self.detection_cfg.detection_metric,
                max_detection_range=self.detection_cfg.max_detection_range,
            )
            gt_filtered = filter_instances(
                gts,
                class_name,
                filter_metric=self.detection_cfg.detection_metric,
                max_detection_range=self.detection_cfg.max_detection_range,
            )

            logger.info(f"{dt_filtered.shape[0]} detections")
            logger.info(f"{gt_filtered.shape[0]} ground truth")
            if dt_filtered.shape[0] > 0:
                ranked_detections, scores = rank(dt_filtered)
                metrics = self.assign(ranked_detections, gt_filtered)
                cls_to_accum[class_name] = np.hstack((metrics, scores))

            cls_to_ninst[class_name] = gt_filtered.shape[0]
        return cls_to_accum, cls_to_ninst

    def assign(self, dts: np.ndarray, gts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Attempt assignment of each detection to a ground truth label.

        Args:
            dts: Detections of shape (N,).
            gts: Ground truth labels of shape (M,).

        Returns:
            metrics: Matrix of true/false positive concatenated with true positive errors (N, K + S) where K is the number
                of true positive thresholds used for AP computation and S is the number of true positive errors.
            scores: Corresponding scores for the true positives/false positives (N,)
        """

        # Ensure the number of boxes considered per class is at most `MAX_NUM_BOXES`.
        if dts.shape[0] > MAX_NUM_BOXES:
            dts = dts[:MAX_NUM_BOXES]

        n_threshs = len(self.detection_cfg.affinity_threshs)
        metrics = np.zeros((dts.shape[0], n_threshs + N_TP_ERRORS))

        # Set the true positive metrics to np.nan since error is undefined on false positives.
        metrics[:, n_threshs : n_threshs + N_TP_ERRORS] = np.nan
        if gts.shape[0] == 0:
            return metrics

        affinity_matrix = compute_affinity_matrix(dts, gts, self.detection_cfg.affinity_fn_type)

        # Get the GT label for each max-affinity GT label, detection pair.
        gt_matches = affinity_matrix.argmax(axis=1)[np.newaxis, :]

        # The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
        # We want to take the corresponding affinity for each of the initial assignments using `gt_matches`.
        # The following line grabs the max affinity for each detection to a ground truth label.
        affinities = np.take_along_axis(affinity_matrix.T, gt_matches, axis=0).squeeze(0)

        # Find the indices of the "first" detection assigned to each GT.
        unique_gt_matches, unique_dt_matches = np.unique(gt_matches, return_index=True)
        for i, thresh in enumerate(self.detection_cfg.affinity_threshs):

            # `tp_mask` may need to be defined differently with other affinities
            tp_mask = affinities[unique_dt_matches] > -thresh
            metrics[unique_dt_matches, i] = tp_mask

            # Only compute true positive error when `thresh` is equal to the tp threshold.
            is_tp_thresh = thresh == self.detection_cfg.tp_thresh
            # Ensure that there are true positives of the respective class in the frame.
            has_true_positives = np.count_nonzero(tp_mask) > 0

            if is_tp_thresh and has_true_positives:
                dt_tp_indices = unique_dt_matches[tp_mask]
                gt_tp_indices = unique_gt_matches[tp_mask]

                dt_df = pd.DataFrame([dt.__dict__ for dt in dts[dt_tp_indices]])
                gt_df = pd.DataFrame([gt.__dict__ for gt in gts[gt_tp_indices]])

                trans_error = dist_fn(dt_df, gt_df, DistFnType.TRANSLATION)
                scale_error = dist_fn(dt_df, gt_df, DistFnType.SCALE)
                orient_error = dist_fn(dt_df, gt_df, DistFnType.ORIENTATION)

                metrics[dt_tp_indices, n_threshs : n_threshs + N_TP_ERRORS] = np.vstack(
                    (trans_error, scale_error, orient_error)
                ).T
        return metrics

    def summarize(
        self, data: DefaultDict[str, np.ndarray], cls_to_ninst: DefaultDict[str, int]
    ) -> DefaultDict[str, List]:
        """Calculate and print the detection metrics.

        Args:
            data: The aggregated data used for summarization.
            cls_to_ninst: Map of classes to number of instances.

        Returns:
            summary: The summary statistics.
        """
        summary: DefaultDict[str, List] = defaultdict(list)
        recalls_interp = np.linspace(0, 1, self.detection_cfg.n_rec_samples)
        num_ths = len(self.detection_cfg.affinity_threshs)
        if not self.figures_fpath.is_dir():
            self.figures_fpath.mkdir(parents=True, exist_ok=True)

        for cls_name, cls_stats in data.items():
            ninst = cls_to_ninst[cls_name]
            ranks = cls_stats[:, -1].argsort()[::-1]
            cls_stats = cls_stats[ranks]

            for i, _ in enumerate(self.detection_cfg.affinity_threshs):
                tp = cls_stats[:, i].astype(bool)

                cumulative_tp = np.cumsum(tp, dtype=np.int)
                cumulative_fp = np.cumsum(~tp, dtype=np.int)
                cumulative_fn = ninst - cumulative_tp

                precisions = cumulative_tp / (cumulative_tp + cumulative_fp + np.finfo(float).eps)
                recalls = cumulative_tp / (cumulative_tp + cumulative_fn)

                precisions = interp(precisions)
                precisions_interp = np.interp(recalls_interp, recalls, precisions, right=0)

                ap_th = precisions_interp.mean()
                summary[cls_name] += [ap_th]

                if self.detection_cfg.save_figs:
                    self.plot(recalls_interp, precisions_interp, cls_name)

            # AP Metric
            ap = np.array(summary[cls_name][:num_ths]).mean()

            # Select only the true positives for each instance.
            tp_metrics_mask = ~np.isnan(cls_stats[:, num_ths : num_ths + N_TP_ERRORS]).all(axis=1)

            # If there are no true positives set tp errors to their maximum values due to normalization below)
            if ~tp_metrics_mask.any():
                tp_metrics = self.detection_cfg.tp_normalization_terms
            else:
                # Calculate TP metrics.
                tp_metrics = np.mean(cls_stats[:, num_ths : num_ths + N_TP_ERRORS][tp_metrics_mask], axis=0)

            # Convert errors to scores
            tp_scores = 1 - (tp_metrics / self.detection_cfg.tp_normalization_terms)

            # Compute Composite Detection Score (CDS)
            cds = ap * tp_scores.mean()

            summary[cls_name] = [ap, *tp_metrics, cds]

        logger.info(f"summary = {summary}")
        return summary

    def plot(self, rec_interp: np.ndarray, prec_interp: np.ndarray, cls_name: str) -> None:
        """Plot and save the precision recall curve.

        Args:
            rec_interp: Interpolated recall data of shape (N,).
            prec_interp: Interpolated precision data of shape (N,).
            cls_name: Class name.
        """
        plt.plot(rec_interp, prec_interp)
        plt.title("PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(f"{self.figures_fpath}/{cls_name}.png")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dt_fpath", type=str, help="Detection root folder path.", required=True)
    parser.add_argument("-g", "--gt_fpath", type=str, help="Ground truth root folder path.", required=True)
    parser.add_argument("-f", "--fig_fpath", type=str, help="Figures root folder path.", default="figs")
    args = parser.parse_args()
    logger.info(f"args == {args}")

    dt_fpath = Path(args.dt_fpath)
    gt_fpath = Path(args.gt_fpath)
    fig_fpath = Path(args.fig_fpath)

    evaluator = DetectionEvaluator(dt_fpath, gt_fpath, fig_fpath)
    metrics = evaluator.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
