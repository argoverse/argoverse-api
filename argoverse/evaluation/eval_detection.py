# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
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
    DistFnType,
    SimFnType,
    compute_affinity_matrix,
    dist_fn,
    filter_instances,
    get_ranks,
    interp,
)

matplotlib.use("Agg")  # isort:skip


import matplotlib.pyplot as plt  # isort:skip


logger = logging.getLogger(__name__)


TP_METRIC_NAMES = ["ATE", "ASE", "AOE"]
METRIC_NAMES: List[str] = ["AP"] + TP_METRIC_NAMES + ["CDS"]
METRIC_DEFAULT_VALUES: List[float] = [0, 1.0, 1.0, 1.0, 0]


@dataclass
class DetectionCfg:
    """Instantiates a DetectionCfg object for configuring a DetectionEvaluator.

    Args:
        affinity_threshs: The affinity thresholds for determining a true positive.
        affinity_fn_type: The type of affinity function to be used for calculating average precision.
        n_rec_samples: Number of recall points to sample uniformly in [0, 1]. Default to 101 recall samples.
        cds_weights: The weight vector for the detection composite score (CDS).
        tp_thresh: Center distance threshold for the true positive metrics (in meters).
        significant_digits: The precision for metrics.
        detection_classes: Detection classes for evaluation.
        save_figs: Flag to save figures.
    """

    affinity_threshs: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    affinity_fn_type: SimFnType = SimFnType.CENTER
    n_rec_samples: int = 101
    cds_weights: List[float] = field(default_factory=lambda: [3, 1, 1, 1])
    tp_thresh: float = 2.0
    significant_digits: int = 3
    detection_classes: List[str] = field(default_factory=lambda: list(OBJ_CLASS_MAPPING_DICT.keys()))
    save_figs: bool = False


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

        init_data = {k: METRIC_DEFAULT_VALUES for k in self.detection_cfg.detection_classes}
        summary = pd.DataFrame.from_dict(init_data, orient="index", columns=METRIC_NAMES)
        summary_update = pd.DataFrame.from_dict(
            self.summarize(data, cls_to_ninst), orient="index", columns=METRIC_NAMES
        )

        summary.update(summary_update)
        summary = summary.round(self.detection_cfg.significant_digits)
        summary.index = summary.index.str.title()

        summary.loc["Average Metrics"] = summary.mean().round(self.detection_cfg.significant_digits)
        return summary

    def accumulate(self, gt_fpath: Path) -> Tuple[DefaultDict[str, np.ndarray], DefaultDict[str, int]]:
        """Accumulate the statistics for each LiDAR frame.

        Args:
            gt_fpath: Ground truth file path.

        Returns:
            cls_stats: Class statistics of shape (N, 8)
            cls_to_ninst: Mapping of the class names to the number of instances in the ground
                          truth dataset.
        """
        log_id = gt_fpath.parents[1].stem
        logger.info(f"log_id = {log_id}")
        ts = gt_fpath.stem.split("_")[-1]

        dt_fpath = self.detection_fpath / f"{log_id}/per_sweep_annotations_amodal/" f"tracked_object_labels_{ts}.json"

        dts = np.array(read_label(str(dt_fpath)))
        gts = np.array(read_label(str(gt_fpath)))

        cls_stats = defaultdict(list)
        cls_to_ninst = defaultdict(int)
        for dt_cls in self.detection_cfg.detection_classes:
            dt_filtered = filter_instances(dts, dt_cls)
            gt_filtered = filter_instances(gts, dt_cls)

            logger.info(f"{dt_filtered.shape[0]} detections")
            logger.info(f"{gt_filtered.shape[0]} ground truth")
            if dt_filtered.shape[0] > 0:
                metrics, scores = self.assign(dt_filtered, gt_filtered)
                cls_stats[dt_cls] = np.hstack((metrics, scores))

            cls_to_ninst[dt_cls] = gt_filtered.shape[0]
        return cls_stats, cls_to_ninst

    def assign(self, dts: np.ndarray, gts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Attempt assignment of each detection to a ground truth label.

        Args:
            dts: Detections of shape (N,).
            gts: Ground truth labels of shape (M,).

        Returns:
            metrics: Flags indicating true/false positive and true positive errors (N, 8).
            scores: Corresponding scores for the true positives/false positives (N,)
        """
        n_threshs = len(self.detection_cfg.affinity_threshs)
        metrics = np.zeros((dts.shape[0], n_threshs + len(TP_METRIC_NAMES)))
        scores, ranks = get_ranks(dts)
        if gts.shape[0] == 0:
            return np.hstack((metrics, scores))

        affinity_matrix = compute_affinity_matrix(dts, gts, self.detection_cfg.affinity_fn_type)

        # Get the most similar GT label for each detection.
        gt_matches = np.expand_dims(affinity_matrix[ranks].argmax(axis=1), axis=0)

        # Grab the corresponding similarity score for each assignment.
        affinities = np.take_along_axis(affinity_matrix[ranks].T, gt_matches, axis=0).squeeze(0)

        # Find the indices of the "first" detection assigned to each GT.
        unique_gt_matches, unique_dt_matches = np.unique(gt_matches, return_index=True)
        for i, thr in enumerate(self.detection_cfg.affinity_threshs):

            # tp_mask may need to be defined differently with other similarity metrics
            tp_mask = affinities[unique_dt_matches] > -thr
            metrics[unique_dt_matches, i] = tp_mask

            if thr == self.detection_cfg.tp_thresh and np.count_nonzero(tp_mask) > 0:
                dt_tp_indices = unique_dt_matches[tp_mask]
                gt_tp_indices = unique_gt_matches[tp_mask]

                dt_df = pd.DataFrame([dt.__dict__ for dt in dts[ranks][dt_tp_indices]])
                gt_df = pd.DataFrame([gt.__dict__ for gt in gts[gt_tp_indices]])

                trans_error = dist_fn(dt_df, gt_df, DistFnType.TRANSLATION)
                scale_error = dist_fn(dt_df, gt_df, DistFnType.SCALE)
                orient_error = dist_fn(dt_df, gt_df, DistFnType.ORIENTATION)

                metrics[dt_tp_indices, n_threshs : n_threshs + 3] = np.vstack(
                    (trans_error, scale_error, orient_error)
                ).T

        return metrics, scores

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

            # TP Error Metrics
            tp_metrics = np.mean(cls_stats[:, num_ths : num_ths + 3], axis=0)
            tp_metrics[2] /= np.pi / 2  # normalize orientation

            # clip so that we don't get negative values in (1 - ATE)
            cds_summands = np.hstack((ap, np.clip(1 - tp_metrics, a_min=0, a_max=None)))

            # Ranking metric
            cds = np.average(cds_summands, weights=self.detection_cfg.cds_weights)

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
    parser.add_argument("-f", "--fig_fpath", type=str, help="Figures root folder path.", default="figs/")
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
