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

            - Average Translation Measure (ATM): ATE / TP_THRESHOLD; 0 <= 1 - ATE / TP_THRESHOLD <= 1.
            - Average Scaling Measure (ASM): 1 - ASE / 1;  0 <= 1 - ASE / 1 <= 1.
            - Average Orientation Measure (AOM): 1 - AOE / PI; 0 <= 1 - AOE / PI <= 1.

            These (as well as AP) are averaged over each detection class to produce:

            - mAP
            - mATM
            - mASM
            - mAOM


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
from multiprocessing import Pool
from pathlib import Path
from typing import DefaultDict, List, NamedTuple

import numpy as np
import pandas as pd

from argoverse.evaluation.detection.constants import N_TP_ERRORS, SIGNIFICANT_DIGITS, STATISTIC_NAMES
from argoverse.evaluation.detection.utils import DetectionCfg, accumulate, calc_ap, plot

logger = logging.getLogger(__name__)


class DetectionEvaluator(NamedTuple):
    """Instantiates a DetectionEvaluator object for evaluation.

    Args:
        dt_fpath_root: Path to the folder which contains the detections.
        gt_fpath_root: Path to the folder which contains the split of logs.
        figs_fpath: Path to the folder which will contain the output figures.
        cfg: Detection configuration settings.
    """

    dt_root_fpath: Path
    gt_root_fpath: Path
    figs_fpath: Path
    cfg: DetectionCfg = DetectionCfg()

    def evaluate(self) -> pd.DataFrame:
        """Evaluate detection output and return metrics. The multiprocessing
        library is used for parallel assignment between detections and ground truth
        annotations.

        Returns:
            Evaluation metrics of shape (C + 1, K) where C + 1 is the number of classes.
            plus a row for their means. K refers to the number of evaluation metrics.
        """
        dt_fpaths = list(self.dt_root_fpath.glob("*/per_sweep_annotations_amodal/*.json"))
        gt_fpaths = list(self.gt_root_fpath.glob("*/per_sweep_annotations_amodal/*.json"))

        assert len(dt_fpaths) == len(gt_fpaths)
        data: DefaultDict[str, np.ndarray] = defaultdict(list)
        cls_to_ninst: DefaultDict[str, int] = defaultdict(int)

        args = [(self.dt_root_fpath, gt_fpath, self.cfg) for gt_fpath in gt_fpaths]
        with Pool(os.cpu_count()) as p:
            accum = p.starmap(accumulate, args)

        for frame_stats, frame_cls_to_inst in accum:
            for cls_name, cls_stats in frame_stats.items():
                data[cls_name].append(cls_stats)
            for cls_name, num_inst in frame_cls_to_inst.items():
                cls_to_ninst[cls_name] += num_inst

        data = defaultdict(np.ndarray, {k: np.vstack(v) for k, v in data.items()})

        init_data = {dt_cls: self.cfg.summary_default_vals for dt_cls in self.cfg.dt_classes}
        summary = pd.DataFrame.from_dict(init_data, orient="index", columns=STATISTIC_NAMES)
        summary_update = pd.DataFrame.from_dict(
            self.summarize(data, cls_to_ninst), orient="index", columns=STATISTIC_NAMES
        )

        summary.update(summary_update)
        summary = summary.round(SIGNIFICANT_DIGITS)
        summary.index = summary.index.str.title()

        summary.loc["Average Metrics"] = summary.mean().round(SIGNIFICANT_DIGITS)
        return summary

    def summarize(
        self, data: DefaultDict[str, np.ndarray], cls_to_ninst: DefaultDict[str, int]
    ) -> DefaultDict[str, List[float]]:
        """Calculate and print the detection metrics.

        Args:
            data: The aggregated data used for summarization.
            cls_to_ninst: Map of classes to number of instances.

        Returns:
            summary: The summary statistics.
        """
        summary: DefaultDict[str, List[float]] = defaultdict(list)
        recalls_interp = np.linspace(0, 1, self.cfg.n_rec_samples)
        num_ths = len(self.cfg.affinity_threshs)
        if not self.figs_fpath.is_dir():
            self.figs_fpath.mkdir(parents=True, exist_ok=True)

        for cls_name, cls_stats in data.items():
            ninst = cls_to_ninst[cls_name]
            ranks = cls_stats[:, -1].argsort()[::-1]  # sort by last column, i.e. confidences
            cls_stats = cls_stats[ranks]

            for i, _ in enumerate(self.cfg.affinity_threshs):
                tp = cls_stats[:, i].astype(bool)
                ap_th, precisions_interp = calc_ap(tp, recalls_interp, ninst)
                summary[cls_name] += [ap_th]

                if self.cfg.save_figs:
                    plot(recalls_interp, precisions_interp, cls_name, self.figs_fpath)

            # AP Metric.
            ap = np.array(summary[cls_name][:num_ths]).mean()

            # Select only the true positives for each instance.
            tp_metrics_mask = ~np.isnan(cls_stats[:, num_ths : num_ths + N_TP_ERRORS]).all(axis=1)

            # If there are no true positives set tp errors to their maximum values due to normalization below).
            if ~tp_metrics_mask.any():
                tp_metrics = self.cfg.tp_normalization_terms
            else:
                # Calculate TP metrics.
                tp_metrics = np.mean(
                    cls_stats[:, num_ths : num_ths + N_TP_ERRORS][tp_metrics_mask],
                    axis=0,
                )

            # Convert errors to scores.
            tp_scores = 1 - (tp_metrics / self.cfg.tp_normalization_terms)

            # Compute Composite Detection Score (CDS).
            cds = ap * tp_scores.mean()

            summary[cls_name] = [ap, *tp_metrics, cds]

        logger.info(f"summary = {summary}")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dt_fpath", type=str, help="Detection root folder path.", required=True)
    parser.add_argument(
        "-g",
        "--gt_fpath",
        type=str,
        help="Ground truth root folder path.",
        required=True,
    )
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
