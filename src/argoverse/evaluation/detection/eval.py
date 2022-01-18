# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
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
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from argoverse.evaluation.detection.constants import SIGNIFICANT_DIGITS, STATISTIC_NAMES, TP_ERROR_NAMES
from argoverse.evaluation.detection.utils import DetectionCfg, accumulate, calc_ap, plot

logger = logging.getLogger(__name__)


def evaluate(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    poses: Optional[pd.DataFrame],
    cfg: DetectionCfg,
) -> Dict[str, pd.DataFrame]:
    """Evaluate detection output and return metrics. The multiprocessing
    library is used for parallel processing of sweeps -- each sweep is
    processed independently, computing assignment between detections and
    ground truth annotations.

    Returns:
        Evaluation metrics of shape (C + 1, K) where C + 1 is the number of classes.
        plus a row for their means. K refers to the number of evaluation metrics.
    """

    jobs = [(dts.loc[k], v, poses, cfg) for k, v in gts.groupby(["log_id", "tov_ns"])]

    ncpus = mp.cpu_count()
    chunksize = max(1, len(jobs) // ncpus)
    # outputs = process_map(accumulate, jobs, max_workers=ncpus, chunksize=chunksize)
    # with mp.Pool(ncpus) as p:
    #     outputs = p.starmap(accumulate, jobs, chunksize=chunksize)
    outputs = [accumulate(*job) for job in jobs]

    dts = pd.concat([o["dts"] for o in outputs]).reset_index(drop=True)
    gts = pd.concat([o["gts"] for o in outputs]).reset_index(drop=True)

    summary = summarize(dts, gts, cfg).round(SIGNIFICANT_DIGITS)
    summary = summary.set_index(summary.index.str.title())
    summary.loc["Average Metrics"] = summary.mean().round(SIGNIFICANT_DIGITS)
    return {"dts": dts, "gts": gts, "summary": summary}


def summarize(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
) -> pd.DataFrame:
    """Calculate and print the detection metrics.

    Args:
        data: The aggregated data used for summarization.
        cls_to_ninst: Map of classes to number of instances.

    Returns:
        summary: The summary statistics.
    """

    data = {stat: cfg.summary_default_vals[i] for i, stat in enumerate(STATISTIC_NAMES)}
    summary = pd.DataFrame(data, index=cfg.dt_classes)
    recalls_interp = np.linspace(0, 1, cfg.n_rec_samples)

    figs_rootdir = Path("figs")
    if not Path(figs_rootdir).is_dir():
        Path(figs_rootdir).mkdir(parents=True, exist_ok=True)

    data = {threshold: 0.0 for threshold in cfg.affinity_threshs}
    average_precisions = pd.DataFrame(data=data, index=cfg.dt_classes)
    for cls_name in cfg.dt_classes:
        is_valid = np.logical_and(dts["category"] == cls_name, dts["is_evaluated"])
        cls_stats = dts[is_valid].reset_index(drop=True)

        cls_stats = cls_stats.sort_values(by="score", ascending=False).reset_index(drop=True)
        ninst = gts.loc[gts["category"] == cls_name, "is_evaluated"].sum()

        if ninst == 0:
            continue

        for _, thresh in enumerate(cfg.affinity_threshs):
            tps = cls_stats[thresh]
            ap_th, precisions_interp = calc_ap(tps, recalls_interp, ninst)

            average_precisions.loc[cls_name, thresh] = ap_th

            if cfg.save_figs:
                plot(
                    recalls_interp,
                    precisions_interp,
                    f"{cls_name}_{thresh}",
                    figs_rootdir,
                )

        # AP Metric.
        ap = average_precisions.loc[cls_name].mean()

        # Select only the true positives for each instance.
        middle_threshold = cfg.affinity_threshs[len(cfg.affinity_threshs) // 2]
        tp_metrics_mask = cls_stats[middle_threshold]

        # If there are no true positives set tps errors to their maximum values due to normalization below).
        if ~tp_metrics_mask.any():
            tp_metrics = cfg.tp_normalization_terms
        else:
            # Calculate TP metrics.
            tp_metrics = cls_stats.loc[tp_metrics_mask, TP_ERROR_NAMES].mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_metrics, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = ap * tp_scores.mean()
        summary.loc[cls_name] = [ap, *tp_metrics, cds]
    return summary
