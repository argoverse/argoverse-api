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
import logging
import multiprocessing as mp
from collections import defaultdict
from typing import DefaultDict, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from argoverse.evaluation.detection.constants import (N_TP_ERRORS,
                                                      SIGNIFICANT_DIGITS,
                                                      STATISTIC_NAMES)
from argoverse.evaluation.detection.utils import (DetectionCfg, accumulate,
                                                  calc_ap)

logger = logging.getLogger(__name__)


def evaluate(
    dts: DataFrame, gts: DataFrame, poses: DataFrame, cfg: DetectionCfg
) -> DataFrame:
    """Evaluate detection output and return metrics. The multiprocessing
    library is used for parallel processing of sweeps -- each sweep is
    processed independently, computing assignment between detections and
    ground truth annotations.

    Returns:
        Evaluation metrics of shape (C + 1, K) where C + 1 is the number of classes.
        plus a row for their means. K refers to the number of evaluation metrics.
    """

    stats = []
    dts["uuid"] = dts["log_id"] + "_" + dts["tov_ns"].astype(str)
    gts["uuid"] = gts["log_id"] + "_" + gts["tov_ns"].astype(str)

    cls_to_ninst_list: List[Dict[str, int]] = []

    jobs = []
    for uuid, log_dts in tqdm(dts.groupby("uuid")):

        # Grab corresponding ground truth data.
        log_gts = gts[gts.uuid == uuid]

        if log_gts.shape[0] == 0:
            continue
        jobs.append((log_dts, log_gts, poses, cfg))

    ncpus = mp.cpu_count()
    chunksize = max(len(jobs) // ncpus, 1)
    outputs = process_map(accumulate, jobs, max_workers=ncpus, chunksize=chunksize)
    for output in outputs:
        accumulation, scene_cls_to_ninst = output
        cls_to_ninst_list.append(scene_cls_to_ninst)
        if accumulation.shape[0] > 0:
            stats.append(accumulation)

    cls_to_ninst = defaultdict(int)
    for item in cls_to_ninst_list:
        for k, v in item.items():
            cls_to_ninst[k] += v

    stats = pd.concat(stats).reset_index(drop=True)
    init_data = {dt_cls: cfg.summary_default_vals for dt_cls in cfg.dt_classes}
    summary = DataFrame.from_dict(init_data, orient="index", columns=STATISTIC_NAMES)
    if len(stats) == 0:
        logger.warning("No matches ...")
        return summary
    summary_update = DataFrame.from_dict(
        summarize(stats, cfg, cls_to_ninst), orient="index", columns=STATISTIC_NAMES
    )

    summary.update(summary_update)
    summary = summary.round(SIGNIFICANT_DIGITS)
    summary = summary.set_index(summary.index.str.title())
    summary.loc["Average Metrics"] = summary.mean().round(SIGNIFICANT_DIGITS)
    return summary


def summarize(
    data: DataFrame,
    cfg: DetectionCfg,
    cls_to_ninst: DefaultDict[str, int],
) -> DefaultDict[str, List[float]]:
    """Calculate and print the detection metrics.

    Args:
        data: The aggregated data used for summarization.
        cls_to_ninst: Map of classes to number of instances.

    Returns:
        summary: The summary statistics.
    """

    summary: DefaultDict[str, List[float]] = defaultdict(list)
    recalls_interp = np.linspace(0, 1, cfg.n_rec_samples)
    num_ths = len(cfg.affinity_threshs)

    # if not Path(figs_rootdir).is_dir():
    #     Path(figs_rootdir).mkdir(parents=True, exist_ok=True)

    for cls_name, cls_stats in data.groupby("label_class"):
        cls_stats = cls_stats.sort_values(by="score", ascending=False).reset_index(
            drop=True
        )
        ninst = cls_to_ninst[cls_name]
        for _, thresh in enumerate(cfg.affinity_threshs):
            tps = cls_stats.loc[:, str(thresh)].reset_index(drop=True)
            ap_th, precisions_interp = calc_ap(tps, recalls_interp, ninst)
            summary[cls_name] += [ap_th]

            # if cfg.save_figs:
            #     plot(recalls_interp, precisions_interp, cls_name, figs_rootdir)

        # AP Metric.
        ap = np.array(summary[cls_name][:num_ths]).mean()

        # Select only the true positives for each instance.
        tp_metrics_mask = ~np.isnan(
            cls_stats.iloc[:, num_ths : num_ths + N_TP_ERRORS]
        ).all(axis=1)

        # If there are no true positives set tps errors to their maximum values due to normalization below).
        if ~tp_metrics_mask.any():
            tp_metrics = cfg.tp_normalization_terms
        else:
            # Calculate TP metrics.
            tp_metrics = cls_stats.iloc[:, num_ths : num_ths + N_TP_ERRORS][
                tp_metrics_mask
            ].mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_metrics, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = ap * tp_scores.mean()

        summary[cls_name] = [ap, *tp_metrics, cds]
    return summary
