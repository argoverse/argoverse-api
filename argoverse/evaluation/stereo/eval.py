# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Stereo evaluation for the Argoverse stereo leaderboard.

Evaluation:

    We consider the disparity of a pixel to be correctly estimated if the absolute disparity error is less than a
    threshold OR its relative error is less than 10% of its true value, similar to the KITTI Stereo 2015 benchmark [1].
    We define three disparity error thresholds: 3, 5, and 10 pixels.

    Some stereo matching methods such as Semi-Global Matching (SGM) might provide sparse disparity maps, meaning that
    some pixels will not have valid disparity values. In those cases, we interpolate the predicted disparity map using
    a simple nearest neighbor interpolation as in the KITTI Stereo 2015 benchmark [1] to assure we can compare it to
    our semi-dense ground-truth disparity map. Current deep stereo matching methods normally predict disparity maps
    with 100% density. Thus, an interpolation step is not needed for the evaluation.

    The disparity errors metrics are the following:

    1. all: Percentage of stereo disparity errors averaged over all ground truth pixels in the reference frame
            (left stereo image).
    2. bg: Percentage of stereo disparity errors averaged only over background regions.
    3. fg: Percentage of stereo disparity errors averaged only over foreground regions.
           The foreground region boundary (or mask) is given by the ground-truth disparity map containing only
           foreground objects, extracted using ground-truth cuboid tracks.

    The * (asterisk) means that the evaluation is performed using only the algorithm predicted disparities.
    Even though the disparities might be sparse, they are not interpolated.

    [1] http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo.

Results:

    We evaluate all metrics using three error thresholds: 3, 5, or 10 pixels.
    The summary results are represented as a table containing the following fields:

        all:10,
        fg:10,
        bg:10,
        all*:10,
        fg*:10,
        bg*:10,
        all:5,
        fg:5,
        bg:5,
        all*:5,
        fg*:5,
        bg*:5,
        all:3,
        fg:3,
        bg:3,
        all*:3,
        fg*:3,
        bg*:3.

    If a model analysis report (i.e., model_analysis_report.txt) is available in the folder which contains the
        stereo predictions, the following information will be added to the summary results:

        #parameters (M),
        #flops (T),
        #activations (G),
        Inference time (ms),
        Device name.

    Note: The `evaluate` function will use all available logical cores on the machine.

"""
import argparse
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from argoverse.evaluation.stereo.constants import DEFAULT_ABS_ERROR_THRESHOLDS, DEFAULT_REL_ERROR_THRESHOLDS
from argoverse.evaluation.stereo.utils import compute_disparity_error

logger = logging.getLogger(__name__)


class StereoEvaluator:
    """Instantiates a StereoEvaluator object for evaluation."""

    def __init__(
        self,
        pred_root_fpath: Path,
        gt_root_fpath: Path,
        figs_fpath: Path,
        abs_error_thresholds: List[int] = DEFAULT_ABS_ERROR_THRESHOLDS,
        rel_error_thresholds: List[float] = DEFAULT_REL_ERROR_THRESHOLDS,
        save_disparity_error_image: bool = False,
        num_procs: int = -1,
    ) -> None:
        """
        Args:
            pred_fpath_root: Path to the folder which contains the stereo predictions.
            gt_fpath_root: Path to the folder which contains the stereo ground truth.
            figs_fpath: Path to the folder which will contain the output figures.
            abs_error_thresholds: Absolute disparity error thresholds, in pixels.
            rel_error_thresholds: Relative disparity error thresholds, in pixels.
            save_disparity_error_image: Saves the disparity image error using the PNG format in the figs_fpath.
            num_procs: Number of processes among which to subdivide work.
                Specifying -1 will use one process per available core.
        """
        self.pred_root_fpath = pred_root_fpath
        self.gt_root_fpath = gt_root_fpath
        self.figs_fpath = figs_fpath
        self.abs_error_thresholds = abs_error_thresholds
        self.rel_error_thresholds = rel_error_thresholds
        self.save_disparity_error_image = save_disparity_error_image
        self.num_procs = os.cpu_count() if num_procs == -1 else num_procs

    def evaluate(self) -> pd.DataFrame:
        """Evaluate stereo output and return metrics. The multiprocessing library is used for parallel processing
        of disparity evaluation.

        Returns:
            Evaluation metrics.
        """
        pred_fpaths = list(self.pred_root_fpath.glob("**/*.png"))
        gt_fpaths = list(self.gt_root_fpath.glob("**/stereo_front_left_rect_disparity/*.png"))
        gt_obj_fpaths = list(self.gt_root_fpath.glob("**/stereo_front_left_rect_objects_disparity/*.png"))
        report_fpath = self.pred_root_fpath / "model_analysis_report.txt"

        if len(pred_fpaths) == 1:
            timestamp = str(pred_fpaths[0]).split("_")[-1][:-4]
            gt_fpaths = [gt_fpath for gt_fpath in gt_fpaths if timestamp in str(gt_fpath)]
            gt_obj_fpaths = [gt_obj_fpath for gt_obj_fpath in gt_obj_fpaths if timestamp in str(gt_obj_fpath)]

        assert len(pred_fpaths) == len(gt_fpaths)

        if self.num_procs == 1:
            errors = []
            for (pred_fpath, gt_fpath, gt_obj_fpath) in zip(pred_fpaths, gt_fpaths, gt_obj_fpaths):
                error = compute_disparity_error(
                    pred_fpath,
                    gt_fpath,
                    gt_obj_fpath,
                    self.figs_fpath,
                    self.abs_error_thresholds,
                    self.rel_error_thresholds,
                    save_disparity_error_image=self.save_disparity_error_image,
                )
                errors.append(error)
        else:
            args = [
                (
                    pred_fpath,
                    gt_fpath,
                    gt_obj_fpath,
                    self.figs_fpath,
                    self.abs_error_thresholds,
                    self.rel_error_thresholds,
                    self.save_disparity_error_image,
                )
                for (pred_fpath, gt_fpath, gt_obj_fpath) in zip(pred_fpaths, gt_fpaths, gt_obj_fpaths)
            ]
            with Pool(self.num_procs) as p:
                errors = p.starmap(compute_disparity_error, args)

        data = pd.concat(errors)
        # Sums over all frames (row dimension) for each disparity error metric
        data_sum = data.sum(axis=0)
        summary: Dict[str, Union[float, str]] = {}

        for abs_error_thresh in self.abs_error_thresholds:
            all = (data_sum[f"num_errors_bg:{abs_error_thresh}"] + data_sum[f"num_errors_fg:{abs_error_thresh}"]) / (
                data_sum["num_pixels_bg"] + data_sum["num_pixels_fg"]
            )
            fg = data_sum[f"num_errors_fg:{abs_error_thresh}"] / data_sum["num_pixels_fg"]
            bg = data_sum[f"num_errors_bg:{abs_error_thresh}"] / data_sum["num_pixels_bg"]
            all_est = (
                data_sum[f"num_errors_bg_est:{abs_error_thresh}"] + data_sum[f"num_errors_fg_est:{abs_error_thresh}"]
            ) / (data_sum["num_pixels_bg_est"] + data_sum["num_pixels_fg_est"])
            fg_est = data_sum[f"num_errors_fg_est:{abs_error_thresh}"] / data_sum["num_pixels_fg_est"]
            bg_est = data_sum[f"num_errors_bg_est:{abs_error_thresh}"] / data_sum["num_pixels_bg_est"]

            summary[f"all:{abs_error_thresh}"] = all * 100
            summary[f"fg:{abs_error_thresh}"] = fg * 100
            summary[f"bg:{abs_error_thresh}"] = bg * 100
            summary[f"all*:{abs_error_thresh}"] = all_est * 100
            summary[f"fg*:{abs_error_thresh}"] = fg_est * 100
            summary[f"bg*:{abs_error_thresh}"] = bg_est * 100

        if report_fpath.is_file():
            # Collect performance metrics from report
            with open(report_fpath, "r") as f:
                result_lines = f.readlines()

            device_name = result_lines[0].split(": ")[-1].strip()
            parameters = int(result_lines[1].split(": ")[-1].strip())
            flops = int(result_lines[2].split(": ")[-1].strip())
            activations = int(result_lines[3].split(": ")[-1].strip())
            inference_time_ms = float(result_lines[4].split(": ")[-1].strip())

            summary["#parameters (M)"] = parameters / 1e6
            summary["#flops (T)"] = flops / 1e12
            summary["#activations (G)"] = activations / 1e9
            summary["Inference time (ms)"] = inference_time_ms
            summary["Device name"] = device_name

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--pred_fpath", type=str, help="Stereo root folder path.", required=True)
    parser.add_argument(
        "-g",
        "--gt_fpath",
        type=str,
        help="Ground truth root folder path.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--num_processes",
        type=int,
        help="Number of processes among which to subdivide work. Specifying -1 will use one process per available core",
        default=-1,
    )

    parser.add_argument("-f", "--fig_fpath", type=str, help="Figures root folder path.", default="figs")
    args = parser.parse_args()
    logger.info(f"args == {args}")

    pred_fpath = Path(args.pred_fpath)
    gt_fpath = Path(args.gt_fpath)
    fig_fpath = Path(args.fig_fpath)

    evaluator = StereoEvaluator(pred_fpath, gt_fpath, fig_fpath, num_procs=args.num_processes)
    metrics = evaluator.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
