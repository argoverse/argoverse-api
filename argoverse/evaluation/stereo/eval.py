# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Stereo evaluation for the Argoverse stereo leaderboard.

Evaluation:

    We consider the disparity of a pixel to be correctly estimated if the absolute disparity error is less than a
    threshold and its relative error is less than 10% of its true value, similar to the KITTI Stereo 2015 benchmark [1].
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

    Note: The `evaluate` function will use all available logical cores on the machine.

"""
import argparse
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import DefaultDict, List

import cv2
import disparity_interpolation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The disparity error image uses a custom log-color scale depicting correct estimates in blue and wrong estimates in
# red color tones, as in the KITTI Stereo 2015 Benchmark [1].
# log_colormap = [disparity error range, RGB color], where the disparity error range is defined as:
# y = 2^x for x = [-Inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, Inf].
log_colormap = [
    [np.array([0, 2 ** -4]), np.array([49, 54, 149], dtype=np.uint8)],
    [np.array([2 ** -4, 2 ** -3]), np.array([69, 117, 180], dtype=np.uint8)],
    [np.array([2 ** -3, 2 ** -2]), np.array([116, 173, 209], dtype=np.uint8)],
    [np.array([2 ** -2, 2 ** -1]), np.array([171, 217, 233], dtype=np.uint8)],
    [np.array([2 ** -1, 2 ** 0]), np.array([224, 243, 248], dtype=np.uint8)],
    [np.array([2 ** 0, 2 ** 1]), np.array([254, 224, 144], dtype=np.uint8)],
    [np.array([2 ** 1, 2 ** 2]), np.array([253, 174, 97], dtype=np.uint8)],
    [np.array([2 ** 2, 2 ** 3]), np.array([244, 109, 67], dtype=np.uint8)],
    [np.array([2 ** 3, 2 ** 4]), np.array([215, 48, 39], dtype=np.uint8)],
    [np.array([2 ** 4, np.Inf]), np.array([165, 0, 38], dtype=np.uint8)],
]


class StereoEvaluator:
    """Instantiates a StereoEvaluator object for evaluation."""

    def __init__(
        self,
        pred_root_fpath: Path,
        gt_root_fpath: Path,
        figs_fpath: Path,
        abs_error_thresholds: List = [10, 5, 3],
        rel_error_thresholds: List = [0.1, 0.1, 0.1],
        save_disparity_error_image: bool = False,
        num_procs: int = -1,
    ) -> None:
        """
        Args:
            pred_fpath_root: Path to the folder which contains the stereo predictions.
            gt_fpath_root: Path to the folder which contains the stereo ground truth.
            figs_fpath: Path to the folder which will contain the output figures.
            num_procs: Number of processes among which to subdivide work.
                Specifying -1 will use one process per available core
        """
        self.pred_root_fpath = pred_root_fpath
        self.gt_root_fpath = gt_root_fpath
        self.figs_fpath = figs_fpath
        self.abs_error_thresholds = abs_error_thresholds
        self.rel_error_thresholds = rel_error_thresholds
        self.save_disparity_error_image = save_disparity_error_image
        self.num_procs = os.cpu_count() if num_procs == -1 else num_procs

    def evaluate(self) -> pd.DataFrame:
        """Evaluate stereo output and return metrics. The multiprocessing
        library is used for parallel processing of disparities.

        Returns:
            Evaluation metrics.
        """
        pred_fpaths = list(self.pred_root_fpath.glob("**/*.png"))
        gt_fpaths = list(self.gt_root_fpath.glob("**/stereo_front_left_rect_disparity/*.png"))
        gt_obj_fpaths = list(self.gt_root_fpath.glob("**/stereo_front_left_rect_objects_disparity/*.png"))

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
                    self.abs_error_thresholds,
                    self.rel_error_thresholds,
                    self.figs_fpath,
                    save_disparity_error_image=self.save_disparity_error_image,
                )
                errors.append(error)
        else:
            args = [
                (
                    pred_fpath,
                    gt_fpath,
                    gt_obj_fpath,
                    self.abs_error_thresholds,
                    self.rel_error_thresholds,
                    self.figs_fpath,
                    self.save_disparity_error_image,
                )
                for (pred_fpath, gt_fpath, gt_obj_fpath) in zip(pred_fpaths, gt_fpaths, gt_obj_fpaths)
            ]
            with Pool(self.num_procs) as p:
                errors = p.starmap(compute_disparity_error, args)

        data = pd.concat(errors)
        data_sum = data.sum()
        summary: dict() = dict()

        for abs_error_threshold in self.abs_error_thresholds:
            d1_all = (
                data_sum[f"num_errors_bg:{str(abs_error_threshold)}"]
                + data_sum[f"num_errors_fg:{str(abs_error_threshold)}"]
            ) / (data_sum["num_pixels_bg"] + data_sum["num_pixels_fg"])
            d1_fg = data_sum[f"num_errors_fg:{str(abs_error_threshold)}"] / data_sum["num_pixels_fg"]
            d1_bg = data_sum[f"num_errors_bg:{str(abs_error_threshold)}"] / data_sum["num_pixels_bg"]
            d1_all_est = (
                data_sum[f"num_errors_bg_est:{str(abs_error_threshold)}"]
                + data_sum[f"num_errors_fg_est:{str(abs_error_threshold)}"]
            ) / (data_sum["num_pixels_bg_est"] + data_sum["num_pixels_fg_est"])
            d1_fg_est = data_sum[f"num_errors_fg_est:{str(abs_error_threshold)}"] / data_sum["num_pixels_fg_est"]
            d1_bg_est = data_sum[f"num_errors_bg_est:{str(abs_error_threshold)}"] / data_sum["num_pixels_bg_est"]

            summary[f"all:{str(abs_error_threshold)}"] = d1_all * 100
            summary[f"fg:{str(abs_error_threshold)}"] = d1_fg * 100
            summary[f"bg:{str(abs_error_threshold)}"] = d1_bg * 100
            summary[f"all*:{str(abs_error_threshold)}"] = d1_all_est * 100
            summary[f"fg*:{str(abs_error_threshold)}"] = d1_fg_est * 100
            summary[f"bg*:{str(abs_error_threshold)}"] = d1_bg_est * 100

        return summary


def compute_disparity_error(
    pred_fpath: Path,
    gt_fpath: Path,
    gt_obj_fpath: Path,
    abs_error_thresholds: List = [10, 5, 3],
    rel_error_thresholds: List = [0.1, 0.1, 0.1],
    figs_fpath: Path = None,
    save_disparity_error_image: bool = False,
):
    """Compute the disparity error metrics."""
    pred_disparity = cv2.imread(str(pred_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_disparity = cv2.imread(str(gt_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_obj_disparity = cv2.imread(str(gt_obj_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    pred_disparity = np.float32(pred_disparity) / 256.0
    gt_disparity = np.float32(gt_disparity) / 256.0
    gt_obj_disparity = np.float32(gt_obj_disparity) / 256.0

    errors = accumulator(abs_error_thresholds)

    # Compute masks
    pred_mask = pred_disparity > 0
    gt_mask = gt_disparity > 0
    obj_mask = gt_obj_disparity > 0
    bg_mask = np.logical_and(gt_mask, np.logical_not(obj_mask))
    fg_mask = np.logical_and(gt_mask, obj_mask)

    # If the density of the predicted disparity is less than 100%, then interpolate to fill up the holes.
    num_pixels_all = pred_disparity.size
    num_pixels_all_est = np.sum(pred_mask)
    density = num_pixels_all_est / max(num_pixels_all, 1.0)

    if density < 1.0:
        pred_disparity = interpolate_disparity(pred_disparity)

    # Compute errors
    abs_err = np.abs(pred_disparity - gt_disparity)
    rel_err = abs_err / np.maximum(gt_disparity, 1)

    errors["num_pixels_bg"] = np.sum(bg_mask)
    errors["num_pixels_fg"] = np.sum(fg_mask)
    errors["num_pixels_bg_est"] = np.sum(bg_mask & pred_mask)
    errors["num_pixels_fg_est"] = np.sum(fg_mask & pred_mask)

    for abs_error_threshold, rel_error_threshold in zip(abs_error_thresholds, rel_error_thresholds):
        bad_pixels = (abs_err > abs_error_threshold) & (rel_err > rel_error_threshold)

        errors[f"num_errors_bg:{str(abs_error_threshold)}"] = np.sum(bg_mask & bad_pixels)
        errors[f"num_errors_fg:{str(abs_error_threshold)}"] = np.sum(fg_mask & bad_pixels)
        errors[f"num_errors_bg_est:{str(abs_error_threshold)}"] = np.sum(bg_mask & pred_mask & bad_pixels)
        errors[f"num_errors_fg_est:{str(abs_error_threshold)}"] = np.sum(fg_mask & pred_mask & bad_pixels)

    if save_disparity_error_image:
        err = np.minimum(abs_err / abs_error_thresholds[0], rel_err / rel_error_thresholds[0])
        disparity_error_image = np.zeros((*pred_disparity.shape, 3), dtype=np.uint8)

        for threshold, color in log_colormap:
            disparity_error_image[np.logical_and(err >= threshold[0], err < threshold[1])] = color

        disparity_error_image[gt_disparity == 0] *= 0

        disparity_error_image = np.uint8(
            cv2.dilate(disparity_error_image, kernel=np.ones((2, 2), np.uint8), iterations=3)
        )

        # Get colormap image
        fig = plt.figure(figsize=(24, 2))
        ax = fig.add_subplot(111)
        x = np.linspace(1, 11, 10)
        y = np.linspace(0, 0, 10)
        sizes = np.logspace(8, 12, num=10, base=2.0)
        scalars = np.array(log_colormap, dtype=object)[:, 0] * abs_error_thresholds[0]
        scalars = [f"[{scalar[0]:.2f}, {scalar[1]:.2f}]" for scalar in scalars]
        colors = np.array(log_colormap, dtype=object)[:, 1]
        colors = [color / 255.0 for color in colors]
        plt.scatter(x, y, c=colors, s=3000)
        plt.xlabel("Disparity range errors in pixels", fontsize=24)
        plt.xticks(x, scalars, fontsize=20)
        plt.yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        plt.show()
        fig.canvas.draw()
        colormap_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        colormap_image = colormap_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        # Add colormap to the disparity error image
        c_img = colormap_image[:, :, ::-1]
        d_img = disparity_error_image[:, :, ::-1]
        x_offset = 30
        y_offset = 30
        d_img[y_offset : y_offset + c_img.shape[0], x_offset : x_offset + c_img.shape[1]] = c_img

        log_id = str(gt_fpath).split("/")[-3]
        timestamp = str(gt_fpath).split("_")[-1][:-4]
        save_dir = f"{figs_fpath}/{log_id}/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{save_dir}/disparity_error_{timestamp}.png", d_img)

    return errors


def accumulator(abs_error_thresholds: List) -> pd.DataFrame:
    """Metrics accumulator."""
    num_fields = 4 + 4 * len(abs_error_thresholds)

    columns = [
        "num_pixels_bg",
        "num_pixels_fg",
        "num_pixels_bg_est",
        "num_pixels_fg_est",
    ]

    for abs_error_threshold in abs_error_thresholds:
        columns += [
            f"num_errors_bg:{str(abs_error_threshold)}",
            f"num_errors_fg:{str(abs_error_threshold)}",
            f"num_errors_bg_est:{str(abs_error_threshold)}",
            f"num_errors_fg_est:{str(abs_error_threshold)}",
        ]

    return pd.DataFrame([[0] * num_fields], columns=columns)


def interpolate_disparity(disp: np.array) -> np.array:
    """Intepolate disparity image to inpaint holes.
    The expected run time for the Argoverse stereo image with 2056 × 2464 pixels is ~50 ms.
    """
    disp[disp == 0] = -1
    disp_interp = disparity_interpolation.disparity_interpolator(disp)

    return disp_interp


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
