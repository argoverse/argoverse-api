# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Stereo utilities to support the Argoverse stereo evaluation."""

from pathlib import Path
from typing import List

import cv2
import disparity_interpolation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argoverse.evaluation.stereo.constants import (
    DEFAULT_ABS_ERROR_THRESHOLDS,
    DEFAULT_REL_ERROR_THRESHOLDS,
    LOG_COLORMAP,
)


def compute_disparity_error(
    pred_fpath: Path,
    gt_fpath: Path,
    gt_obj_fpath: Path,
    abs_error_thresholds: List = DEFAULT_ABS_ERROR_THRESHOLDS,
    rel_error_thresholds: List = DEFAULT_REL_ERROR_THRESHOLDS,
    figs_fpath: Path = None,
    save_disparity_error_image: bool = False,
) -> pd.DataFrame:
    """Compute the disparity error metrics."""
    pred_disparity = cv2.imread(str(pred_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_disparity = cv2.imread(str(gt_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_obj_disparity = cv2.imread(str(gt_obj_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    pred_disparity = np.float32(pred_disparity) / 256.0
    gt_disparity = np.float32(gt_disparity) / 256.0
    gt_obj_disparity = np.float32(gt_obj_disparity) / 256.0

    errors = accumulate_stereo_metrics(abs_error_thresholds)

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

    for abs_error_thresh, rel_error_thresh in zip(abs_error_thresholds, rel_error_thresholds):
        bad_pixels = (abs_err > abs_error_thresh) & (rel_err > rel_error_thresh)

        errors[f"num_errors_bg:{abs_error_thresh}"] = np.sum(bg_mask & bad_pixels)
        errors[f"num_errors_fg:{abs_error_thresh}"] = np.sum(fg_mask & bad_pixels)
        errors[f"num_errors_bg_est:{abs_error_thresh}"] = np.sum(bg_mask & pred_mask & bad_pixels)
        errors[f"num_errors_fg_est:{abs_error_thresh}"] = np.sum(fg_mask & pred_mask & bad_pixels)

    if save_disparity_error_image:
        err = np.minimum(abs_err / abs_error_thresholds[0], rel_err / rel_error_thresholds[0])
        disparity_error_image = np.zeros((*pred_disparity.shape, 3), dtype=np.uint8)

        for threshold, color in LOG_COLORMAP:
            disparity_error_image[np.logical_and(err >= threshold[0], err < threshold[1])] = color

        disparity_error_image[gt_disparity == 0] *= 0

        disparity_error_image = np.uint8(
            cv2.dilate(disparity_error_image, kernel=np.ones((2, 2), np.uint8), iterations=3)
        )

        # Plot the custom log-colormap and blend it with the disparity error image.
        fig = plt.figure(figsize=(24, 2))
        ax = fig.add_subplot(111)
        x = np.linspace(1, 11, 10)
        y = np.linspace(0, 0, 10)
        scalars = np.array(LOG_COLORMAP, dtype=object)[:, 0] * abs_error_thresholds[0]
        scalars = [f"[{scalar[0]:.2f}, {scalar[1]:.2f}]" for scalar in scalars]
        colors = np.array(LOG_COLORMAP, dtype=object)[:, 1]
        colors = [color / 255.0 for color in colors]

        # Plot the custom log-colormap as colored circles with the disparity error ranges as labels.
        plt.scatter(x, y, c=colors, s=3000)
        plt.xlabel("Disparity range errors in pixels", fontsize=24)
        plt.xticks(x, scalars, fontsize=20)
        plt.yticks([])

        # Remove the plot frame to make the log-colormap more clear.
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()

        # Recover the log-colormap drawing as a np.array image.
        fig.canvas.draw()
        colormap_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        colormap_image = colormap_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        # Blend the log-colormap image to the top of the disparity error image.
        c_img = colormap_image[:, :, ::-1]
        d_img = disparity_error_image[:, :, ::-1]
        x_offset = 30
        y_offset = 30
        d_img[y_offset : y_offset + c_img.shape[0], x_offset : x_offset + c_img.shape[1]] = c_img

        # Save the blended disparity error image to a PNG file.
        log_id = str(gt_fpath).split("/")[-3]
        timestamp = str(gt_fpath).split("_")[-1][:-4]
        save_dir = f"{figs_fpath}/{log_id}/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{save_dir}/disparity_error_{timestamp}.png", d_img)

    return errors


def accumulate_stereo_metrics(abs_error_thresholds: List) -> pd.DataFrame:
    """Metrics accumulator."""
    num_fields = 4 + 4 * len(abs_error_thresholds)

    columns = [
        "num_pixels_bg",
        "num_pixels_fg",
        "num_pixels_bg_est",
        "num_pixels_fg_est",
    ]

    for abs_error_thresh in abs_error_thresholds:
        columns += [
            f"num_errors_bg:{abs_error_thresh}",
            f"num_errors_fg:{abs_error_thresh}",
            f"num_errors_bg_est:{abs_error_thresh}",
            f"num_errors_fg_est:{abs_error_thresh}",
        ]

    return pd.DataFrame([[0] * num_fields], columns=columns)


def interpolate_disparity(disp: np.ndarray) -> np.ndarray:
    """Intepolate disparity image to inpaint holes.
    The expected run time for the Argoverse stereo image with 2056 × 2464 pixels is ~50 ms.
    """
    disp[disp == 0] = -1
    disp_interp = disparity_interpolation.disparity_interpolator(disp)

    return disp_interp
