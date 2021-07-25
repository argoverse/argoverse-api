# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Stereo utilities to support the Argoverse stereo evaluation."""

from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

from argoverse.evaluation.stereo.constants import (
    DEFAULT_ABS_ERROR_THRESHOLDS,
    DEFAULT_REL_ERROR_THRESHOLDS,
    DISPARITY_NORMALIZATION,
    LOG_COLORMAP,
    NUM_EVAL_ERROR_REGIONS,
    NUM_EVAL_PIXEL_REGIONS,
)


def compute_disparity_error(
    pred_fpath: Path,
    gt_fpath: Path,
    gt_obj_fpath: Path,
    figs_fpath: Path,
    abs_error_thresholds: List[int] = DEFAULT_ABS_ERROR_THRESHOLDS,
    rel_error_thresholds: List[float] = DEFAULT_REL_ERROR_THRESHOLDS,
    save_disparity_error_image: bool = False,
) -> pd.DataFrame:
    """Compute the disparity errors.

    Args:
        pred_fpath: Path to predicted disparity map.
        gt_fpath: Path to ground-truth disparity map.
        gt_obj_fpath: Path to ground-truth disparity map for foreground objects.
        figs_fpath: Path to the folder which will contain the output figures.
        abs_error_thresholds: Absolute disparity error thresholds, in pixels.
        rel_error_thresholds: Relative disparity error thresholds, in pixels.
        save_disparity_error_image: Saves the disparity image error using the PNG format in the figs_fpath.

    Returns:
        errors: Pandas DataFrame containing disparity error information.

    """
    pred_disparity = cv2.imread(str(pred_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_disparity = cv2.imread(str(gt_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_obj_disparity = cv2.imread(str(gt_obj_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    pred_disparity = np.float32(pred_disparity) / DISPARITY_NORMALIZATION
    gt_disparity = np.float32(gt_disparity) / DISPARITY_NORMALIZATION
    gt_obj_disparity = np.float32(gt_obj_disparity) / DISPARITY_NORMALIZATION

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
    density = num_pixels_all_est / num_pixels_all

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
        timestamp = int(Path(gt_fpath).stem.split("_")[-1])
        log_id = Path(gt_fpath).parts[-3]
        write_dir_path = Path(f"{figs_fpath}/{log_id}")

        write_disparity_error_image(
            pred_disparity,
            gt_disparity,
            timestamp,
            write_dir_path,
            abs_error_thresholds,
            rel_error_thresholds,
        )

    return errors


def accumulate_stereo_metrics(abs_error_thresholds: List[int]) -> pd.DataFrame:
    """Stereo metrics accumulator.

    Initializes the stereo metrics accumulator with zeroes in all fields.
    """
    num_fields = NUM_EVAL_PIXEL_REGIONS + NUM_EVAL_ERROR_REGIONS * len(abs_error_thresholds)

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


@njit(nogil=True)  # type: ignore
def interpolate_disparity(disparity: np.ndarray) -> np.ndarray:
    """Interpolate disparity image to inpaint holes.

    The predicted disparity map might contain holes which need to be interpolated for a dense result.
    Interpolation is performed by simply propagating valid disparities through neighboring invalid disparity areas.

    Please refer to the paper "Stereo Processing by Semi-Global Matching and Mutual Information" available at
    https://core.ac.uk/download/pdf/11134866.pdf for more information about the algorithm to interpolate disparity maps.

    The expected run time for the Argoverse stereo image with 2056 × 2464 pixels is ~50 ms.

    Args:
        disparity: Array of shape (M, N) representing a float32 single-channel disparity map.

    Returns:
        disparity: Array of shape (M, N) representing a float32 single-channel interpolated disparity map.
    """
    height, width = disparity.shape

    disparity = np.where(disparity == 0, -1, disparity)  # Set invalid disparities to -1

    for v in range(height):
        count = 0
        for u in range(width):
            # If disparity is valid
            if disparity[v, u] >= 0:
                #  At least one pixel requires interpolation
                if count >= 1:
                    # First and last value for interpolation
                    u1 = u - count
                    u2 = u - 1
                    # Set pixel to min disparity
                    if u1 > 0 and u2 < width - 1:
                        d_ipol = min(disparity[v, u1 - 1], disparity[v, u2 + 1])
                        for u_curr in range(u1, u2 + 1):
                            disparity[v, u_curr] = d_ipol
                count = 0
            else:
                count += 1

        # Extrapolate to the left
        for u in range(width):
            if disparity[v, u] >= 0:
                for u2 in range(u):
                    disparity[v, u2] = disparity[v, u]
                break

        # Extrapolate to the right
        for u in range(width - 1, -1, -1):
            if disparity[v, u] >= 0:
                for u2 in range(u + 1, width):
                    disparity[v, u2] = disparity[v, u]
                break

    for u in range(0, width):
        # Extrapolate to the top
        for v in range(0, height):
            if disparity[v, u] >= 0:
                for v2 in range(0, v):
                    disparity[v2, u] = disparity[v, u]
                break

        # Extrapolate to the bottom
        for v in range(height - 1, -1, -1):
            if disparity[v, u] >= 0:
                for v2 in range(v + 1, height):
                    disparity[v2, u] = disparity[v, u]
                break

    return disparity


def write_disparity_error_image(
    pred_disparity: np.ndarray,
    gt_disparity: np.ndarray,
    timestamp: int,
    write_dir_path: Path,
    abs_error_thresholds: List[int] = DEFAULT_ABS_ERROR_THRESHOLDS,
    rel_error_thresholds: List[float] = DEFAULT_REL_ERROR_THRESHOLDS,
) -> None:
    """Write the disparity error image to disk in the PNG format.

    Args:
        pred_disparity: Predicted disparity map.
        gt_disparity: Ground-truth disparity map.
        timestamp: timestamp of the disparity image.
        write_dir_path: Path to the directory for writing the disparity error image.
        abs_error_thresholds: Absolute disparity error thresholds, in pixels.
        rel_error_thresholds: Relative disparity error thresholds, in pixels.
    """
    disparity_error_image = compute_disparity_error_image(
        pred_disparity,
        gt_disparity,
        abs_error_thresholds=abs_error_thresholds,
        rel_error_thresholds=rel_error_thresholds,
    )

    # Save the disparity error image to a PNG file.
    Path(write_dir_path).mkdir(parents=True, exist_ok=True)
    # Write the disparity error image. First, switch RGB -> BGR for writing with OpenCV.
    cv2.imwrite(f"{write_dir_path}/disparity_error_{timestamp}.png", disparity_error_image[:, :, ::-1])


def compute_disparity_error_image(
    pred_disparity: np.ndarray,
    gt_disparity: np.ndarray,
    abs_error_thresholds: List[int] = DEFAULT_ABS_ERROR_THRESHOLDS,
    rel_error_thresholds: List[float] = DEFAULT_REL_ERROR_THRESHOLDS,
) -> np.ndarray:
    """Compute the disparity error image as in the KITTI Stereo 2015 benchmark.
    The disparity error map uses a log colormap depicting correct estimates in blue and wrong estimates in red color
    tones. We define correct disparity estimates when the absolute disparity error is less than 10 pixels and the
    relative error is less than 10% of its true value.

    Args:
        pred_disparity: Predicted disparity map.
        gt_disparity: Ground-truth disparity map.
        abs_error_thresholds: Absolute disparity error thresholds, in pixels.
        rel_error_thresholds: Relative disparity error thresholds, in pixels.

    Returns:
        disparity_error_image: Disparity error image.
    """
    # Compute errors
    abs_err = np.abs(pred_disparity - gt_disparity)
    rel_err = abs_err / np.maximum(gt_disparity, 1)

    # Compute the errors using the 0th threshold (i.e. 10 pixels) which is the principal threshold used to sort the
    # stereo evaluation leaderboard.
    err = np.minimum(abs_err / abs_error_thresholds[0], rel_err / rel_error_thresholds[0])
    disparity_error_image = np.zeros((*pred_disparity.shape, 3), dtype=np.uint8)

    for threshold, color in LOG_COLORMAP:
        disparity_error_image[np.logical_and(err >= threshold[0], err < threshold[1])] = color

    disparity_error_image[gt_disparity == 0] = 0

    disparity_error_image = np.uint8(cv2.dilate(disparity_error_image, kernel=np.ones((2, 2), np.uint8), iterations=3))

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

    # Copy the log-colormap image to the top of the disparity error image using the x and y offsets, in pixels.
    x_offset = 30
    y_offset = 30
    h, w, _ = colormap_image.shape
    disparity_error_image[y_offset : y_offset + h, x_offset : x_offset + w] = colormap_image

    return disparity_error_image
