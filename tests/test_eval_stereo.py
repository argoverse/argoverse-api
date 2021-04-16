# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Stereo evaluation unit tests."""

import math
from pathlib import Path

from argoverse.evaluation.stereo.constants import DEFAULT_ABS_ERROR_THRESHOLDS, DEFAULT_REL_ERROR_THRESHOLDS
from argoverse.evaluation.stereo.eval import StereoEvaluator
from argoverse.evaluation.stereo.utils import compute_disparity_error

_ROOT = Path(__file__).resolve().parent


def test_stereo_evaluation() -> None:
    """Test the stereo evaluation using real predictions and known results."""

    pred_dir = Path(f"{_ROOT}/test_data/stereo/prediction/")
    gt_dir = Path(f"{_ROOT}/test_data/stereo/disparity_maps_v1.1/test/")

    evaluator = StereoEvaluator(
        pred_dir,
        gt_dir,
        figs_fpath=None,
        save_disparity_error_image=False,
        num_procs=1,
    )

    summary = evaluator.evaluate()

    assert math.isclose(summary["all:10"], 5.446183637)
    assert math.isclose(summary["fg:10"], 9.600825877)
    assert math.isclose(summary["bg:10"], 3.556616323)
    assert math.isclose(summary["all*:10"], 3.811615737)
    assert math.isclose(summary["fg*:10"], 7.697195243)
    assert math.isclose(summary["bg*:10"], 1.664911488)

    assert math.isclose(summary["all:5"], 21.389381959)
    assert math.isclose(summary["fg:5"], 16.070199587)
    assert math.isclose(summary["bg:5"], 23.808592221)
    assert math.isclose(summary["all*:5"], 17.171651915)
    assert math.isclose(summary["fg*:5"], 14.114550240)
    assert math.isclose(summary["bg*:5"], 18.860638884)

    assert math.isclose(summary["all:3"], 29.672960034)
    assert math.isclose(summary["fg:3"], 18.530626290)
    assert math.isclose(summary["bg:3"], 34.740590030)
    assert math.isclose(summary["all*:3"], 24.646294980)
    assert math.isclose(summary["fg*:3"], 16.61069256)
    assert math.isclose(summary["bg*:3"], 29.08580311)


def test_compute_disparity_error_dummy_1() -> None:
    """Test the computation of the disparity errors using an exact disparity prediction for background regions only.

    Dummy test images (3 x 3):
        pred = np.array([[  1.0,   5.0,  10.0],
                         [ 50.0,  70.0,  90.0],
                         [100.0, 150.0, 200.0]], dtype=np.float32)

        gt = np.array([[  1.0,   5.0,  10.0],
                       [ 50.0,  70.0,  90.0],
                       [100.0, 150.0, 200.0]], dtype=np.float32)

        gt_obj = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]], dtype=np.float32)

    pred is the predicted disparity map, gt is the ground-truth disparity map, and gt_obj is the ground-truth disparity
    map for foreground objects.
    """
    gt_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_1/disparity_gt/disparity_1.png")
    gt_obj_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_1/disparity_gt/disparity_objects_1.png")
    pred_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_1/disparity_pred/disparity_1.png")

    errors = compute_disparity_error(
        pred_fpath,
        gt_fpath,
        gt_obj_fpath,
        figs_fpath=None,
        abs_error_thresholds=DEFAULT_ABS_ERROR_THRESHOLDS,
        rel_error_thresholds=DEFAULT_REL_ERROR_THRESHOLDS,
        save_disparity_error_image=False,
    )

    assert int(errors["num_pixels_bg"]) == 9
    assert int(errors["num_pixels_fg"]) == 0
    assert int(errors["num_pixels_bg_est"]) == 9
    assert int(errors["num_pixels_fg_est"]) == 0
    assert int(errors["num_errors_bg:10"]) == 0
    assert int(errors["num_errors_fg:10"]) == 0
    assert int(errors["num_errors_bg_est:10"]) == 0
    assert int(errors["num_errors_fg_est:10"]) == 0
    assert int(errors["num_errors_bg:5"]) == 0
    assert int(errors["num_errors_fg:5"]) == 0
    assert int(errors["num_errors_bg_est:5"]) == 0
    assert int(errors["num_errors_fg_est:5"]) == 0
    assert int(errors["num_errors_bg:3"]) == 0
    assert int(errors["num_errors_fg:3"]) == 0
    assert int(errors["num_errors_bg_est:3"]) == 0
    assert int(errors["num_errors_fg_est:3"]) == 0


def test_compute_disparity_error_dummy_2() -> None:
    """Test the computation of the disparity errors using an exact disparity prediction for background and foreground
    regions.

    Dummy test images (3 x 3):
        pred = np.array([[  1.0,   5.0,  10.0],
                         [ 50.0,  70.0,  90.0],
                         [100.0, 150.0, 200.0]], dtype=np.float32)

        gt = np.array([[ 1.0,  5.0, 0.0],
                       [50.0, 70.0, 0.0],
                       [ 0.0,  0.0, 0.0]], dtype=np.float32)

        gt_obj = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]], dtype=np.float32)

    pred is the predicted disparity map, gt is the ground-truth disparity map, and gt_obj is the ground-truth disparity
    map for foreground objects.
    """
    gt_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_2/disparity_gt/disparity_1.png")
    gt_obj_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_2/disparity_gt/disparity_objects_1.png")
    pred_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_2/disparity_pred/disparity_1.png")

    errors = compute_disparity_error(
        pred_fpath,
        gt_fpath,
        gt_obj_fpath,
        figs_fpath=None,
        abs_error_thresholds=DEFAULT_ABS_ERROR_THRESHOLDS,
        rel_error_thresholds=DEFAULT_REL_ERROR_THRESHOLDS,
        save_disparity_error_image=False,
    )

    assert int(errors["num_pixels_bg"]) == 5
    assert int(errors["num_pixels_fg"]) == 4
    assert int(errors["num_pixels_bg_est"]) == 5
    assert int(errors["num_pixels_fg_est"]) == 4
    assert int(errors["num_errors_bg:10"]) == 0
    assert int(errors["num_errors_fg:10"]) == 0
    assert int(errors["num_errors_bg_est:10"]) == 0
    assert int(errors["num_errors_fg_est:10"]) == 0
    assert int(errors["num_errors_bg:5"]) == 0
    assert int(errors["num_errors_fg:5"]) == 0
    assert int(errors["num_errors_bg_est:5"]) == 0
    assert int(errors["num_errors_fg_est:5"]) == 0
    assert int(errors["num_errors_bg:3"]) == 0
    assert int(errors["num_errors_fg:3"]) == 0
    assert int(errors["num_errors_bg_est:3"]) == 0
    assert int(errors["num_errors_fg_est:3"]) == 0


def test_compute_disparity_error_dummy_3() -> None:
    """Test the computation of the disparity errors using a non-exact disparity prediction for background regions only.

    Dummy test images (3 x 3):
        pred = np.array([[  2.0,   4.0,  10.0],
                         [ 70.0,  60.0,  50.0],
                         [150.0, 120.0, 190.0]], dtype=np.float32)

        gt = np.array([[  1.0,   5.0,  10.0],
                       [ 50.0,  70.0,  90.0],
                       [100.0, 150.0, 200.0]], dtype=np.float32)

        gt_obj = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]], dtype=np.float32)

    pred is the predicted disparity map, gt is the ground-truth disparity map, and gt_obj is the ground-truth disparity
    map for foreground objects.
    """
    gt_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_3/disparity_gt/disparity_1.png")
    gt_obj_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_3/disparity_gt/disparity_objects_1.png")
    pred_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_3/disparity_pred/disparity_1.png")

    errors = compute_disparity_error(
        pred_fpath,
        gt_fpath,
        gt_obj_fpath,
        figs_fpath=None,
        abs_error_thresholds=DEFAULT_ABS_ERROR_THRESHOLDS,
        rel_error_thresholds=DEFAULT_REL_ERROR_THRESHOLDS,
        save_disparity_error_image=False,
    )

    assert int(errors["num_pixels_bg"]) == 9
    assert int(errors["num_pixels_fg"]) == 0
    assert int(errors["num_pixels_bg_est"]) == 9
    assert int(errors["num_pixels_fg_est"]) == 0
    assert int(errors["num_errors_bg:10"]) == 4
    assert int(errors["num_errors_fg:10"]) == 0
    assert int(errors["num_errors_bg_est:10"]) == 4
    assert int(errors["num_errors_fg_est:10"]) == 0
    assert int(errors["num_errors_bg:5"]) == 5
    assert int(errors["num_errors_fg:5"]) == 0
    assert int(errors["num_errors_bg_est:5"]) == 5
    assert int(errors["num_errors_fg_est:5"]) == 0
    assert int(errors["num_errors_bg:3"]) == 5
    assert int(errors["num_errors_fg:3"]) == 0
    assert int(errors["num_errors_bg_est:3"]) == 5
    assert int(errors["num_errors_fg_est:3"]) == 0


if __name__ == "__main__":
    """ """
    test_stereo_evaluation()
    test_compute_disparity_error_dummy_1()
    test_compute_disparity_error_dummy_2()
    test_compute_disparity_error_dummy_3()
