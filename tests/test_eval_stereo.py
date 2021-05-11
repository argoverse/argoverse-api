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
        pred_dir, gt_dir, figs_fpath=Path("figs"), save_disparity_error_image=False, num_procs=1,
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

    The computed values are the following:

        num_pixels_bg: Number of pixels in the background region.
        num_pixels_fg: Number of pixels in the foreground region.
        num_pixels_bg_est: Number of pixels in the background region.
            Counts only the estimated disparities (no interpolation).
        num_pixels_fg_est: Number of pixels in the foreground region.
            Counts only the estimated disparities (no interpolation).
        num_errors_bg:THD: Counts the number of disparity errors (bad pixels) in the background regions using:
            bad_pixels = (abs_err > abs_error_thresh) & (rel_err > rel_error_thresh),
            where abs_err = np.abs(pred_disparity - gt_disparity), rel_err = abs_err / gt_disparity,
            abs_error_thresh (THD) is one of 10, 5, or 3 pixels, and rel_error_thresh is 0.1 (10%).
        num_errors_fg:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
        num_errors_bg_est:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
            Counts only the estimated pixels (no interpolation).
        num_errors_fg_est:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
            Counts only the estimated pixels (no interpolation).

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

    The dummy data should produce the following results:
        num_pixels_bg: 9 pixels because the dummy gt disparity map is 3 x 3 and all pixels have valid disparities.
        num_pixels_fg: 0 pixels because the dummy gt_obj has no foreground objects.
        num_pixels_bg_est: 9 pixels because the dummy pred disparity map is 3 x 3 and all pixels have valid disparities.
        num_pixels_fg_est: 0 pixels because the dummy gt_obj has no foreground objects.

        num_errors_bg:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
        num_errors_fg:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
        num_errors_bg_est:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
        num_errors_fg_est:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
    """
    gt_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_1/disparity_gt/disparity_1.png")
    gt_obj_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_1/disparity_gt/disparity_objects_1.png")
    pred_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_1/disparity_pred/disparity_1.png")

    errors = compute_disparity_error(
        pred_fpath,
        gt_fpath,
        gt_obj_fpath,
        figs_fpath=Path("figs"),
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

    The computed values are the following:

        num_pixels_bg: Number of pixels in the background region.
        num_pixels_fg: Number of pixels in the foreground region.
        num_pixels_bg_est: Number of pixels in the background region.
            Counts only the estimated disparities (no interpolation).
        num_pixels_fg_est: Number of pixels in the foreground region.
            Counts only the estimated disparities (no interpolation).
        num_errors_bg:THD: Counts the number of disparity errors (bad pixels) in the background regions using:
            bad_pixels = (abs_err > abs_error_thresh) & (rel_err > rel_error_thresh),
            where abs_err = np.abs(pred_disparity - gt_disparity), rel_err = abs_err / gt_disparity,
            abs_error_thresh (THD) is one of 10, 5, or 3 pixels, and rel_error_thresh is 0.1 (10%).
        num_errors_fg:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
        num_errors_bg_est:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
            Counts only the estimated pixels (no interpolation).
        num_errors_fg_est:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
            Counts only the estimated pixels (no interpolation).

    Dummy test images (3 x 3):
        pred = np.array([[  1.0,   5.0,  10.0],
                         [ 50.0,  70.0,  90.0],
                         [100.0, 150.0, 200.0]], dtype=np.float32)

        gt = np.array([[  1.0,   5.0,  10.0],
                       [ 50.0,  70.0,  90.0],
                       [100.0, 150.0, 200.0]], dtype=np.float32)

        gt_obj = np.array([[ 1.0,  5.0, 0.0],
                           [50.0, 70.0, 0.0],
                           [ 0.0,  0.0, 0.0]], dtype=np.float32)

    pred is the predicted disparity map, gt is the ground-truth disparity map, and gt_obj is the ground-truth disparity
    map for foreground objects.

    The dummy data should produce the following results:
        num_pixels_bg: 5 pixels in the background because now there are 4 pixels in the foreground region.
        num_pixels_fg: 4 pixels in the foreground region.
        num_pixels_bg_est: 5 pixels in the background because now there are 4 pixels in the foreground region.
        num_pixels_fg_est: 4 pixels in the foreground region.

        num_errors_bg:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
        num_errors_fg:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
        num_errors_bg_est:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
        num_errors_fg_est:THD: 0 errors in all thresholds because the gt=pred (using the equations described earlier).
    """
    gt_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_2/disparity_gt/disparity_1.png")
    gt_obj_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_2/disparity_gt/disparity_objects_1.png")
    pred_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_2/disparity_pred/disparity_1.png")

    errors = compute_disparity_error(
        pred_fpath,
        gt_fpath,
        gt_obj_fpath,
        figs_fpath=Path("figs"),
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

    The computed values are the following:

        num_pixels_bg: Number of pixels in the background region.
        num_pixels_fg: Number of pixels in the foreground region.
        num_pixels_bg_est: Number of pixels in the background region.
            Counts only the estimated disparities (no interpolation).
        num_pixels_fg_est: Number of pixels in the foreground region.
            Counts only the estimated disparities (no interpolation).
        num_errors_bg:THD: Counts the number of disparity errors (bad pixels) in the background regions using:
            bad_pixels = (abs_err > abs_error_thresh) & (rel_err > rel_error_thresh),
            where abs_err = np.abs(pred_disparity - gt_disparity), rel_err = abs_err / gt_disparity,
            abs_error_thresh (THD) is one of 10, 5, or 3 pixels, and rel_error_thresh is 0.1 (10%).
        num_errors_fg:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
        num_errors_bg_est:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
            Counts only the estimated pixels (no interpolation).
        num_errors_fg_est:THD: Counts the number of disparity errors (bad pixels) in the foreground regions.
            Counts only the estimated pixels (no interpolation).

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

    The dummy data should produce the following results:
        num_pixels_bg: 9 pixels because the dummy gt disparity map is 3 x 3 and all pixels have valid disparities.
        num_pixels_fg: 0 pixels because the dummy gt_obj has no foreground objects.
        num_pixels_bg_est: 9 pixels because the dummy pred disparity map is 3 x 3 and all pixels have valid disparities.
        num_pixels_fg_est: 0 pixels because the dummy gt_obj has no foreground objects.

        num_errors_bg:10: 4 errors if computed as: num_errors_bg:10 = (abs_err > 10) & (rel_err > 0.1).
        num_errors_fg:10: 0 errors because there are no foreground objects.
        num_errors_bg_est:10: 4 errors if computed as: num_errors_bg_est:10 = (abs_err > 10) & (rel_err > 0.1).
        num_errors_fg_est:10: 0 errors because there are no foreground objects.

        num_errors_bg:5: 5 errors if computed as: num_errors_bg:5 = (abs_err > 5) & (rel_err > 0.1).
        num_errors_fg:5: 0 errors because there are no foreground objects.
        num_errors_bg_est:5: 5 errors if computed as: num_errors_bg_est:5 = (abs_err > 5) & (rel_err > 0.1).
        num_errors_fg_est:5: 0 errors because there are no foreground objects.

        num_errors_bg:3: 5 errors if computed as: num_errors_bg:3 = (abs_err > 3) & (rel_err > 0.1).
        num_errors_fg:3: 0 errors because there are no foreground objects.
        num_errors_bg_est:3: 5 errors if computed as: num_errors_bg_est:3 = (abs_err > 3) & (rel_err > 0.1).
        num_errors_fg_est:3: 0 errors because there are no foreground objects.
    """
    gt_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_3/disparity_gt/disparity_1.png")
    gt_obj_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_3/disparity_gt/disparity_objects_1.png")
    pred_fpath = Path(f"{_ROOT}/test_data/stereo/eval_test_cases/dummy_case_3/disparity_pred/disparity_1.png")

    errors = compute_disparity_error(
        pred_fpath,
        gt_fpath,
        gt_obj_fpath,
        figs_fpath=Path("figs"),
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
