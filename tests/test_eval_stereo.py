# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Stereo evaluation unit tests.

Run the stereo evaluation on the sample data and compare the results to the known metrics computed beforehand.
"""

import math
from pathlib import Path

from argoverse.evaluation.stereo.eval import StereoEvaluator

_ROOT = Path(__file__).resolve().parent


def test_stereo_evaluation() -> None:
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


if __name__ == "__main__":
    """ """
    test_stereo_evaluation()
