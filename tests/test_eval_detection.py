# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
"""Detection evaluation unit tests"""

import logging
import pathlib

import numpy as np
import pytest
from pandas.core.frame import DataFrame
from scipy.spatial.transform import Rotation as R

from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.evaluation.detection_utils import AffFnType, DistFnType, compute_affinity_matrix, dist_fn, iou_aligned_3d
from argoverse.evaluation.eval_detection import DetectionCfg, DetectionEvaluator
from argoverse.utils.transform import quat_scipy2argo_vectorized

TEST_DATA_LOC = pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "detection"
logging.getLogger("matplotlib.font_manager").disabled = True


@pytest.fixture
def evaluator() -> DetectionEvaluator:
    """Define an evaluator that compares a set of results to itself."""
    detection_cfg = DetectionCfg(significant_digits=32, detection_classes=["VEHICLE"])
    return DetectionEvaluator(
        TEST_DATA_LOC / "detections", TEST_DATA_LOC, TEST_DATA_LOC / "test_figures", detection_cfg
    )


@pytest.fixture
def metrics(evaluator: DetectionEvaluator) -> DataFrame:
    return evaluator.evaluate()


def test_affinity_center() -> None:
    """Intialize a detection and a ground truth label. Verify that calculated distance matches expected affinity
    under the specified `AffFnType`.
    """
    dts = [ObjectLabelRecord(np.array([0, 0, 0, 0]), np.array([0, 0, 0]), 5.0, 5.0, 5.0, 0)]
    gts = [ObjectLabelRecord(np.array([0, 0, 0, 0]), np.array([3, 4, 0]), 5.0, 5.0, 5.0, 0)]
    assert compute_affinity_matrix(dts, gts, AffFnType.CENTER) == -5


def test_translation_distance() -> None:
    """Intialize a detection and a ground truth label with only translation
    parameters. Verify that calculated distance matches expected distance under
    the specified `DistFnType`.
    """
    dts = DataFrame([{"translation": [0.0, 0.0, 0.0]}])
    gts = DataFrame([{"translation": [5.0, 5.0, 5.0]}])
    assert dist_fn(dts, gts, DistFnType.TRANSLATION) == 75 ** (1 / 2)


def test_scale_distance() -> None:
    """Intialize a detection and a ground truth label with only shape
    parameters (only shape parameters due to alignment assumption).
    Verify that calculated scale error matches the expected value.
    """
    dts = DataFrame([{"width": 5, "height": 5, "length": 5}])
    gts = DataFrame([{"width": 10, "height": 10, "length": 10}])
    assert (dist_fn(dts, gts, DistFnType.SCALE) == 1 - 0.125).all()


def test_orientation_distance() -> None:
    """Intialize a detection and a ground truth label with only orientation
    parameters. Verify that calculated orientation error matches the expected
    smallest angle between the detection and ground truth label.
    """
    # check all of the 45 degree angles
    vecs_45_apart = [angle * np.array([0, 0, 1]) for angle in np.arange(0, 2 * np.pi, np.pi / 4)]
    for i in range(len(vecs_45_apart) - 1):
        dts = DataFrame([{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(vecs_45_apart[i]).as_quat())}])
        gts = DataFrame([{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(vecs_45_apart[i + 1]).as_quat())}])
        assert np.isclose(dist_fn(dts, gts, DistFnType.ORIENTATION), np.pi / 4)
        assert np.isclose(dist_fn(gts, dts, DistFnType.ORIENTATION), np.pi / 4)
    # check all of the 90 degree angles
    vecs_90_apart = [angle * np.array([0, 0, 1]) for angle in np.arange(0, 2 * np.pi, np.pi / 2)]
    for i in range(len(vecs_90_apart) - 1):
        dts = DataFrame([{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(vecs_90_apart[i]).as_quat())}])
        gts = DataFrame([{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(vecs_90_apart[i + 1]).as_quat())}])
        assert np.isclose(dist_fn(dts, gts, DistFnType.ORIENTATION), np.pi / 2)
        assert np.isclose(dist_fn(gts, dts, DistFnType.ORIENTATION), np.pi / 2)


def test_iou_aligned_3d() -> None:
    """Intialize a detection and a ground truth label with only shape
    parameters (only shape parameters due to alignment assumption).
    Verify that calculated intersection-over-union matches the expected
    value between the detection and ground truth label.
    """
    dt_dims = DataFrame([{"width": 10, "height": 3, "length": 4}])
    gt_dims = DataFrame([{"width": 5, "height": 2, "length": 9}])

    # Intersection is 40 = 4 * 5 * 2 (min of all dimensions).
    # Union is the sum of the two volumes, minus intersection: 270 = (10 * 3 * 4) + (5 * 2 * 9) - 40.
    assert (iou_aligned_3d(dt_dims, gt_dims) == (40 / 270.0)).all()


def test_ap(metrics: DataFrame) -> None:
    """Test that AP is 1 for the self-compared results."""
    assert metrics.AP.loc["Average Metrics"] == 1


def test_translation_error(metrics: DataFrame) -> None:
    """Test that ATE is 0 for the self-compared results."""
    assert metrics.ATE.loc["Average Metrics"] == 0


def test_scale_error(metrics: DataFrame) -> None:
    """Test that ASE is 0 for the self-compared results."""
    assert metrics.ASE.loc["Average Metrics"] == 0


def test_orientation_error(metrics: DataFrame) -> None:
    """Test that AOE is 0 for the self-compared results."""
    assert metrics.AOE.loc["Average Metrics"] == 0
