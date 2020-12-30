# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
"""Detection evaluation unit tests.

All unit tests here do not use map ROI information (no filtering of objects
that have their centroid located outside of the ROI).
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pytest
from pandas.core.frame import DataFrame
from scipy.spatial.transform import Rotation as R

from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.evaluation.detection.eval import DetectionEvaluator
from argoverse.evaluation.detection.utils import (
    AffFnType,
    DetectionCfg,
    DistFnType,
    FilterMetric,
    accumulate,
    assign,
    compute_affinity_matrix,
    dist_fn,
    filter_instances,
    interp,
    iou_aligned_3d,
    plot,
    wrap_angle,
)
from argoverse.utils.transform import quat_scipy2argo_vectorized

TEST_DATA_LOC = Path(__file__).parent.parent / "tests" / "test_data" / "detection"
logging.getLogger("matplotlib.font_manager").disabled = True


@pytest.fixture  # type: ignore
def evaluator_identity() -> DetectionEvaluator:
    """Define an evaluator that compares a set of results to itself."""
    detection_cfg = DetectionCfg(dt_classes=["VEHICLE"], eval_only_roi_instances=False)
    return DetectionEvaluator(
        TEST_DATA_LOC / "detections_identity",
        TEST_DATA_LOC,
        TEST_DATA_LOC / "test_figures",
        detection_cfg,
    )


@pytest.fixture  # type: ignore
def evaluator_assignment() -> DetectionEvaluator:
    """Define an evaluator that compares a set of results to one with an extra detection to check assignment."""
    detection_cfg = DetectionCfg(dt_classes=["VEHICLE"], eval_only_roi_instances=False)
    return DetectionEvaluator(
        TEST_DATA_LOC / "detections_assignment",
        TEST_DATA_LOC,
        TEST_DATA_LOC / "test_figures",
        detection_cfg,
    )


@pytest.fixture  # type: ignore
def evaluator() -> DetectionEvaluator:
    """Definte an evaluator that compares a set of detections with known error to the ground truth."""
    detection_cfg = DetectionCfg(dt_classes=["VEHICLE"], eval_only_roi_instances=False)
    return DetectionEvaluator(
        TEST_DATA_LOC / "detections",
        TEST_DATA_LOC,
        TEST_DATA_LOC / "test_figures",
        detection_cfg,
    )


@pytest.fixture  # type: ignore
def metrics_identity(evaluator_identity: DetectionEvaluator) -> DataFrame:
    """Get the metrics for an evaluator that compares a set of results to itself."""
    return evaluator_identity.evaluate()


@pytest.fixture  # type: ignore
def metrics_assignment(evaluator_assignment: DetectionEvaluator) -> DataFrame:
    """Get the metrics for an evaluator that has extra detections to test for assignment errors."""
    return evaluator_assignment.evaluate()


@pytest.fixture  # type: ignore
def metrics(evaluator: DetectionEvaluator) -> DataFrame:
    """Get the metrics for an evaluator with known error."""
    return evaluator.evaluate()


def test_affinity_center() -> None:
    """Initialize a detection and a ground truth label. Verify that calculated distance matches expected affinity
    under the specified `AffFnType`.
    """
    dts: List[ObjectLabelRecord] = [
        ObjectLabelRecord(
            quaternion=np.array([1, 0, 0, 0]),
            translation=np.array([0, 0, 0]),
            length=5.0,
            width=5.0,
            height=5.0,
            occlusion=0,
        )
    ]
    gts: List[ObjectLabelRecord] = [
        ObjectLabelRecord(
            quaternion=np.array([1, 0, 0, 0]),
            translation=np.array([3, 4, 0]),
            length=5.0,
            width=5.0,
            height=5.0,
            occlusion=0,
        )
    ]

    expected_result: float = -5
    assert compute_affinity_matrix(dts, gts, AffFnType.CENTER) == expected_result


def test_translation_distance() -> None:
    """Initialize a detection and a ground truth label with only translation
    parameters. Verify that calculated distance matches expected distance under
    the specified `DistFnType`.
    """
    dts: DataFrame = DataFrame([{"translation": [0.0, 0.0, 0.0]}])
    gts: DataFrame = DataFrame([{"translation": [5.0, 5.0, 5.0]}])

    expected_result: float = np.sqrt(25 + 25 + 25)
    assert dist_fn(dts, gts, DistFnType.TRANSLATION) == expected_result


def test_scale_distance() -> None:
    """Initialize a detection and a ground truth label with only shape
    parameters (only shape parameters due to alignment assumption).
    Verify that calculated scale error matches the expected value.
    """
    dts: DataFrame = DataFrame([{"width": 5, "height": 5, "length": 5}])
    gts: DataFrame = DataFrame([{"width": 10, "height": 10, "length": 10}])

    expected_result: float = 1 - 0.125
    assert dist_fn(dts, gts, DistFnType.SCALE) == expected_result


def test_orientation_quarter_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation
    parameters. Verify that calculated orientation error matches the expected
    smallest angle ((2 * np.pi) / 4) between the detection and ground truth label.
    """

    # Check all of the 90 degree angles
    expected_result: float = (2 * np.pi) / 4
    quarter_angles = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * np.pi, expected_result)]
    for i in range(len(quarter_angles) - 1):
        dts: DataFrame = DataFrame(
            [{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(quarter_angles[i]).as_quat())}]
        )
        gts: DataFrame = DataFrame(
            [{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(quarter_angles[i + 1]).as_quat())}]
        )

        assert np.isclose(dist_fn(dts, gts, DistFnType.ORIENTATION), expected_result)
        assert np.isclose(dist_fn(gts, dts, DistFnType.ORIENTATION), expected_result)


def test_orientation_eighth_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation
    parameters. Verify that calculated orientation error matches the expected
    smallest angle ((2 * np.pi) / 8) between the detection and ground truth label.
    """
    expected_result: float = (2 * np.pi) / 8
    eigth_angle = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * np.pi, expected_result)]
    for i in range(len(eigth_angle) - 1):
        dts: DataFrame = DataFrame(
            [{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(eigth_angle[i]).as_quat())}]
        )
        gts: DataFrame = DataFrame(
            [{"quaternion": quat_scipy2argo_vectorized(R.from_rotvec(eigth_angle[i + 1]).as_quat())}]
        )

        assert np.isclose(dist_fn(dts, gts, DistFnType.ORIENTATION), expected_result)
        assert np.isclose(dist_fn(gts, dts, DistFnType.ORIENTATION), expected_result)


def test_wrap_angle() -> None:
    theta: np.ndarray = np.array([-3 * np.pi / 2])

    expected_result: float = np.array([np.pi / 2])
    assert wrap_angle(theta) == expected_result


def test_accumulate() -> None:
    """Verify that the accumulate function matches known output for a self-comparison."""
    cfg = DetectionCfg(eval_only_roi_instances=False)
    # compare a set of labels to itself
    cls_to_accum, cls_to_ninst = accumulate(
        TEST_DATA_LOC / "detections",
        TEST_DATA_LOC / "detections/1/per_sweep_annotations_amodal/tracked_object_labels_0.json",
        cfg,
        avm=None,  # ArgoverseMap instance not required when not using ROI info in evaluation
    )
    # ensure the detections match at all thresholds, have 0 TP errors, and have AP = 1
    expected_ATE = 0.0
    expected_ASE = 0.0
    expected_AOE = 0.0
    expected_AP = 1.0
    assert (
        cls_to_accum["VEHICLE"]
        == np.array(
            [
                [1.0, 1.0, 1.0, 1.0, expected_ATE, expected_ASE, expected_AOE, expected_AP],
                [1.0, 1.0, 1.0, 1.0, expected_ATE, expected_ASE, expected_AOE, expected_AP],
            ]
        )
    ).all()
    assert cls_to_ninst["VEHICLE"] == 2  # there are 2 vehicle labels in this file
    assert sum(cls_to_ninst.values()) == 2  # and no other labels


def test_assign() -> None:
    """Verify that the assign functions as expected by checking ATE of assigned detections against known distance."""
    cfg = DetectionCfg(eval_only_roi_instances=False)
    dts: np.ndarray = np.array(
        [
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([0, 0, 0]),
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
            ),
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([10, 10, 10]),
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
            ),
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([20, 20, 20]),
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
            ),
        ]
    )
    gts: np.ndarray = np.array(
        [
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([-10, -10, -10]),
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
            ),
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([0.1, 0, 0]),  # off by 0.1
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
            ),
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([10.1, 10, 10]),  # off by 0.1
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
            ),
        ]
    )
    metrics = assign(dts, gts, cfg)
    # if these assign correctly, we should get an ATE of 0.1 for the first two
    expected_result: float = 0.1
    ATE_COL_IDX = 4
    assert np.isclose(metrics[0, ATE_COL_IDX], expected_result)  # instance 0
    assert np.isclose(metrics[1, ATE_COL_IDX], expected_result)  # instance 1
    assert np.isnan(metrics[2, ATE_COL_IDX])  # instance 32


def test_filter_instances() -> None:
    """Generate 100 different detections and filter them based on Euclidean distance."""
    dts: List[ObjectLabelRecord] = [
        ObjectLabelRecord(
            translation=[i, i, 0],
            quaternion=np.array([0, 0, 0, 0]),
            length=5.0,
            width=2.0,
            height=3.0,
            occlusion=0,
            label_class="VEHICLE",
        )
        for i in range(100)
    ]

    target_class_name: str = "VEHICLE"
    filter_metric: FilterMetric = FilterMetric.EUCLIDEAN
    max_detection_range: float = 100.0

    expected_result: int = 71
    assert len(filter_instances(dts, target_class_name, filter_metric, max_detection_range)) == expected_result


def test_interp() -> None:
    """Test non-decreasing `interpolation` constraint enforced on precision results.
    See equation 2 in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    for more information."""
    prec: np.ndarray = np.array([1.0, 0.5, 0.33, 0.5])

    expected_result: np.ndarray = np.array([1.0, 0.5, 0.5, 0.5])
    assert (interp(prec) == expected_result).all()


def test_plot() -> None:
    """Test plotting functionality (i.e., plots are written to specified file)."""
    prec_interp: np.ndarray = np.array([1.0, 0.5, 0.25, 0.125])
    rec_interp: np.ndarray = np.array([0.25, 0.5, 0.75, 1.0])
    cls_name: str = "VEHICLE"
    figs_fpath: Path = Path("/tmp/figs")
    if not figs_fpath.is_dir():
        figs_fpath.mkdir(parents=True, exist_ok=True)

    expected_result: Path = Path(figs_fpath / (cls_name + ".png"))
    assert plot(rec_interp, prec_interp, cls_name, figs_fpath) == expected_result


def test_iou_aligned_3d() -> None:
    """Initialize a detection and a ground truth label with only shape
    parameters (only shape parameters due to alignment assumption).
    Verify that calculated intersection-over-union matches the expected
    value between the detection and ground truth label.
    """
    dt_dims: DataFrame = DataFrame([{"width": 10, "height": 3, "length": 4}])
    gt_dims: DataFrame = DataFrame([{"width": 5, "height": 2, "length": 9}])

    # Intersection is 40 = 4 * 5 * 2 (min of all dimensions).
    # Union is the sum of the two volumes, minus intersection: 270 = (10 * 3 * 4) + (5 * 2 * 9) - 40.
    expected_result: float = 40 / 270.0
    assert iou_aligned_3d(dt_dims, gt_dims) == expected_result


def test_assignment(metrics_assignment: DataFrame) -> None:
    """Verify that assignment works as expected; should have one duplicate in the provided results."""
    expected_result: float = 0.976
    assert metrics_assignment.AP.loc["Average Metrics"] == expected_result


def test_ap(metrics_identity: DataFrame, metrics: DataFrame) -> None:
    """Test that AP is 1 for the self-compared results."""
    expected_result: float = 1.0
    assert metrics_identity.AP.loc["Average Metrics"] == expected_result


def test_translation_error(metrics_identity: DataFrame, metrics: DataFrame) -> None:
    """Test that ATE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.017  # 0.1 / 6, one of six dets is off by 0.1
    assert metrics_identity.ATE.loc["Average Metrics"] == expected_result_identity
    assert metrics.ATE.loc["Average Metrics"] == expected_result_det


def test_scale_error(metrics_identity: DataFrame, metrics: DataFrame) -> None:
    """Test that ASE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.033  # 0.2 / 6, one of six dets is off by 20% in IoU
    assert metrics_identity.ASE.loc["Average Metrics"] == expected_result_identity
    assert metrics.ASE.loc["Average Metrics"] == expected_result_det


def test_orientation_error(metrics_identity: DataFrame, metrics: DataFrame) -> None:
    """Test that AOE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.524  # pi / 6, since one of six dets is off by pi
    assert metrics_identity.AOE.loc["Average Metrics"] == expected_result_identity
    assert metrics.AOE.loc["Average Metrics"] == expected_result_det
