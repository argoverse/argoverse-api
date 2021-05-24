# <Copyright 2020, Argo AI, LLC. Released under the MIT license.>
"""Detection evaluation unit tests.

Only the last two unit tests here use map ROI information.
The rest apply no filtering to objects that have their corners located outside of the ROI).
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
from pandas.core.frame import DataFrame
from scipy.spatial.transform import Rotation as R

from argoverse.data_loading.object_label_record import ObjectLabelRecord, json_label_dict_to_obj_record
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
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
    filter_objs_to_roi,
    interp,
    iou_aligned_3d,
    plot,
    rank,
    remove_duplicate_instances,
    wrap_angle,
)
from argoverse.map_representation.map_api import ArgoverseMap
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
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    expected_ATE,
                    expected_ASE,
                    expected_AOE,
                    expected_AP,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    expected_ATE,
                    expected_ASE,
                    expected_AOE,
                    expected_AP,
                ],
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


def test_remove_duplicate_instances() -> None:
    """Ensure a duplicate ground truth cuboid can be filtered out correctly."""
    instances = [
        SimpleNamespace(**{"translation": np.array([1, 1, 0])}),
        SimpleNamespace(**{"translation": np.array([5, 5, 0])}),
        SimpleNamespace(**{"translation": np.array([2, 2, 0])}),
        SimpleNamespace(**{"translation": np.array([5, 5, 0])}),
    ]
    instances = np.array(instances)
    cfg = DetectionCfg(eval_only_roi_instances=False)
    unique_instances = remove_duplicate_instances(instances, cfg)

    assert len(unique_instances) == 3
    assert np.allclose(unique_instances[0].translation, np.array([1, 1, 0]))
    assert np.allclose(unique_instances[1].translation, np.array([5, 5, 0]))
    assert np.allclose(unique_instances[2].translation, np.array([2, 2, 0]))


def test_remove_duplicate_instances_ground_truth() -> None:
    """Ensure that if an extra duplicate cuboid is present in ground truth, it would be ignored."""
    dt_fpath = TEST_DATA_LOC / "remove_duplicates_detections"
    gt_fpath = TEST_DATA_LOC / "remove_duplicates_ground_truth"
    fig_fpath = TEST_DATA_LOC / "test_figures"

    cfg = DetectionCfg(eval_only_roi_instances=False)
    evaluator = DetectionEvaluator(dt_fpath, gt_fpath, fig_fpath, cfg)
    metrics = evaluator.evaluate()
    assert metrics.AP.loc["Vehicle"] == 1.0
    assert metrics.AP.loc["Pedestrian"] == 1.0


def test_filter_objs_to_roi() -> None:
    """Use the map to filter out an object that lies outside the ROI in a parking lot."""
    avm = ArgoverseMap()

    # should be outside of ROI
    outside_obj = {
        "center": {
            "x": -14.102872067388489,
            "y": 19.466695178746022,
            "z": 0.11740010190455852,
        },
        "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": -0.038991328555453404,
            "w": 0.9992395490058831,
        },
        "length": 4.56126567460171,
        "width": 1.9370055686754908,
        "height": 1.5820081349372281,
        "track_label_uuid": "03a321bf955a4d7781682913884abf06",
        "timestamp": 315970611820366000,
        "label_class": "VEHICLE",
    }

    # should be inside the ROI
    inside_obj = {
        "center": {
            "x": -20.727430239506702,
            "y": 3.4488006757501353,
            "z": 0.4036619561689685,
        },
        "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0013102003738908123,
            "w": 0.9999991416871218,
        },
        "length": 4.507580779458834,
        "width": 1.9243189627993598,
        "height": 1.629934978730058,
        "track_label_uuid": "bb0f40e4f68043e285d64a839f2f092c",
        "timestamp": 315970611820366000,
        "label_class": "VEHICLE",
    }

    log_city_name = "PIT"
    lidar_ts = 315970611820366000
    dataset_dir = TEST_DATA_LOC / "roi_based_test"
    log_id = "21e37598-52d4-345c-8ef9-03ae19615d3d"
    city_SE3_egovehicle = get_city_SE3_egovehicle_at_sensor_t(lidar_ts, dataset_dir, log_id)

    dts = np.array([json_label_dict_to_obj_record(item) for item in [outside_obj, inside_obj]])
    dts_filtered = filter_objs_to_roi(dts, avm, city_SE3_egovehicle, log_city_name)

    assert dts_filtered.size == 1
    assert dts_filtered.dtype == "O"  # array of objects
    assert isinstance(dts_filtered, np.ndarray)
    assert dts_filtered[0].track_id == "bb0f40e4f68043e285d64a839f2f092c"


def test_AP_on_filtered_instances() -> None:
    """Test AP calculation on instances filtered on region-of-interest."""
    dt_fpath = TEST_DATA_LOC / "remove_nonroi_detections"
    gt_fpath = TEST_DATA_LOC / "remove_nonroi_ground_truth"
    fig_fpath = TEST_DATA_LOC / "test_figures"

    cfg = DetectionCfg(eval_only_roi_instances=True)
    evaluator = DetectionEvaluator(dt_fpath, gt_fpath, fig_fpath, cfg)
    metrics = evaluator.evaluate()

    assert metrics.AP.loc["Vehicle"] == 1.0


def test_rank() -> None:
    """Test ranking of detections and scores during detection evaluation."""
    dts: np.ndarray = np.array(
        [
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([0, 0, 0]),
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
                score=0.7,
                track_id="0",
            ),
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([10, 10, 10]),
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
                score=0.9,
                track_id="1",
            ),
            ObjectLabelRecord(
                quaternion=np.array([1, 0, 0, 0]),
                translation=np.array([20, 20, 20]),
                length=5.0,
                width=5.0,
                height=5.0,
                occlusion=0,
                score=0.8,
                track_id="2",
            ),
        ]
    )

    ranked_dts, ranked_scores = rank(dts)
    track_ids = np.array([dt.track_id for dt in ranked_dts.tolist()])
    expected_track_ids = np.array(["1", "2", "0"])
    expected_scores = np.array([0.9, 0.8, 0.7])
    assert (track_ids == expected_track_ids).all() and (ranked_scores == expected_scores).all()
