# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Detection evaluation unit tests.

Only the last two unit tests here use map ROI information.
The rest apply no filtering to objects that have their corners located outside of the ROI).
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

from argoverse.evaluation.detection.eval import evaluate
from argoverse.evaluation.detection.utils import (
    AffFnType,
    DetectionCfg,
    DistFnType,
    accumulate,
    assign,
    compute_affinity_matrix,
    dist_fn,
    filter_instances,
    interp,
    iou_aligned_3d,
    rank,
    wrap_angle,
)

TEST_DATA_LOC = Path(__file__).parent.parent / "detection" / "data"
logging.getLogger("matplotlib.font_manager").disabled = True


@pytest.fixture  # type: ignore
def metrics_identity() -> pd.DataFrame:
    """Define an evaluator that compares a set of results to itself."""
    detection_cfg = DetectionCfg(dt_classes=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_LOC / "detections_identity.feather")
    metrics = evaluate(dts, dts, None, detection_cfg)
    return metrics


@pytest.fixture  # type: ignore
def metrics_assignment() -> pd.DataFrame:
    """Define an evaluator that compares a set of results to one with an extra detection to check assignment."""
    detection_cfg = DetectionCfg(dt_classes=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_LOC / "detections_assignment.feather")
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_LOC / "labels.feather")
    metrics = evaluate(dts, gts, None, detection_cfg)
    return metrics


@pytest.fixture  # type: ignore
def metrics() -> pd.DataFrame:
    detection_cfg = DetectionCfg(dt_classes=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_LOC / "detections.feather")
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_LOC / "labels.feather")
    metrics = evaluate(dts, gts, None, detection_cfg)
    return metrics


def test_affinity_center() -> None:
    """Initialize a detection and a ground truth label. Verify that calculated distance matches expected affinity
    under the specified `AffFnType`.
    """

    columns = ["tx", "ty", "tz", "length", "width", "height", "qw", "qx", "qy", "qz"]
    dts = pd.DataFrame([[0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]], columns=columns)
    gts = pd.DataFrame([[3.0, 4.0, 0.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]], columns=columns)

    expected_result: float = -5
    assert compute_affinity_matrix(dts, gts, AffFnType.CENTER) == expected_result


def test_translation_distance() -> None:
    """Initialize a detection and a ground truth label with only translation
    parameters. Verify that calculated distance matches expected distance under
    the specified `DistFnType`.
    """
    columns = ["tx", "ty", "tz"]
    dts = pd.DataFrame([[0.0, 0.0, 0.0]], columns=columns)
    gts = pd.DataFrame([[5.0, 5.0, 5.0]], columns=columns)

    expected_result: float = np.sqrt(25 + 25)
    assert np.allclose(dist_fn(dts, gts, DistFnType.TRANSLATION), expected_result)


def test_scale_distance() -> None:
    """Initialize a detection and a ground truth label with only shape
    parameters (only shape parameters due to alignment assumption).
    Verify that calculated scale error matches the expected value.
    """
    columns = ["length", "width", "height"]
    dts = pd.DataFrame([[5.0, 5.0, 5.0]], columns=columns)
    gts = pd.DataFrame([[10.0, 10.0, 10.0]], columns=columns)

    expected_result: float = 1 - 0.125
    assert np.allclose(dist_fn(dts, gts, DistFnType.SCALE), expected_result)


def test_orientation_quarter_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation
    parameters. Verify that calculated orientation error matches the expected
    smallest angle ((2 * np.pi) / 4) between the detection and ground truth label.
    """

    # Check all of the 90 degree angles
    expected_result: float = (2 * np.pi) / 4
    quarter_angles = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * np.pi, expected_result)]

    for i in range(len(quarter_angles) - 1):
        quat_xyzw_dts = R.from_rotvec(quarter_angles[i : i + 1]).as_quat()
        quat_xyzw_gts = R.from_rotvec(quarter_angles[i + 1 : i + 2]).as_quat()

        quat_wxyz_dts = quat_xyzw_dts[..., [3, 0, 1, 2]]
        quat_wxyz_gts = quat_xyzw_gts[..., [3, 0, 1, 2]]

        columns = ["qw", "qx", "qy", "qz"]
        dts = pd.DataFrame(quat_wxyz_dts, columns=columns)
        gts = pd.DataFrame(quat_wxyz_gts, columns=columns)

        assert np.isclose(dist_fn(dts, gts, DistFnType.ORIENTATION), expected_result)
        assert np.isclose(dist_fn(gts, dts, DistFnType.ORIENTATION), expected_result)


def test_orientation_eighth_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation
    parameters. Verify that calculated orientation error matches the expected
    smallest angle ((2 * np.pi) / 8) between the detection and ground truth label.
    """
    expected_result: float = (2 * np.pi) / 8
    eigth_angles = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * np.pi, expected_result)]

    for i in range(len(eigth_angles) - 1):
        quat_xyzw_dts = R.from_rotvec(eigth_angles[i : i + 1]).as_quat()
        quat_xyzw_gts = R.from_rotvec(eigth_angles[i + 1 : i + 2]).as_quat()

        quat_wxyz_dts = quat_xyzw_dts[..., [3, 0, 1, 2]]
        quat_wxyz_gts = quat_xyzw_gts[..., [3, 0, 1, 2]]

        columns = ["qw", "qx", "qy", "qz"]
        dts = pd.DataFrame(quat_wxyz_dts, columns=columns)
        gts = pd.DataFrame(quat_wxyz_gts, columns=columns)

        assert np.isclose(dist_fn(dts, gts, DistFnType.ORIENTATION), expected_result)
        assert np.isclose(dist_fn(gts, dts, DistFnType.ORIENTATION), expected_result)


def test_wrap_angle() -> None:
    theta = np.array([-3 * np.pi / 2])
    expected_result = np.array([np.pi / 2])
    assert np.isclose(wrap_angle(theta), expected_result)


def test_accumulate() -> None:
    """Verify that the accumulate function matches known output for a self-comparison."""
    cfg = DetectionCfg(eval_only_roi_instances=False)
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_LOC / "labels.feather")
    poses = None

    for _, group in gts.groupby(["log_id", "tov_ns"]):
        job = (group, group, poses, cfg)
        cat_stats, cls_to_ninst = accumulate(job)

        # Check that there's a true positive under every threshold.
        assert np.all(cat_stats.iloc[:, :4])

        # Check that all error metrics are zero.
        assert np.nonzero(cat_stats.iloc[:, 4:7].to_numpy())[0].shape[0] == 0

        # Check that there are 2 regular vehicles.
        assert cls_to_ninst["REGULAR_VEHICLE"] == 2

        # Check that there are no other labels.
        assert sum(cls_to_ninst.values()) == 2


def test_assign() -> None:
    """Verify that the assign functions as expected by checking ATE of assigned detections against known distance."""
    cfg = DetectionCfg(eval_only_roi_instances=False)

    columns = ["length", "width", "height", "qw", "qx", "qy", "qz", "tx", "ty", "tz", "score"]
    dts = pd.DataFrame(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 1.0],
        ],
        columns=columns,
    )

    gts = pd.DataFrame(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 10.1, 10.0, 10.0, 1.0],
        ],
        columns=columns,
    )

    metrics = assign(dts, gts, cfg)
    # if these assign correctly, we should get an ATE of 0.1 for the first two
    expected_result: float = 0.1
    ATE_COL_IDX = 4

    assert math.isclose(metrics.iloc[0, ATE_COL_IDX], expected_result)  # instance 0
    assert math.isclose(metrics.iloc[1, ATE_COL_IDX], expected_result)  # instance 1
    assert math.isnan(metrics.iloc[2, ATE_COL_IDX])  # instance 32


def test_filter_instances() -> None:
    """Generate 100 different detections and filter them based on Euclidean distance."""
    columns = ["category", "length", "width", "height", "qw", "qx", "qy", "qz", "tx", "ty", "tz"]

    dts = pd.DataFrame(
        [["REGULAR_VEHICLE", 0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 3.0, i, i, 0] for i in range(100)], columns=columns
    )
    cfg = DetectionCfg(eval_only_roi_instances=False)

    expected_result: int = 71
    assert len(filter_instances(dts, cfg)) == expected_result


def test_interp() -> None:
    """Test non-decreasing `interpolation` constraint enforced on precision results.
    See equation 2 in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    for more information."""
    prec: np.ndarray = np.array([1.0, 0.5, 0.33, 0.5])
    expected_result: np.ndarray = np.array([1.0, 0.5, 0.5, 0.5])
    assert np.isclose(interp(prec), expected_result).all()


# def test_plot() -> None:
#     """Test plotting functionality (i.e., plots are written to specified file)."""
#     prec_interp: np.ndarray = np.array([1.0, 0.5, 0.25, 0.125])
#     rec_interp: np.ndarray = np.array([0.25, 0.5, 0.75, 1.0])
#     cls_name: str = "VEHICLE"
#     figs_fpath: Path = Path("/tmp/figs")
#     if not figs_fpath.is_dir():
#         figs_fpath.mkdir(parents=True, exist_ok=True)

#     expected_result: Path = Path(figs_fpath / (cls_name + ".png"))
#     assert plot(rec_interp, prec_interp, cls_name, figs_fpath) == expected_result


def test_iou_aligned_3d() -> None:
    """Initialize a detection and a ground truth label with only shape
    parameters (only shape parameters due to alignment assumption).
    Verify that calculated intersection-over-union matches the expected
    value between the detection and ground truth label.
    """
    columns = ["length", "width", "height"]
    dt_dims = pd.DataFrame([[4.0, 10.0, 3.0]], columns=columns).to_numpy()
    gt_dims = pd.DataFrame([[9.0, 5.0, 2.0]], columns=columns).to_numpy()

    # Intersection is 40 = 4 * 5 * 2 (min of all dimensions).
    # Union is the sum of the two volumes, minus intersection: 270 = (10 * 3 * 4) + (5 * 2 * 9) - 40.
    expected_result: float = 40 / 270.0
    assert iou_aligned_3d(dt_dims, gt_dims) == expected_result


def test_assignment(metrics_assignment: pd.DataFrame) -> None:
    """Verify that assignment works as expected; should have one duplicate in the provided results."""
    expected_result: float = 0.999
    assert metrics_assignment.loc["Average Metrics", "AP"] == expected_result


def test_ap(metrics_identity: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Test that AP is 1 for the self-compared results."""
    expected_result: float = 1.0
    assert metrics_identity.loc["Average Metrics", "AP"] == expected_result


def test_translation_error(metrics_identity: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Test that ATE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.017  # 0.1 / 6, one of six dets is off by 0.1
    assert metrics_identity.loc["Average Metrics", "ATE"] == expected_result_identity
    assert metrics.loc["Average Metrics", "ATE"] == expected_result_det


def test_scale_error(metrics_identity: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Test that ASE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.033  # 0.2 / 6, one of six dets is off by 20% in IoU
    assert metrics_identity.loc["Average Metrics", "ASE"] == expected_result_identity
    assert metrics.loc["Average Metrics", "ASE"] == expected_result_det


def test_orientation_error(metrics_identity: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Test that AOE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.524  # pi / 6, since one of six dets is off by pi

    assert metrics_identity.loc["Average Metrics", "AOE"] == expected_result_identity
    assert metrics.loc["Average Metrics", "AOE"] == expected_result_det


# def test_remove_duplicate_instances() -> None:
#     """Ensure a duplicate ground truth cuboid can be filtered out correctly."""
#     instances = [
#         SimpleNamespace(**{"translation": np.array([1, 1, 0])}),
#         SimpleNamespace(**{"translation": np.array([5, 5, 0])}),
#         SimpleNamespace(**{"translation": np.array([2, 2, 0])}),
#         SimpleNamespace(**{"translation": np.array([5, 5, 0])}),
#     ]
#     instances = np.array(instances)
#     cfg = DetectionCfg(eval_only_roi_instances=False)
#     unique_instances = remove_duplicate_instances(instances, cfg)

#     assert len(unique_instances) == 3
#     assert np.allclose(unique_instances[0].translation, np.array([1, 1, 0]))
#     assert np.allclose(unique_instances[1].translation, np.array([5, 5, 0]))
#     assert np.allclose(unique_instances[2].translation, np.array([2, 2, 0]))


# def test_remove_duplicate_instances_ground_truth() -> None:
#     """Ensure that if an extra duplicate cuboid is present in ground truth, it would be ignored."""
#     dt_fpath = TEST_DATA_LOC / "remove_duplicates_detections"
#     gt_fpath = TEST_DATA_LOC / "remove_duplicates_ground_truth"
#     fig_fpath = TEST_DATA_LOC / "test_figures"

#     cfg = DetectionCfg(eval_only_roi_instances=False)
#     evaluator = DetectionEvaluator(dt_fpath, gt_fpath, fig_fpath, cfg)
#     metrics = evaluator.evaluate()
#     assert metrics.AP.loc["Vehicle"] == 1.0
#     assert metrics.AP.loc["Pedestrian"] == 1.0


# def test_filter_objs_to_roi() -> None:
#     """Use the map to filter out an object that lies outside the ROI in a parking lot."""
#     avm = ArgoverseMap()

#     # should be outside of ROI
#     outside_obj = {
#         "center": {
#             "x": -14.102872067388489,
#             "y": 19.466695178746022,
#             "z": 0.11740010190455852,
#         },
#         "rotation": {
#             "x": 0.0,
#             "y": 0.0,
#             "z": -0.038991328555453404,
#             "w": 0.9992395490058831,
#         },
#         "length": 4.56126567460171,
#         "width": 1.9370055686754908,
#         "height": 1.5820081349372281,
#         "track_label_uuid": "03a321bf955a4d7781682913884abf06",
#         "timestamp": 315970611820366000,
#         "label_class": "VEHICLE",
#     }

#     # should be inside the ROI
#     inside_obj = {
#         "center": {
#             "x": -20.727430239506702,
#             "y": 3.4488006757501353,
#             "z": 0.4036619561689685,
#         },
#         "rotation": {
#             "x": 0.0,
#             "y": 0.0,
#             "z": 0.0013102003738908123,
#             "w": 0.9999991416871218,
#         },
#         "length": 4.507580779458834,
#         "width": 1.9243189627993598,
#         "height": 1.629934978730058,
#         "track_label_uuid": "bb0f40e4f68043e285d64a839f2f092c",
#         "timestamp": 315970611820366000,
#         "label_class": "VEHICLE",
#     }

#     log_city_name = "PIT"
#     lidar_ts = 315970611820366000
#     dataset_dir = TEST_DATA_LOC / "roi_based_test"
#     log_id = "21e37598-52d4-345c-8ef9-03ae19615d3d"
#     city_SE3_egovehicle = get_city_SE3_egovehicle_at_sensor_t(lidar_ts, dataset_dir, log_id)

#     dts = np.array([json_label_dict_to_obj_record(item) for item in [outside_obj, inside_obj]])
#     dts_filtered = filter_objs_to_roi(dts, avm, city_SE3_egovehicle, log_city_name)

#     assert dts_filtered.size == 1
#     assert dts_filtered.dtype == "O"  # array of objects
#     assert isinstance(dts_filtered, np.ndarray)
#     assert dts_filtered[0].track_id == "bb0f40e4f68043e285d64a839f2f092c"


# def test_AP_on_filtered_instances() -> None:
#     """Test AP calculation on instances filtered on region-of-interest."""
#     dt_fpath = TEST_DATA_LOC / "remove_nonroi_detections"
#     gt_fpath = TEST_DATA_LOC / "remove_nonroi_ground_truth"
#     fig_fpath = TEST_DATA_LOC / "test_figures"

#     cfg = DetectionCfg(eval_only_roi_instances=True)
#     evaluator = DetectionEvaluator(dt_fpath, gt_fpath, fig_fpath, cfg)
#     metrics = evaluator.evaluate()

#     assert metrics.AP.loc["Vehicle"] == 1.0


def test_rank() -> None:
    """Test ranking of detections and scores during detection evaluation."""
    columns = ["track_uuid", "length", "width", "height", "qw", "qx", "qy", "qz", "tx", "ty", "tz", "score"]
    dts = pd.DataFrame(
        [
            ["00000000-0000-0000-0000-000000000000", 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
            ["00000000-0000-0000-0000-000000000001", 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.9],
            ["00000000-0000-0000-0000-000000000002", 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 0.8],
        ],
        columns=columns,
    )

    ranked_dts = rank(dts)
    track_uuids = ranked_dts["track_uuid"]
    expected_track_ids = np.array(
        [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000000",
        ]
    )
    expected_scores = np.array([0.9, 0.8, 0.7])
    assert np.all(track_uuids == expected_track_ids) and np.all(ranked_dts["score"] == expected_scores)
