#!/usr/bin/python3

import os
import shutil
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from argoverse.evaluation.eval_tracking import eval_tracks, get_orientation_error_deg
from argoverse.utils.json_utils import save_json_dict

_ROOT = Path(__file__).resolve().parent

"""
FRAG: the number of track fragmentations (FM) counts how many times a 
ground truth trajectory is interrupted (untracked). In other words, a
fragmentation is counted each time a trajectory changes its status
from tracked to untracked and tracking of that same trajectory is
resumed at a later point.
(ref: Milan et al., MOT16, https://arxiv.org/pdf/1603.00831.pdf)

IDSW: an identity switch, is counted if a ground truth target i
is matched to track j and the last known assignment was k != j
(ref: Milan et al., MOT16, https://arxiv.org/pdf/1603.00831.pdf)

MT: a target is mostly tracked if it is successfully tracked
for at least 80% of its life span. Note that it is irrelevant
for this measure whether the ID remains the same throughout the track.
(ref: Leal-Taixe et al., MOT15, https://arxiv.org/pdf/1504.01942.pdf)

Note: IDF1 is not the same as F1 score. It uses the number of false
negatives matches after global min-cost matching.
(https://arxiv.org/pdf/1609.01775.pdf)
"""


def check_mkdir(dirpath: str) -> None:
    """ """
    if not Path(dirpath).exists():
        os.makedirs(dirpath, exist_ok=True)


def yaw_to_quaternion3d(yaw: float) -> Tuple[float, float, float, float]:
    """
		Args:
		-   yaw: rotation about the z-axis, in radians
		Returns:
		-   qx,qy,qz,qw: quaternion coefficients
	"""
    qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
    return qx, qy, qz, qw


class TrackedObjRec(NamedTuple):
    l: float
    w: float
    h: float
    qx: float
    qy: float
    qz: float
    qw: float
    cx: float
    cy: float
    cz: float
    track_id: str
    label_class: str


class TrackedObjects:
    def __init__(self, log_id: str, is_gt: bool) -> None:
        """ """
        self.ts_to_trackedlabels_dict = defaultdict(list)
        self.log_id = log_id

        tracks_type = "gt" if is_gt else "pred"
        self.log_dir = f"{_ROOT}/test_data/"
        self.log_dir += f"eval_tracking_dummy_logs_{tracks_type}/{self.log_id}"

    def add_obj(self, o: TrackedObjRec, ts_ns: int) -> None:
        """
			Args:
			-	ts_ns: timestamp in nanoseconds
		"""
        self.ts_to_trackedlabels_dict[ts_ns] += [
            {
                "center": {"x": o.cx, "y": o.cy, "z": o.cz},
                "rotation": {"x": o.qx, "y": o.qy, "z": o.qz, "w": o.qw},
                "length": o.l,
                "width": o.w,
                "height": o.h,
                "track_label_uuid": o.track_id,
                "timestamp": ts_ns,  # 1522688014970187
                "label_class": o.label_class,
            }
        ]

    def save_to_disk(self) -> None:
        """
		Labels and predictions should be saved in JSON e.g.
			`tracked_object_labels_315969629019741000.json`
		"""
        for ts_ns, ts_trackedlabels in self.ts_to_trackedlabels_dict.items():
            json_fpath = f"{self.log_dir}/per_sweep_annotations_amodal/"
            check_mkdir(json_fpath)
            json_fpath += f"tracked_object_labels_{ts_ns}.json"
            save_json_dict(json_fpath, ts_trackedlabels)


def dump_1obj_scenario_json(centers, yaw_angles, log_id: str, is_gt: bool) -> None:
    """
	Egovehicle stationary (represented by `o`).
	Sequence of 4-nanosecond timestamps.
	"""
    t_objs = TrackedObjects(log_id=log_id, is_gt=is_gt)

    l = 2
    w = 2
    h = 1
    track_id = "obj_a"
    label_class = "VEHICLE"

    for ts_ns, (center, yaw_angle) in enumerate(zip(centers, yaw_angles)):
        cx, cy, cz = center
        qx, qy, qz, qw = yaw_to_quaternion3d(yaw=yaw_angle)
        tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
        t_objs.add_obj(tor, ts_ns=ts_ns)

    t_objs.save_to_disk()


def run_eval(exp_name: str) -> Mapping[str, Any]:
    """ """
    pred_log_dir = f"{_ROOT}/test_data/eval_tracking_dummy_logs_pred"
    gt_log_dir = f"{_ROOT}/test_data/eval_tracking_dummy_logs_gt"

    out_fpath = f"{_ROOT}/test_data/{exp_name}.txt"
    out_file = open(out_fpath, "w")
    eval_tracks(
        path_tracker_output_root=pred_log_dir,
        path_dataset_root=gt_log_dir,
        d_min=0,
        d_max=100,
        out_file=out_file,
        centroid_method="average",
        diffatt=None,
        category="VEHICLE",
    )
    out_file.close()

    with open(out_fpath, "r") as f:
        result_lines = f.readlines()
        result_vals = result_lines[0].strip().split(" ")

        fn, num_frames, mota, motp_c, motp_o, motp_i, idf1 = result_vals[:7]
        most_track, most_lost, num_fp, num_miss, num_sw, num_frag = result_vals[7:]

        result_dict = {
            "filename": fn,
            "num_frames": int(num_frames),
            "mota": float(mota),
            "motp_c": float(motp_c),
            "motp_o": float(motp_o),
            "motp_i": float(motp_i),
            "idf1": float(idf1),
            "most_track": float(most_track),
            "most_lost": float(most_lost),
            "num_fp": int(num_fp),
            "num_miss": int(num_miss),
            "num_sw": int(num_sw),
            "num_frag": int(num_frag),
        }
    shutil.rmtree(pred_log_dir)
    shutil.rmtree(gt_log_dir)
    return result_dict


def get_1obj_gt_scenario():
    """
	Egovehicle stationary (represented by `o`).
	Seqeuence of 4-nanosecond timestamps.

	|-|
	| |
	|-|

	|-|
	| |
	|-|
			o (x,y,z) = (0,0,0)
	|-|
	| |
	|-|

	|-|
	| | (x,y,z)=(-3,2,0)
	|-|
	"""
    centers = []
    # timestamp 0
    cx = -3
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 1
    cx = -1
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 2
    cx = 1
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 3
    cx = 3
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    yaw_angles = [0, 0, 0, 0]
    return centers, yaw_angles


def test_1obj_perfect() -> None:
    """ """
    log_id = "1obj_perfect"
    gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

    centers = gt_centers
    yaw_angles = gt_yaw_angles

    # dump the ground truth first
    dump_1obj_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True)
    dump_1obj_scenario_json(centers, yaw_angles, log_id, is_gt=False)
    result_dict = run_eval(exp_name=log_id)

    assert result_dict["num_frames"] == 4
    assert result_dict["mota"] == 100.0
    assert result_dict["motp_c"] == 0.0
    assert result_dict["motp_o"] == 0.0
    assert result_dict["motp_i"] == 0.0
    assert result_dict["idf1"] == 1.0
    assert result_dict["most_track"] == 1.0
    assert result_dict["most_lost"] == 0.0
    assert result_dict["num_fp"] == 0
    assert result_dict["num_miss"] == 0
    assert result_dict["num_sw"] == 0
    assert result_dict["num_frag"] == 0


def test_1obj_offset_translation() -> None:
    """ """
    log_id = "1obj_offset_translation"

    centers = []

    # timestamp 0
    cx = -4
    cy = 3
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 1
    cx = -2
    cy = 3
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 2
    cx = 0
    cy = 3
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 3
    cx = 2
    cy = 3
    cz = 0
    centers += [(cx, cy, cz)]

    yaw_angles = [0, 0, 0, 0]

    # dump the ground truth first
    gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

    # dump the ground truth first
    dump_1obj_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True)
    dump_1obj_scenario_json(centers, yaw_angles, log_id, is_gt=False)
    result_dict = run_eval(exp_name=log_id)

    assert result_dict["num_frames"] == 4
    assert result_dict["mota"] == 100.0
    # Centroids will be (1,1) away from true centroid each time
    assert np.allclose(result_dict["motp_c"], np.sqrt(2), atol=0.01)
    assert result_dict["motp_o"] == 0.0
    assert result_dict["motp_i"] == 0.0
    assert result_dict["idf1"] == 1.0
    assert result_dict["most_track"] == 1.0
    assert result_dict["most_lost"] == 0.0
    assert result_dict["num_fp"] == 0
    assert result_dict["num_miss"] == 0
    assert result_dict["num_sw"] == 0
    assert result_dict["num_frag"] == 0


def test_1obj_poor_translation() -> None:
    """
	Miss in 1st frame, TP in 2nd frame,
	lost in 3rd frame, retrack as TP in 4th frame

	Yields 1 fragmentation. Prec=0.5, recall=0.5, F1=0.5

	mostly tracked if it is successfully tracked
	for at least 80% of its life span

	If a track is only recovered for less than 20% of its
	total length, it is said to be mostly lost (ML)
	"""
    log_id = "1obj_poor_translation"

    centers = []

    # timestamp 0
    cx = -5
    cy = 4
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 1
    cx = -2
    cy = 3
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 2
    cx = 1
    cy = 4
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 3
    cx = 4
    cy = 3
    cz = 0
    centers += [(cx, cy, cz)]

    yaw_angles = [0, 0, 0, 0]

    # dump the ground truth first
    gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

    # dump the ground truth first
    dump_1obj_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True)
    dump_1obj_scenario_json(centers, yaw_angles, log_id, is_gt=False)
    result_dict = run_eval(exp_name=log_id)

    assert result_dict["num_frames"] == 4
    sw = 0
    mota = 1 - ((2 + 2 + 0) / 4)  # 1 - (FN+FP+SW)/#GT
    assert mota == 0.0
    assert result_dict["mota"] == 0.0
    assert np.allclose(result_dict["motp_c"], np.sqrt(2), atol=0.01)  # (1,1) away each time
    assert result_dict["motp_o"] == 0.0
    assert result_dict["motp_i"] == 0.0
    prec = 0.5
    recall = 0.5
    f1 = 2 * prec * recall / (prec + recall)
    assert f1 == 0.5
    assert result_dict["idf1"] == 0.5
    assert result_dict["most_track"] == 0.0
    assert result_dict["most_lost"] == 0.0
    assert result_dict["num_fp"] == 2
    assert result_dict["num_miss"] == 2  # false-negatives
    assert result_dict["num_sw"] == 0
    assert result_dict["num_frag"] == 1


def test_1obj_poor_orientation() -> None:
    """ """
    log_id = "1obj_poor_orientation"

    centers = []
    # timestamp 0
    cx = -3
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 1
    cx = -1
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 2
    cx = 1
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 3
    cx = 3
    cy = 2
    cz = 0
    centers += [(cx, cy, cz)]

    yaw_angles = [0.25, -0.25, 0.25, -0.25]

    # dump the ground truth first
    gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

    # dump the ground truth first
    dump_1obj_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True)
    dump_1obj_scenario_json(centers, yaw_angles, log_id, is_gt=False)
    result_dict = run_eval(exp_name=log_id)

    assert result_dict["num_frames"] == 4
    assert result_dict["mota"] == 100.0
    assert result_dict["motp_c"] == 0
    assert np.allclose(result_dict["motp_o"], 14.32, atol=0.01)
    assert result_dict["motp_i"] == 0.0
    assert result_dict["idf1"] == 1.0
    assert result_dict["most_track"] == 1.0
    assert result_dict["most_lost"] == 0.0
    assert result_dict["num_fp"] == 0
    assert result_dict["num_miss"] == 0
    assert result_dict["num_sw"] == 0
    assert result_dict["num_frag"] == 0


def test_orientation_error1() -> None:
    """ """
    yaw1 = np.deg2rad(179)
    yaw2 = np.deg2rad(-179)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    assert np.allclose(error_deg, 2.0, atol=1e-2)


def test_orientation_error2() -> None:
    """ """
    yaw1 = np.deg2rad(-179)
    yaw2 = np.deg2rad(179)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    print(error_deg)
    assert np.allclose(error_deg, 2.0, atol=1e-2)


def test_orientation_error3() -> None:
    """ """
    yaw1 = np.deg2rad(179)
    yaw2 = np.deg2rad(178)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    assert np.allclose(error_deg, 1.0, atol=1e-2)


def test_orientation_error4() -> None:
    """ """
    yaw1 = np.deg2rad(178)
    yaw2 = np.deg2rad(179)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    assert np.allclose(error_deg, 1.0, atol=1e-2)


def test_orientation_error5() -> None:
    """ """
    yaw1 = np.deg2rad(3)
    yaw2 = np.deg2rad(-3)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    assert np.allclose(error_deg, 6.0, atol=1e-2)


def test_orientation_error6() -> None:
    """ """
    yaw1 = np.deg2rad(-3)
    yaw2 = np.deg2rad(3)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    assert np.allclose(error_deg, 6.0, atol=1e-2)


def test_orientation_error7() -> None:
    """ """
    yaw1 = np.deg2rad(-177)
    yaw2 = np.deg2rad(-179)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    assert np.allclose(error_deg, 2.0, atol=1e-2)


def test_orientation_error8() -> None:
    """ """
    yaw1 = np.deg2rad(-179)
    yaw2 = np.deg2rad(-177)

    error_deg = get_orientation_error_deg(yaw1, yaw2)
    assert np.allclose(error_deg, 2.0, atol=1e-2)


def get_mot16_scenario_a():
    """
	https://arxiv.org/pdf/1603.00831.pdf
	"""
    centers = []
    # timestamp 0
    cx = 0
    cy = -1
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 1
    cx = 2
    cy = 1
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 2
    cx = 4
    cy = 1
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 3
    cx = 6
    cy = 0
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 4
    cx = 8
    cy = -1
    cz = 0
    centers += [(cx, cy, cz)]

    # timestamp 5
    cx = 10
    cy = 0
    cz = 0
    centers += [(cx, cy, cz)]

    yaw_angles = [0, 0, 0, 0, 0, 0]
    return centers, yaw_angles


def test_mot16_scenario_a() -> None:
    """
	See page 8 of MOT16 paper: https://arxiv.org/pdf/1603.00831.pdf
	"""
    log_id = "mot16_scenario_a"
    gt_centers, gt_yaw_angles = get_mot16_scenario_a()
    dump_1obj_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True)

    t_objs = TrackedObjects(log_id=log_id, is_gt=False)

    l = 2
    w = 2
    h = 1

    label_class = "VEHICLE"

    # ----------- Red track --------------------------------------------
    track_id = "red_obj"
    cx, cy, cz = (0, -3, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=0)

    cx, cy, cz = (2, 0, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=1)

    cx, cy, cz = (4, 0, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=2)

    cx, cy, cz = (6, 1, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=3)

    cx, cy, cz = (8, 3, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=4)

    # ----------- Blue track -------------------------------------------
    track_id = "blue_obj"
    cx, cy, cz = (4, -4, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=2)

    cx, cy, cz = (6, -2, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=3)

    cx, cy, cz = (8, 0, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=4)

    cx, cy, cz = (10, 1, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=5)

    t_objs.save_to_disk()
    result_dict = run_eval(exp_name=log_id)

    assert result_dict["num_frames"] == 6
    assert result_dict["mota"] == 0.0  # 1 - (4+1+1)/6 = 0
    assert result_dict["motp_c"] == 1  # off by 1 meter at every frame
    assert result_dict["motp_o"] == 0.0
    assert result_dict["motp_i"] == 0.0  # using same-sized box for GT and predictions
    assert result_dict["most_track"] == 1.0  # GT obj is tracked for 80% of lifetime
    assert result_dict["most_lost"] == 0.0
    assert result_dict["num_fp"] == 4
    assert result_dict["num_miss"] == 1  # just 1 false negative
    assert result_dict["num_sw"] == 1  # switch from red to blue
    assert result_dict["num_frag"] == 0


def test_mot16_scenario_b() -> None:
    """
	See page 8 of MOT16 paper: https://arxiv.org/pdf/1603.00831.pdf
	Scenario `a` and Scenario `b` share the same ground truth.
	"""
    log_id = "mot16_scenario_b"
    gt_centers, gt_yaw_angles = get_mot16_scenario_a()
    dump_1obj_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True)

    t_objs = TrackedObjects(log_id=log_id, is_gt=False)

    l = 2
    w = 2
    h = 1

    label_class = "VEHICLE"

    # ----------- Red track --------------------------------------------
    track_id = "red_obj"
    cx, cy, cz = (0, -0.5, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=0)

    cx, cy, cz = (2, 0, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=1)

    cx, cy, cz = (4, 3, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=2)

    # ----------- Blue track -------------------------------------------
    track_id = "blue_obj"
    cx, cy, cz = (6, -2, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=3)

    cx, cy, cz = (8, -1, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=4)

    cx, cy, cz = (10, 1, 0)
    qx, qy, qz, qw = yaw_to_quaternion3d(yaw=0)
    tor = TrackedObjRec(l, w, h, qx, qy, qz, qw, cx, cy, cz, track_id, label_class)
    t_objs.add_obj(tor, ts_ns=5)

    t_objs.save_to_disk()

    result_dict = run_eval(exp_name=log_id)
    assert result_dict["num_frames"] == 6
    assert result_dict["mota"] == 16.67  # 1 - (2+2+1)/6 = 0.1667
    assert result_dict["motp_c"] == 0.62  # off by [0.5,1,0,1] -> 0.625 truncated
    assert result_dict["motp_o"] == 0.0
    assert result_dict["motp_i"] == 0.0  # using same-sized box for GT and predictions
    assert result_dict["most_track"] == 0.0  # GT obj is tracked for only 67% of lifetime
    assert result_dict["most_lost"] == 0.0
    assert result_dict["num_fp"] == 2
    assert result_dict["num_miss"] == 2  # 2 false negatives
    assert result_dict["num_sw"] == 1  # switch from red to blue
    assert result_dict["num_frag"] == 1  # 1 frag, since tracked->untracked->tracked


"""
try 2 tracks
then try 2 logs
"""

if __name__ == "__main__":
    """ """
    test_1obj_perfect()
    test_1obj_offset_translation()
    test_1obj_poor_translation()
    test_1obj_poor_orientation()
    test_mot16_scenario_a()
    test_mot16_scenario_b()

    test_orientation_error1()
    test_orientation_error2()
    test_orientation_error3()
    test_orientation_error4()
    test_orientation_error5()
    test_orientation_error6()
    test_orientation_error7()
    test_orientation_error8()
