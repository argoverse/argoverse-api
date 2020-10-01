import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.interpolate as interpolate
import torch
import torch.nn.functional as F

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.pkl_utils import save_pkl_dictionary

dict_color: Dict[str, Tuple[float, float, float]] = {}
dict_color["easy"] = (0.0, 1.0, 0.0)  # BGR green
dict_color["far"] = (0.0, 0.4, 0.0)  # BGR dark green
dict_color["occ"] = (0.0, 0.0, 1.0)  # BGR red
dict_color["fast"] = (0.0, 1.0, 1.0)  # BGR yellow
dict_color["short"] = (0.8, 0.3, 0.3)  # BGR dark purple

list_attritubes = ["short", "occ", "fast", "far", "easy"]
LIDAR_FPS = 10  # 10 Hz frequency
MAX_OCCLUSION_PCT = 20
NEAR_DISTANCE_THRESH = 50
SHORT_TRACK_LENGTH_THRESH = 10
SHORT_TRACK_COUNT_THRESH = 10
FAST_TRACK_THRESH = 1


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.2
fontColor = (255, 255, 255)
lineType = 3
visualize = True
check_track_label_folder = True


def save_bev_img(
    path_output_vis: str,
    list_bboxes: List[Any],
    list_difficulty_att: List[Any],
    dataset_name: str,
    log_id: str,
    lidar_timestamp: int,
    pc: np.ndarray,
) -> None:
    """
    Plot results on bev images and save
    """
    image_size = 2000
    image_scale = 10
    img = np.zeros((image_size, image_size, 3))
    pc = pc * image_scale
    pc[:, 0] += int(image_size / 2)
    pc[:, 1] += int(image_size / 2)
    pc = pc.astype("int")

    ind_valid = np.logical_and.reduce([pc[:, 0] >= 0, pc[:, 1] >= 0, pc[:, 0] < image_size, pc[:, 1] < image_size])
    img[pc[ind_valid, 0], pc[ind_valid, 1], :] = 0.4

    path_imgs = os.path.join(path_output_vis, "bev")
    if not os.path.exists(path_imgs):
        os.mkdir(path_imgs)

    for bbox, difficulty_att in zip(list_bboxes, list_difficulty_att):

        qw, qx, qy, qz = (
            bbox["rotation"]["w"],
            bbox["rotation"]["x"],
            bbox["rotation"]["y"],
            bbox["rotation"]["z"],
        )
        theta_local = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        pose_local = np.array([bbox["center"]["x"], bbox["center"]["y"], bbox["center"]["z"]])

        color_0 = 0.0
        color_1 = 0.0
        color_2 = 0.0
        for att in difficulty_att:
            color_0 += dict_color[att][0] / len(difficulty_att)
            color_1 += dict_color[att][1] / len(difficulty_att)
            color_2 += dict_color[att][2] / len(difficulty_att)
        color = (color_0, color_1, color_2)

        w, l, h = bbox["width"], bbox["length"], bbox["height"]
        bbox_2d = np.array(
            [
                [-l / 2, -w / 2, 0],
                [l / 2, -w / 2, 0],
                [-l / 2, w / 2, 0],
                [l / 2, w / 2, 0],
            ]
        )
        R = np.array(
            [
                [np.cos(theta_local), -np.sin(theta_local), 0],
                [np.sin(theta_local), np.cos(theta_local), 0],
                [0, 0, 1],
            ]
        )
        bbox_2d = np.matmul(R, bbox_2d.transpose()).transpose() + pose_local[0:3]
        edge_2d = np.array([[0, 1], [0, 2], [2, 3], [1, 3]])

        for ii in range(len(edge_2d)):
            p1 = (
                int(bbox_2d[edge_2d[ii][0], 1] * image_scale + image_size / 2),
                int(bbox_2d[edge_2d[ii][0], 0] * image_scale + image_size / 2),
            )
            p2 = (
                int(bbox_2d[edge_2d[ii][1], 1] * image_scale + image_size / 2),
                int(bbox_2d[edge_2d[ii][1], 0] * image_scale + image_size / 2),
            )
            cv2.line(img, p1, p2, color=color)

    kernel = np.ones((5, 5), np.float)
    img = cv2.dilate(img, kernel, iterations=1)
    cv2.putText(
        img,
        "%s_%s_%d" % (dataset_name, log_id, lidar_timestamp),
        (100, image_size - 100),
        font,
        fontScale,
        fontColor,
        lineType,
    )

    offset = 0
    for key in dict_color.keys():
        cv2.putText(
            img,
            key,
            (100 + offset, image_size - 50),
            font,
            fontScale,
            dict_color[key],
            lineType,
        )
        offset += 150

    print("Saving img: ", path_imgs)
    cv2.imwrite(
        os.path.join(path_imgs, "%s_%s_%d.jpg" % (dataset_name, log_id, lidar_timestamp)),
        img * 255,
    )


def bspline_1d(x: np.array, y: np.array, s: float = 20.0, k: int = 3) -> np.array:
    """Perform B-Spline smoothing of trajectories for temporal noise reduction

    Args:
        x: N-length np array
        y: N-length np array
        s: smoothing condition
        k: degree of the spline fit

    Returns:
        smoothed trajectory
    """

    if len(x) <= k:
        return y

    tck = interpolate.splrep(x, y[x], s=s, k=k)

    return interpolate.splev(np.arange(y.shape[0]), tck)


def derivative(x: np.array) -> np.array:
    """Compute time derivatives for velocity and acceleration

    Args:
        x: N-length Numpy array, with indices at consecutive timestamps

    Returns:
        dx/dt: N-length Numpy array, with derivative of x w.r.t. timestep
    """
    x_tensor = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
    x_padded = torch.cat(
        (
            x_tensor,
            (x_tensor[:, :, -1] - x_tensor[:, :, -2] + x_tensor[:, :, -1]).unsqueeze(0),
        ),
        dim=2,
    )
    filters = torch.Tensor([-1, 1]).unsqueeze(0).unsqueeze(0)

    return F.conv1d(x_padded, filters)[0, 0].numpy()


def compute_v_a(traj: np.array) -> Tuple[np.array, np.array]:
    """
    Compute velocity and acceleration

    Args:
        traj: Numpy array of shape Nx3 representing 3-d trajectory

    Returns:
        velocity: Numpy array representing (d traj)/dt
        acceleration: Numpy array representing (d velocity)/dt
    """
    ind_valid = np.nonzero(1 - np.isnan(traj[:, 0]))[0]
    dx = bspline_1d(ind_valid, traj[:, 0])
    dy = bspline_1d(ind_valid, traj[:, 1])
    dz = bspline_1d(ind_valid, traj[:, 2])

    if len(ind_valid) > 0:
        x_max = dx[ind_valid].max()
        x_min = dx[ind_valid].min()
        y_max = dy[ind_valid].max()
        y_min = dy[ind_valid].min()

        if np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) < 10:
            return np.zeros_like(traj), np.zeros_like(traj)

    vx = derivative(dx) * LIDAR_FPS  # lidar freq 10 hz
    vy = derivative(dy) * LIDAR_FPS
    vz = derivative(dz) * LIDAR_FPS

    ax = derivative(vx) * LIDAR_FPS
    ay = derivative(vy) * LIDAR_FPS
    az = derivative(vz) * LIDAR_FPS

    v = np.concatenate((vx[:, np.newaxis], vy[:, np.newaxis], vz[:, np.newaxis]), axis=1)
    a = np.concatenate((ax[:, np.newaxis], ay[:, np.newaxis], az[:, np.newaxis]), axis=1)

    v[np.isnan(traj[:, 0])] = np.nan
    a[np.isnan(traj[:, 0])] = np.nan
    return v, a


def make_att_files(root_dir: str) -> None:
    """ Write a .pkl file with difficulty attributes per track """
    path_output_vis = "vis_output"
    filename_output = "att_file.npy"

    if not os.path.exists(path_output_vis):
        os.mkdir(path_output_vis)

    list_folders = ["test"]
    list_name_class = ["VEHICLE", "PEDESTRIAN"]
    count_track = 0
    dict_att_all: Dict[str, Any] = {}

    for name_folder in list_folders:

        dict_att_all[name_folder] = {}
        list_log_folders = glob.glob(os.path.join(root_dir, name_folder, "*"))
        for ind_log, path_log in enumerate(list_log_folders):

            id_log = f"{Path(path_log).name}"
            print("%s %s %d/%d" % (name_folder, id_log, ind_log, len(list_log_folders)))

            if check_track_label_folder:
                list_path_label_persweep = glob.glob(os.path.join(path_log, "per_sweep_annotations_amodal", "*"))
                list_path_label_persweep.sort()

                dict_track_labels: Dict[str, Any] = {}
                for path_label_persweep in list_path_label_persweep:
                    data = read_json_file(path_label_persweep)
                    for data_obj in data:
                        id_obj = data_obj["track_label_uuid"]

                        if id_obj not in dict_track_labels.keys():
                            dict_track_labels[id_obj] = []
                        dict_track_labels[id_obj].append(data_obj)

                data_amodal: Dict[str, Any] = {}
                for key in dict_track_labels.keys():
                    dict_amodal: Dict[str, Any] = {}
                    data_amodal[key] = dict_amodal
                    data_amodal[key]["label_class"] = dict_track_labels[key][0]["label_class"]
                    data_amodal[key]["uuid"] = dict_track_labels[key][0]["track_label_uuid"]
                    data_amodal[key]["log_id"] = id_log
                    data_amodal[key]["track_label_frames"] = dict_track_labels[key]

            argoverse_loader = ArgoverseTrackingLoader(os.path.join(root_dir, name_folder))
            data_log = argoverse_loader.get(id_log)
            list_lidar_timestamp = data_log.lidar_timestamp_list

            dict_tracks: Dict[str, Any] = {}
            for id_track in data_amodal.keys():

                data = data_amodal[id_track]
                if data["label_class"] not in list_name_class:
                    continue

                data_per_frame = data["track_label_frames"]

                dict_per_track: Dict[str, Any] = {}
                dict_tracks[id_track] = dict_per_track
                dict_tracks[id_track]["ind_lidar_min"] = -1
                dict_tracks[id_track]["ind_lidar_max"] = -1
                length_log = len(list_lidar_timestamp)
                dict_tracks[id_track]["list_city_se3"] = [None] * length_log
                dict_tracks[id_track]["list_bbox"] = [None] * length_log
                count_track += 1

                dict_tracks[id_track]["list_center"] = np.full([length_log, 3], np.nan)
                dict_tracks[id_track]["list_center_w"] = np.full([length_log, 3], np.nan)
                dict_tracks[id_track]["list_dist"] = np.full([length_log], np.nan)
                dict_tracks[id_track]["exists"] = np.full([length_log], False)

                for box in data_per_frame:

                    if box["timestamp"] in list_lidar_timestamp:
                        ind_lidar = list_lidar_timestamp.index(box["timestamp"])
                    else:
                        continue

                    if dict_tracks[id_track]["ind_lidar_min"] == -1:
                        dict_tracks[id_track]["ind_lidar_min"] = ind_lidar

                    dict_tracks[id_track]["ind_lidar_max"] = max(ind_lidar, dict_tracks[id_track]["ind_lidar_max"])

                    center = np.array([box["center"]["x"], box["center"]["y"], box["center"]["z"]])
                    city_SE3_egovehicle = argoverse_loader.get_pose(ind_lidar, id_log)
                    if city_SE3_egovehicle is None:
                        print("Pose not found!")
                        continue
                    center_w = city_SE3_egovehicle.transform_point_cloud(center[np.newaxis, :])[0]

                    dict_tracks[id_track]["list_center"][ind_lidar] = center
                    dict_tracks[id_track]["list_center_w"][ind_lidar] = center_w
                    dict_tracks[id_track]["list_dist"][ind_lidar] = np.linalg.norm(center[0:2])
                    dict_tracks[id_track]["exists"][ind_lidar] = True
                    dict_tracks[id_track]["list_city_se3"][ind_lidar] = city_SE3_egovehicle
                    dict_tracks[id_track]["list_bbox"][ind_lidar] = box

                length_track = dict_tracks[id_track]["ind_lidar_max"] - dict_tracks[id_track]["ind_lidar_min"] + 1

                assert not (
                    dict_tracks[id_track]["ind_lidar_max"] == -1 and dict_tracks[id_track]["ind_lidar_min"] == -1
                ), "zero-length track"
                dict_tracks[id_track]["length_track"] = length_track

                (
                    dict_tracks[id_track]["list_vel"],
                    dict_tracks[id_track]["list_acc"],
                ) = compute_v_a(dict_tracks[id_track]["list_center_w"])
                dict_tracks[id_track]["num_missing"] = (
                    dict_tracks[id_track]["length_track"] - dict_tracks[id_track]["exists"].sum()
                )
                dict_tracks[id_track]["difficult_att"] = []
                # get scalar velocity per timestamp as 2-norm of (vx, vy)
                vel_abs = np.linalg.norm(dict_tracks[id_track]["list_vel"][:, 0:2], axis=1)
                acc_abs = np.linalg.norm(dict_tracks[id_track]["list_acc"][:, 0:2], axis=1)

                ind_valid = np.nonzero(1 - np.isnan(dict_tracks[id_track]["list_dist"]))[0]
                ind_close = np.nonzero(dict_tracks[id_track]["list_dist"][ind_valid] < NEAR_DISTANCE_THRESH)[0]

                if len(ind_close) > 0:
                    ind_close_max = ind_close.max() + 1
                    ind_close_min = ind_close.min()

                # Only compute "fast" and "occluded" tags for near objects
                # The thresholds are not very meaningful for faraway objects, since they are usually pretty short.
                if dict_tracks[id_track]["list_dist"][ind_valid].min() > NEAR_DISTANCE_THRESH:
                    dict_tracks[id_track]["difficult_att"].append("far")
                else:
                    is_short_len_track1 = dict_tracks[id_track]["length_track"] < SHORT_TRACK_LENGTH_THRESH
                    is_short_len_track2 = dict_tracks[id_track]["exists"].sum() < SHORT_TRACK_COUNT_THRESH
                    if is_short_len_track1 or is_short_len_track2:
                        dict_tracks[id_track]["difficult_att"].append("short")
                    else:
                        if (ind_close_max - ind_close_min) - dict_tracks[id_track]["exists"][
                            ind_close_min:ind_close_max
                        ].sum() > MAX_OCCLUSION_PCT:
                            dict_tracks[id_track]["difficult_att"].append("occ")

                        if np.quantile(vel_abs[ind_valid][ind_close], 0.9) > FAST_TRACK_THRESH:
                            dict_tracks[id_track]["difficult_att"].append("fast")

                if len(dict_tracks[id_track]["difficult_att"]) == 0:
                    dict_tracks[id_track]["difficult_att"].append("easy")

            if visualize:
                for ind_lidar, timestamp_lidar in enumerate(list_lidar_timestamp):

                    list_bboxes = []
                    list_difficulty_att = []

                    for id_track in dict_tracks.keys():
                        if dict_tracks[id_track]["exists"][ind_lidar]:
                            list_bboxes.append(dict_tracks[id_track]["list_bbox"][ind_lidar])
                            list_difficulty_att.append(dict_tracks[id_track]["difficult_att"])

                    path_lidar = os.path.join(path_log, "lidar", "PC_%s.ply" % timestamp_lidar)
                    pc = np.asarray(o3d.io.read_point_cloud(path_lidar).points)
                    list_lidar_timestamp = data_log.lidar_timestamp_list
                    save_bev_img(
                        path_output_vis,
                        list_bboxes,
                        list_difficulty_att,
                        "argoverse_%s" % name_folder,
                        id_log,
                        timestamp_lidar,
                        pc,
                    )

            for id_track in dict_tracks.keys():
                list_key = list(dict_tracks[id_track].keys()).copy()
                for key in list_key:
                    if key != "difficult_att":
                        del dict_tracks[id_track][key]

            dict_att_all[name_folder][id_log] = dict_tracks

    save_pkl_dictionary(filename_output, dict_att_all)


if __name__ == "__main__":
    # set root_dir to the correct path to your dataset folder
    root_dir = "test_set/"
    make_att_files(root_dir)
