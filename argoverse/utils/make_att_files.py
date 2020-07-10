import glob
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate

import argoverse
import open3d as o3d
import torch
import torch.nn.functional as F
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

dict_color: Dict[str, Tuple[float, float, float]] = {}
dict_color["easy"] = (0.0, 1.0, 0.0)
dict_color["far"] = (0.0, 0.4, 0.0)
dict_color["occ"] = (0.0, 0.0, 1.0)
dict_color["fast"] = (0.0, 1.0, 1.0)
dict_color["short"] = (0.8, 0.3, 0.3)

list_attritubes = ["short", "occ", "fast", "far", "easy"]

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
    lidar_timestamp: str, 
    pc: np.ndarray,
    egovehicle_to_city_se3: Any,
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

    ind_valid = np.logical_and(
        np.logical_and(pc[:, 0] >= 0, pc[:, 1] >= 0), np.logical_and(pc[:, 0] < image_size, pc[:, 1] < image_size)
    )
    img[pc[ind_valid, 0], pc[ind_valid, 1], :] = 0.4

    path_imgs = os.path.join(path_output_vis, "bev")
    if not os.path.exists(path_imgs):
        os.mkdir(path_imgs)

    for bbox, difficulty_att in zip(list_bboxes, list_difficulty_att):

        q0, q1, q2, q3 = bbox["rotation"]["w"], bbox["rotation"]["x"], bbox["rotation"]["y"], bbox["rotation"]["z"]
        theta_local = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
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
        bbox_2d = np.array([[-l / 2, -w / 2, 0], [l / 2, -w / 2, 0], [-l / 2, w / 2, 0], [l / 2, w / 2, 0]])
        R = np.array(
            [[np.cos(theta_local), -np.sin(theta_local), 0], [np.sin(theta_local), np.cos(theta_local), 0], [0, 0, 1]]
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
            )            cv2.line(img, p1, p2, color=color)

    kernel = np.ones((5, 5), np.float)
    img = cv2.dilate(img, kernel, iterations=1)
    cv2.putText(
        img,
        "%s_%s_%s" % (dataset_name, log_id, lidar_timestamp),
        (100, image_size - 100),
        font,
        fontScale,
        fontColor,
        lineType,
    )

    offset = 0
    for key in dict_color.keys():
        cv2.putText(img, key, (100 + offset, image_size - 50), font, fontScale, dict_color[key], lineType)
        offset += 150

    print("Saving img: ", path_imgs)
    cv2.imwrite(os.path.join(path_imgs, "%s_%s_%s.jpg" % (dataset_name, log_id, lidar_timestamp)), img * 255)



def bspline_1d(x: np.array, y: np.array, s: float = 20.0, k: int = 3) -> np.array:
    """
    Do Bspline smoothing
    """

    if len(x) <= k:
        return y

    tck = interpolate.splrep(x, y[x], s=s, k=k)

    return interpolate.splev(np.arange(y.shape[0]), tck)


def derivative(x: np.array) -> np.array:
    """
    Compute derivative for velocity and acceleration 
    """
    x_tensor = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
    x_padded = torch.cat((x_tensor, (x_tensor[:, :, -1] - x_tensor[:, :, -2] + x_tensor[:, :, -1]).unsqueeze(0)), dim=2)
    filters = torch.Tensor([-1, 1]).unsqueeze(0).unsqueeze(0)

    return F.conv1d(x_padded, filters)[0, 0].numpy()


def compute_v_a(traj: np.array) -> Tuple[np.array, np.array]:  # traj:Nx3
    """
    Compute velocity and acceleration
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

    vx = derivative(dx) * 10  # lidar freq 10 hz
    vy = derivative(dy) * 10
    vz = derivative(dz) * 10

    ax = derivative(vx) * 10
    ay = derivative(vy) * 10
    az = derivative(vz) * 10

    v = np.concatenate((vx[:, np.newaxis], vy[:, np.newaxis], vz[:, np.newaxis]), axis=1)
    a = np.concatenate((ax[:, np.newaxis], ay[:, np.newaxis], az[:, np.newaxis]), axis=1)

    v[np.isnan(traj[:, 0])] = np.nan
    a[np.isnan(traj[:, 0])] = np.nan
    return v, a


def read_json_file(fpath: str) -> None:
    with open(fpath, "rb") as f:
        return json.load(f)


def save_json_file(fpath: str, x: Any) -> None:
    with open(fpath, "w") as f:
        return json.dump(x, f)


# set root_dir to the correct path to your dataset folder
root_dir = "test_set/"
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

        id_log = path_log.split("/")[-1]
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

            data_amodel: Dict[str, Any] = {}
            for key in dict_track_labels.keys():
                data_amodel[key]["label_class"] = dict_track_labels[key][0]["label_class"]
                data_amodel[key]["uuid"] = dict_track_labels[key][0]["track_label_uuid"]
                data_amodel[key]["log_id"] = id_log
                data_amodel[key]["track_label_frames"] = dict_track_labels[key]

        argoverse_loader = ArgoverseTrackingLoader(os.path.join(root_dir, name_folder))
        data_log = argoverse_loader.get(id_log)
        list_lidar_timestamp = data_log.lidar_timestamp_list

        dict_tracks: Dict[str, Any] = {}
        for id_track in data_amodel.keys():

            data = data_amodel[id_track]
            if data["label_class"] not in list_name_class:
                continue

            data_per_frame = data["track_label_frames"]

            dict_tracks[id_track]: Dict[str, Any] = {}
            dict_tracks[id_track]["ind_lidar_min"] = -1
            dict_tracks[id_track]["ind_lidar_max"] = -1
            length_log = len(list_lidar_timestamp)
            dict_tracks[id_track]["list_world_se3"] = [None] * length_log
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

                if dict_tracks[id_track]["ind_lidar_max"] == -1 or ind_lidar > dict_tracks[id_track]["ind_lidar_max"]:
                    dict_tracks[id_track]["ind_lidar_max"] = ind_lidar

                center = np.array([box["center"]["x"], box["center"]["y"], box["center"]["z"]])
                egovehicle_to_city_se3 = argoverse_loader.get_pose(ind_lidar, id_log)
                center_w = egovehicle_to_city_se3.transform_point_cloud(center[np.newaxis, :])[0]
 
                dict_tracks[id_track]["list_center"][ind_lidar] = center
                dict_tracks[id_track]["list_center_w"][ind_lidar] = center_w
                dict_tracks[id_track]["list_dist"][ind_lidar] = np.linalg.norm(center[0:2])
                dict_tracks[id_track]["exists"][ind_lidar] = True
                dict_tracks[id_track]["list_world_se3"][ind_lidar] = egovehicle_to_city_se3
                dict_tracks[id_track]["list_bbox"][ind_lidar] = box

            length_track = dict_tracks[id_track]["ind_lidar_max"] - dict_tracks[id_track]["ind_lidar_min"] + 1
            if dict_tracks[id_track]["ind_lidar_max"] == -1 and dict_tracks[id_track]["ind_lidar_min"] == -1:
                # this shouldn't happen
                dict_tracks[id_track]["length_track"] = 0
            else:
                dict_tracks[id_track]["length_track"] = length_track

            dict_tracks[id_track]["list_vel"], dict_tracks[id_track]["list_acc"] = compute_v_a(
                dict_tracks[id_track]["list_center_w"]
            )
            dict_tracks[id_track]["num_missing"] = (
                dict_tracks[id_track]["length_track"] - dict_tracks[id_track]["exists"].sum()
            )
            dict_tracks[id_track]["difficult_att"] = []
            vel_abs = np.linalg.norm(dict_tracks[id_track]["list_vel"][:, 0:2], axis=1)
            acc_abs = np.linalg.norm(dict_tracks[id_track]["list_acc"][:, 0:2], axis=1)

            dist_close = 50
            ind_valid = np.nonzero(1 - np.isnan(dict_tracks[id_track]["list_dist"]))[0]
            ind_close = np.nonzero(dict_tracks[id_track]["list_dist"][ind_valid] < dist_close)[0]

            if len(ind_close) > 0:
                ind_close_max = ind_close.max() + 1
                ind_close_min = ind_close.min()

            if dict_tracks[id_track]["length_track"] == 0:
                # this shouldn't happen
                dict_tracks[id_track]["difficult_att"].append("zero_length")

            else:
                if dict_tracks[id_track]["list_dist"][ind_valid].min() > dist_close:
                    dict_tracks[id_track]["difficult_att"].append("far")
                else:
                    if dict_tracks[id_track]["length_track"] < 10 or dict_tracks[id_track]["exists"].sum() < 10:
                        dict_tracks[id_track]["difficult_att"].append("short")
                    else:
                        if (ind_close_max - ind_close_min) - dict_tracks[id_track]["exists"][
                            ind_close_min:ind_close_max
                        ].sum() > 20:
                            dict_tracks[id_track]["difficult_att"].append("occ")

                        if np.quantile(vel_abs[ind_valid][ind_close], 0.9) > 1:
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
                egovehicle_to_city_se3 = argoverse_loader.get_pose(ind_lidar, id_log)
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
                    egovehicle_to_city_se3,
                )

        for id_track in dict_tracks.keys():
            list_key = list(dict_tracks[id_track].keys()).copy()
            for key in list_key:
                if key != "difficult_att":
                    del dict_tracks[id_track][key]

        dict_att_all[name_folder][id_log] = dict_tracks

pickle_out = open(filename_output, "wb")
pickle.dump(dict_att_all, pickle_out)
pickle_out.close()