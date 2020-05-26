# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.camera_stats import CAMERA_LIST
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.utils.subprocess_utils import run_command
from argoverse.utils.transform import quat2rotmat
from argoverse.visualization.ground_visualization import draw_ground_pts_in_image


def visualize_ground_lidar_pts(log_id: str, dataset_dir: str, experiment_prefix: str):
    """Process a log by drawing the LiDAR returns that are classified as belonging
    to the ground surface in a red to green colormap in the image.

    Args:
        log_id: The ID of a log
        dataset_dir: Where the dataset is stored
        experiment_prefix: Output prefix
    """
    sdb = SynchronizationDB(dataset_dir, collect_single_log_id=log_id)

    city_info_fpath = f"{dataset_dir}/{log_id}/city_info.json"
    city_info = read_json_file(city_info_fpath)
    city_name = city_info["city_name"]
    avm = ArgoverseMap()

    ply_fpaths = sorted(glob.glob(f"{dataset_dir}/{log_id}/lidar/PC_*.ply"))

    for i, ply_fpath in enumerate(ply_fpaths):
        if i % 500 == 0:
            print(f"\tOn file {i} of {log_id}")
        lidar_timestamp_ns = ply_fpath.split("/")[-1].split(".")[0].split("_")[-1]

        pose_fpath = f"{dataset_dir}/{log_id}/poses/city_SE3_egovehicle_{lidar_timestamp_ns}.json"
        if not Path(pose_fpath).exists():
            continue

        pose_data = read_json_file(pose_fpath)
        rotation = np.array(pose_data["rotation"])
        translation = np.array(pose_data["translation"])
        city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(rotation), translation=translation)

        lidar_pts = load_ply(ply_fpath)

        lidar_timestamp_ns = int(lidar_timestamp_ns)
        draw_ground_pts_in_image(
            sdb,
            lidar_pts,
            city_to_egovehicle_se3,
            avm,
            log_id,
            lidar_timestamp_ns,
            city_name,
            dataset_dir,
            experiment_prefix,
        )

    for camera_name in CAMERA_LIST:
        if "stereo" in camera_name:
            fps = 5
        else:
            fps = 10
        cmd = (
            f"ffmpeg -r {fps} -f image2 -i '{experiment_prefix}_ground_viz/{log_id}/{camera_name}/%*.jpg'"
            + f" {experiment_prefix}_ground_viz/{experiment_prefix}_{log_id}_{camera_name}_{fps}fps.mp4"
        )

        print(cmd)
        run_command(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help="Path to where the logs are stored")
    parser.add_argument(
        "--log-ids",
        type=str,
        help="Comma separated list of log ids, as found in {args.dataset-dir}/argoverse-tracking/sample/[log_id]",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="ground_lidar_viz",
        type=str,
        help="Results will be saved in a folder with this prefix for its name",
    )
    arguments = parser.parse_args()

    print("Start visualizing Argoverse-Tracking3D ground.")

    for log_id in arguments.log_ids.split(","):
        visualize_ground_lidar_pts(log_id.strip(), arguments.dataset_dir, arguments.experiment_prefix)
