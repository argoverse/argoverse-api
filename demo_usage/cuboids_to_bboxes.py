# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import argparse
import copy
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, Union

import cv2
import imageio
import numpy as np
from typing_extensions import Final

from argoverse.data_loading.object_label_record import json_label_dict_to_obj_record
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.sensor_dataset_config import ArgoverseConfig
from argoverse.utils.calibration import (
    CameraConfig,
    get_calibration_config,
    point_cloud_to_homogeneous,
    project_lidar_to_img_motion_compensated,
    project_lidar_to_undistorted_img,
)
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.city_visibility_utils import clip_point_cloud_to_visible_region
from argoverse.utils.cv2_plotting_utils import draw_clipped_line_segment
from argoverse.utils.cv2_video_utils import VideoWriter
from argoverse.utils.ffmpeg_utils import ffmpeg_compress_video, write_nonsequential_idx_video
from argoverse.utils.frustum_clipping import generate_frustum_planes
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

#: Any numeric type
Number = Union[int, float]

# jigger lane pixel values by [-10,10] range
LANE_COLOR_NOISE: Final = 20
STEREO_FPS: Final = ArgoverseConfig.stereo_cam_fps
RING_CAM_FPS: Final = ArgoverseConfig.ring_cam_fps


def plot_lane_centerlines_in_img(
    lidar_pts: np.ndarray,
    city_SE3_egovehicle: SE3,
    img: np.ndarray,
    city_name: str,
    avm: ArgoverseMap,
    camera_config: CameraConfig,
    planes: Iterable[Tuple[np.array, np.array, np.array, np.array, np.array]],
    color: Tuple[int, int, int] = (0, 255, 255),
    linewidth: Number = 10,
) -> np.ndarray:
    """
    Args:
        city_SE3_egovehicle: SE(3) transformation representing egovehicle to city transformation
        img: Array of shape (M,N,3) representing updated image
        city_name: str, string representing city name, i.e. 'PIT' or 'MIA'
        avm: instance of ArgoverseMap
        camera_config: instance of CameraConfig
        planes: five frustum clipping planes
        color: RGB-tuple representing color
        linewidth: Number = 10) -> np.ndarray

    Returns:
        img: Array of shape (M,N,3) representing updated image
    """
    R = camera_config.extrinsic[:3, :3]
    t = camera_config.extrinsic[:3, 3]
    cam_SE3_egovehicle = SE3(rotation=R, translation=t)

    query_x, query_y, _ = city_SE3_egovehicle.translation
    local_centerlines = avm.find_local_lane_centerlines(query_x, query_y, city_name)

    for centerline_city_fr in local_centerlines:
        color = [intensity + np.random.randint(0, LANE_COLOR_NOISE) - LANE_COLOR_NOISE // 2 for intensity in color]

        ground_heights = avm.get_ground_height_at_xy(centerline_city_fr, city_name)

        valid_idx = np.isnan(ground_heights)
        centerline_city_fr = centerline_city_fr[~valid_idx]

        centerline_egovehicle_fr = city_SE3_egovehicle.inverse().transform_point_cloud(centerline_city_fr)
        centerline_uv_cam = cam_SE3_egovehicle.transform_point_cloud(centerline_egovehicle_fr)

        # can also clip point cloud to nearest LiDAR point depth
        centerline_uv_cam = clip_point_cloud_to_visible_region(centerline_uv_cam, lidar_pts)
        for i in range(centerline_uv_cam.shape[0] - 1):
            draw_clipped_line_segment(
                img,
                centerline_uv_cam[i],
                centerline_uv_cam[i + 1],
                camera_config,
                linewidth,
                planes,
                color,
            )
    return img


def dump_clipped_3d_cuboids_to_images(
    log_ids: Sequence[str],
    max_num_images_to_render: int,
    data_dir: str,
    experiment_prefix: str,
    motion_compensate: bool = True,
    omit_centerlines: bool = False,
    generate_video_only: bool = False,
) -> List[str]:
    """
    We bring the 3D points into each camera coordinate system, and do the clipping there in 3D.

    Args:
        log_ids: A list of log IDs
        max_num_images_to_render: maximum numbers of images to render.
        data_dir: path to dataset with the latest data
        experiment_prefix: Output directory
        motion_compensate: Whether to motion compensate when projecting
        omit_centerlines: whether to omit map vector lane geometry from rendering
        generate_video_only: whether to generate mp4s only without dumping individual frames

    Returns:
        saved_img_fpaths
    """
    saved_img_fpaths = []
    dl = SimpleArgoverseTrackingDataLoader(data_dir=data_dir, labels_dir=data_dir)
    if not omit_centerlines:
        avm = ArgoverseMap()
    fps_map = {
        cam_name: STEREO_FPS if "stereo" in cam_name else RING_CAM_FPS
        for cam_name in RING_CAMERA_LIST + STEREO_CAMERA_LIST
    }
    category_subdir = "amodal_labels"
    if not Path(f"{experiment_prefix}_{category_subdir}").exists():
        os.makedirs(f"{experiment_prefix}_{category_subdir}")
    video_output_dir = f"{experiment_prefix}_{category_subdir}"

    for log_id in log_ids:
        save_dir = f"{experiment_prefix}_{log_id}"
        if not generate_video_only and not Path(save_dir).exists():
            # JPG images will be dumped here, if requested by arguments
            os.makedirs(save_dir)

        city_name = dl.get_city_name(log_id)
        log_calib_data = dl.get_log_calibration_data(log_id)

        flag_done = False
        for cam_idx, camera_name in enumerate(RING_CAMERA_LIST + STEREO_CAMERA_LIST):
            fps = fps_map[camera_name]
            if generate_video_only:
                mp4_path = f"{video_output_dir}/{log_id}_{camera_name}_{fps}fps.mp4"
                video_writer = VideoWriter(mp4_path)
            cam_im_fpaths = dl.get_ordered_log_cam_fpaths(log_id, camera_name)
            for i, im_fpath in enumerate(cam_im_fpaths):
                if i % 50 == 0:
                    logging.info("\tOn file %s of camera %s of %s", i, camera_name, log_id)

                cam_timestamp = Path(im_fpath).stem.split("_")[-1]
                cam_timestamp = int(cam_timestamp)

                # load PLY file path, e.g. 'PC_315978406032859416.ply'
                ply_fpath = dl.get_closest_lidar_fpath(log_id, cam_timestamp)
                if ply_fpath is None:
                    continue
                lidar_pts = load_ply(ply_fpath)
                save_img_fpath = f"{save_dir}/{camera_name}_{cam_timestamp}.jpg"
                if Path(save_img_fpath).exists():
                    saved_img_fpaths += [save_img_fpath]
                    if max_num_images_to_render != -1 and len(saved_img_fpaths) > max_num_images_to_render:
                        flag_done = True
                        break
                    continue

                city_SE3_egovehicle = dl.get_city_SE3_egovehicle(log_id, cam_timestamp)
                if city_SE3_egovehicle is None:
                    continue

                lidar_timestamp = Path(ply_fpath).stem.split("_")[-1]
                lidar_timestamp = int(lidar_timestamp)
                labels = dl.get_labels_at_lidar_timestamp(log_id, lidar_timestamp)
                if labels is None:
                    logging.info("\tLabels missing at t=%s", lidar_timestamp)
                    continue

                # Swap channel order as OpenCV expects it -- BGR not RGB
                # must make a copy to make memory contiguous
                img = imageio.imread(im_fpath)[:, :, ::-1].copy()
                camera_config = get_calibration_config(log_calib_data, camera_name)
                planes = generate_frustum_planes(camera_config.intrinsic.copy(), camera_name)

                if not omit_centerlines:
                    img = plot_lane_centerlines_in_img(
                        lidar_pts,
                        city_SE3_egovehicle,
                        img,
                        city_name,
                        avm,
                        camera_config,
                        planes,
                    )

                for label_idx, label in enumerate(labels):
                    obj_rec = json_label_dict_to_obj_record(label)
                    if obj_rec.occlusion == 100:
                        continue

                    cuboid_vertices = obj_rec.as_3d_bbox()
                    points_h = point_cloud_to_homogeneous(cuboid_vertices).T
                    if motion_compensate:
                        (uv, uv_cam, valid_pts_bool, K,) = project_lidar_to_img_motion_compensated(
                            points_h,  # these are recorded at lidar_time
                            copy.deepcopy(log_calib_data),
                            camera_name,
                            cam_timestamp,
                            lidar_timestamp,
                            data_dir,
                            log_id,
                            return_K=True,
                        )
                    else:
                        # project_lidar_to_img
                        (
                            uv,
                            uv_cam,
                            valid_pts_bool,
                            camera_config,
                        ) = project_lidar_to_undistorted_img(points_h, copy.deepcopy(log_calib_data), camera_name)
                    if valid_pts_bool.sum() == 0:
                        continue

                    img = obj_rec.render_clip_frustum_cv2(
                        img,
                        uv_cam.T[:, :3],
                        planes.copy(),
                        copy.deepcopy(camera_config),
                    )

                if generate_video_only:
                    video_writer.add_frame(rgb_frame=img[:, :, ::-1])
                else:
                    cv2.imwrite(save_img_fpath, img)
                    saved_img_fpaths += [save_img_fpath]
                if (
                    not generate_video_only
                    and max_num_images_to_render != -1
                    and len(saved_img_fpaths) > max_num_images_to_render
                ):
                    flag_done = True
                    break
            if generate_video_only:
                video_writer.complete()
                ffmpeg_compress_video(mp4_path, fps)
            if flag_done:
                break

        if not generate_video_only:
            for cam_idx, camera_name in enumerate(RING_CAMERA_LIST + STEREO_CAMERA_LIST):
                # Write the cuboid video from individual frames -- could also write w/ fps=20,30,40
                fps = fps_map[camera_name]
                img_wildcard = f"{save_dir}/{camera_name}_%*.jpg"
                output_fpath = f"{video_output_dir}/{log_id}_{camera_name}_{fps}fps.mp4"
                write_nonsequential_idx_video(img_wildcard, output_fpath, fps)

    return saved_img_fpaths


def main(args: Any):
    """Run the example."""
    log_ids = [log_id.strip() for log_id in args.log_ids.split(",")]
    if args.use_multiprocessing:
        single_process_args = [
            (
                [log_id],
                args.max_num_images_to_render * 9,
                args.dataset_dir,
                args.experiment_prefix,
                not args.no_motion_compensation,
                args.omit_centerlines,
                args.generate_video_only,
            )
            for log_id in log_ids
        ]
        with Pool(os.cpu_count()) as p:
            p.starmap(dump_clipped_3d_cuboids_to_images, single_process_args)

    else:
        # run in a single process, instead
        dump_clipped_3d_cuboids_to_images(
            log_ids=log_ids,
            max_num_images_to_render=args.max_num_images_to_render * 9,
            data_dir=args.dataset_dir,
            experiment_prefix=args.experiment_prefix,
            motion_compensate=not args.no_motion_compensation,
            omit_centerlines=args.omit_centerlines,
            generate_video_only=args.generate_video_only,
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-num-images-to-render",
        default=5,
        type=int,
        help="number of images within which to render 3d cuboids",
    )
    parser.add_argument("--dataset-dir", type=str, required=True, help="path to the dataset folder")
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="uses multiprocessing only if arg is specified on command line, otherwise single process",
    )
    parser.add_argument(
        "--no-motion-compensation",
        action="store_true",
        help="motion compensate by default, unless arg is specified on command line to not do so",
    )
    parser.add_argument(
        "--omit-centerlines",
        action="store_true",
        help="renders centerlines by default, will omit them if arg is specified on command line",
    )
    parser.add_argument(
        "--generate-video-only",
        action="store_true",
        help="produces mp4 files only, without dumping any individual frames/images to JPGs",
    )
    parser.add_argument(
        "--log-ids",
        type=str,
        required=True,
        help="comma separated list of log ids, each log_id represents a log directory, e.g. found at "
        " {args.dataset-dir}/argoverse-tracking/train/{log_id} or "
        " {args.dataset-dir}/argoverse-tracking/sample/{log_id} or ",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="output",
        type=str,
        help="results will be saved in a folder with this prefix for its name",
    )
    args = parser.parse_args()
    logger.info(args)

    if args.log_ids is None:
        logger.error(f"Please provide a comma seperated list of log ids")
        raise ValueError(f"Please provide a comma seperated list of log ids")

    main(args)
