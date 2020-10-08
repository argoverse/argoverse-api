# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import copy
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.object_classes import OBJ_CLASS_MAPPING_DICT
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.utils.calibration import Calibration, determine_valid_cam_coords, proj_cam_to_uv
from argoverse.utils.frustum_clipping import generate_frustum_planes

point_size = 0.01
axes_limits = [
    [-10, 10],
    [-10, 10],
    [-3, 10],
]  # X axis range  # Y axis range  # Z axis range
axes_str = ["X", "Y", "Z"]

_COLOR_MAP = [
    (float(np.random.rand()), float(np.random.rand()), float(np.random.rand()))
    for i in range(len(OBJ_CLASS_MAPPING_DICT) + 1)
]


logger = logging.getLogger(__name__)


def _get_axes_or_default(axes: Optional[Any]) -> Any:
    if axes is None:
        return [1, 0, 2]
    else:
        return axes


def draw_point_cloud(
    ax: plt.Axes,
    title: str,
    argoverse_data: ArgoverseTrackingLoader,
    idx: int,
    axes: Optional[Any] = None,
    xlim3d: Any = None,
    ylim3d: Any = None,
    zlim3d: Any = None,
) -> None:
    axes = _get_axes_or_default(axes)
    pc = argoverse_data.get_lidar(idx)
    assert isinstance(pc, np.ndarray)
    objects = argoverse_data.get_label_object(idx)
    ax.scatter(*np.transpose(pc[:, axes]), s=point_size, c=pc[:, 2], cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("{} axis".format(axes_str[axes[0]]))
    ax.set_ylabel("{} axis".format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel("{} axis".format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
    if xlim3d != None:
        ax.set_xlim3d(xlim3d)
    if ylim3d != None:
        ax.set_ylim3d(ylim3d)
    if zlim3d != None:
        ax.set_zlim3d(zlim3d)

    for obj in objects:
        if obj.occlusion == 100:
            continue

        if obj.label_class is None:
            logger.warning("No label class, just picking the default color.")
            color = _COLOR_MAP[-1]
        else:
            color = _COLOR_MAP[OBJ_CLASS_MAPPING_DICT[obj.label_class]]

        draw_box(ax, obj.as_3d_bbox().T, axes=axes, color=color)


def draw_point_cloud_trajectory(
    ax: plt.Axes,
    title: str,
    argoverse_data: ArgoverseTrackingLoader,
    idx: int,
    axes: Optional[Any] = None,
    xlim3d: Any = None,
    ylim3d: Any = None,
    zlim3d: Any = None,
) -> None:
    axes = _get_axes_or_default(axes)
    unique_id_list = set()
    for i in range(len(argoverse_data.lidar_list)):
        for label in argoverse_data.get_label_object(i):
            unique_id_list.add(label.track_id)
    color_map = {
        track_id: (
            float(np.random.rand()),
            float(np.random.rand()),
            float(np.random.rand()),
        )
        for track_id in unique_id_list
    }
    pc = argoverse_data.get_lidar(idx)
    assert isinstance(pc, np.ndarray)
    objects = argoverse_data.get_label_object(idx)
    ax.scatter(*np.transpose(pc[:, axes]), s=point_size, c=pc[:, 2], cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("{} axis".format(axes_str[axes[0]]))
    ax.set_ylabel("{} axis".format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel("{} axis".format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
    if xlim3d != None:
        ax.set_xlim3d(xlim3d)
    if ylim3d != None:
        ax.set_ylim3d(ylim3d)
    if zlim3d != None:
        ax.set_zlim3d(zlim3d)
    visible_track_id = set()
    for obj in objects:
        if obj.occlusion == 100:
            continue
        draw_box(ax, obj.as_3d_bbox().T, axes=axes, color=color_map[obj.track_id])

        visible_track_id.add(obj.track_id)

    current_pose = argoverse_data.get_pose(idx)
    traj_by_id: Dict[Optional[str], List[Any]] = defaultdict(list)
    for i in range(0, idx, 1):
        if current_pose is None:
            logger.warning("`current_pose` is missing at index %d", idx)
            break

        pose = argoverse_data.get_pose(i)
        if pose is None:
            logger.warning("`pose` is missing at index %d", i)
            continue

        objects = argoverse_data.get_label_object(i)

        for obj in objects:
            if obj.occlusion == 100:
                continue
            if obj.track_id is None or obj.track_id not in visible_track_id:
                continue
            x, y, z = pose.transform_point_cloud(np.array([np.array(obj.translation)]))[0]

            x, y, _ = current_pose.inverse_transform_point_cloud(np.array([np.array([x, y, z])]))[0]

            # ax.scatter(x,y, s=point_size, c=color_map[obj.track_id])
            if obj.track_id is None:
                logger.warning("Label has no track_id.  Collisions with other tracks that are missing IDs could happen")

            traj_by_id[obj.track_id].append([x, y])

    for track_id in traj_by_id.keys():
        traj = np.array(traj_by_id[track_id])
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color_map[track_id],
            linestyle="--",
            linewidth=1,
        )


def draw_box(
    pyplot_axis: plt.Axes,
    vertices: np.ndarray,
    axes: Optional[Any] = None,
    color: Union[str, Tuple[float, float, float]] = "red",
) -> None:
    axes = _get_axes_or_default(axes)
    vertices = vertices[axes, :]
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


def show_image_with_boxes(img: np.ndarray, objects: Iterable[ObjectLabelRecord], calib: Calibration) -> np.ndarray:
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)
    K = calib.K
    d = calib.d

    h, w = np.shape(img1)[0:2]
    planes = generate_frustum_planes(calib.K, calib.camera)
    assert planes is not None

    for obj in objects:
        if obj.occlusion == 100:
            continue
        box3d_pts_3d = obj.as_3d_bbox()
        uv = calib.project_ego_to_image(box3d_pts_3d)
        uv_cam = calib.project_ego_to_cam(box3d_pts_3d)

        img1 = obj.render_clip_frustum_cv2(
            img1,
            uv_cam[:, :3],
            planes.copy(),
            copy.deepcopy(calib.camera_config),
            linewidth=3,
        )

    return img1


def make_grid_ring_camera(argoverse_data: ArgoverseTrackingLoader, idx: int) -> Tuple[plt.Figure, plt.Axes]:

    f, ax = plt.subplots(3, 3, figsize=(20, 15))

    camera = "ring_front_left"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes(img, objects, calib))
    ax[0, 0].imshow(img_vis)
    ax[0, 0].set_title("Ring Front Left")
    ax[0, 0].axis("off")

    camera = "ring_front_center"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes(img, objects, calib))
    ax[0, 1].imshow(img_vis)
    ax[0, 1].set_title("Right Front Center")
    ax[0, 1].axis("off")

    camera = "ring_front_right"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes(img, objects, calib))
    ax[0, 2].imshow(img_vis)
    ax[0, 2].set_title("Ring Front Right")
    ax[0, 2].axis("off")

    camera = "ring_side_left"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes(img, objects, calib))
    ax[1, 0].imshow(img_vis)
    ax[1, 0].set_title("Ring Side Left")
    ax[1, 0].axis("off")

    ax[1, 1].axis("off")

    camera = "ring_side_right"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes(img, objects, calib))
    ax[1, 2].imshow(img_vis)
    ax[1, 2].set_title("Ring Side Right")
    ax[1, 2].axis("off")

    camera = "ring_rear_left"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes(img, objects, calib))
    ax[2, 0].imshow(img_vis)
    ax[2, 0].set_title("Ring Rear Left")
    ax[2, 0].axis("off")

    ax[2, 1].axis("off")

    camera = "ring_rear_right"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes(img, objects, calib))
    ax[2, 2].imshow(img_vis)
    ax[2, 2].set_title("Ring Rear Right")
    ax[2, 2].axis("off")

    return f, ax
