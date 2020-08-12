#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import copy
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from argoverse.utils.calibration import CameraConfig
from argoverse.utils.cv2_plotting_utils import draw_clipped_line_segment
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat


class ObjectLabelRecord:
    def __init__(
        self,
        quaternion: np.array,
        translation: np.array,
        length: float,
        width: float,
        height: float,
        occlusion: int,
        label_class: Optional[str] = None,
        track_id: Optional[str] = None,
    ) -> None:
        """Create an ObjectLabelRecord.

        Args:
           quaternion: Numpy vector representing quaternion, box/cuboid orientation
           translation: Numpy vector representing translation, center of box given as x, y, z.
           length: object length.
           width: object width.
           height: object height.
           occlusion: occlusion value.
           label_class: class label, see object_classes.py for all possible class in argoverse
           track_id: object track id, this is unique for each track
        """
        self.quaternion = quaternion
        self.translation = translation
        self.length = length
        self.width = width
        self.height = height
        self.occlusion = occlusion
        self.label_class = label_class
        self.track_id = track_id

    def as_2d_bbox(self) -> np.ndarray:
        """Construct a 2D bounding box from this label.

        Length is x, width is y, and z is height

        Alternatively could write code like::

            x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
            y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
            z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
            corners = np.vstack((x_corners, y_corners, z_corners))
        """
        bbox_object_frame = np.array(
            [
                [self.length / 2.0, self.width / 2.0, self.height / 2.0],
                [self.length / 2.0, -self.width / 2.0, self.height / 2.0],
                [-self.length / 2.0, self.width / 2.0, self.height / 2.0],
                [-self.length / 2.0, -self.width / 2.0, self.height / 2.0],
            ]
        )

        egovehicle_SE3_object = SE3(rotation=quat2rotmat(self.quaternion), translation=self.translation)
        bbox_in_egovehicle_frame = egovehicle_SE3_object.transform_point_cloud(bbox_object_frame)
        return bbox_in_egovehicle_frame

    def as_3d_bbox(self) -> np.ndarray:
        r"""Calculate the 8 bounding box corners.

        Args:
            None

        Returns:
            Numpy array of shape (8,3)

        Corner numbering::

             5------4
             |\\    |\\
             | \\   | \\
             6--\\--7  \\
             \\  \\  \\ \\
         l    \\  1-------0    h
          e    \\ ||   \\ ||   e
           n    \\||    \\||   i
            g    \\2------3    g
             t      width.     h
              h.               t.

        First four corners are the ones facing forward.
        The last four are the ones facing backwards.
        """
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = self.length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = self.width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = self.height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners_object_frame = np.vstack((x_corners, y_corners, z_corners)).T

        egovehicle_SE3_object = SE3(rotation=quat2rotmat(self.quaternion), translation=self.translation)
        corners_egovehicle_frame = egovehicle_SE3_object.transform_point_cloud(corners_object_frame)
        return corners_egovehicle_frame

    def render_clip_frustum_cv2(
        self,
        img: np.array,
        corners: np.array,
        planes: List[Tuple[np.array, np.array, np.array, np.array, np.array]],
        camera_config: CameraConfig,
        colors: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]] = (
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 0),
        ),
        linewidth: int = 2,
    ) -> np.ndarray:
        r"""We bring the 3D points into each camera, and do the clipping there.

        Renders box using OpenCV2. Roughly based on
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes_utils/data_classes.py

        ::

                5------4
                |\\    |\\
                | \\   | \\
                6--\\--7  \\
                \\  \\  \\ \\
            l    \\  1-------0    h
             e    \\ ||   \\ ||   e
              n    \\||    \\||   i
               g    \\2------3    g
                t      width.     h
                 h.               t.

        Args:
            img: Numpy array of shape (M,N,3)
            corners: Numpy array of shape (8,3) in camera coordinate frame.
            planes: Iterable of 5 clipping planes. Each plane is defined by 4 points.
            camera_config: CameraConfig object
            colors: tuple of RGB 3-tuples, Colors for front, side & rear.
                defaults are    0. blue (0,0,255) in RGB and (255,0,0) in OpenCV's BGR
                                1. red (255,0,0) in RGB and (0,0,255) in OpenCV's BGR
                                2. green (0,255,0) in RGB and BGR alike.
            linewidth: integer, linewidth for plot

        Returns:
            img: Numpy array of shape (M,N,3), representing updated image
        """

        def draw_rect(selected_corners: np.array, color: Tuple[int, int, int]) -> None:
            prev = selected_corners[-1]
            for corner in selected_corners:
                draw_clipped_line_segment(img, prev.copy(), corner.copy(), camera_config, linewidth, planes, color)
                prev = corner

        # Draw the sides in green
        for i in range(4):
            # between front and back corners
            draw_clipped_line_segment(
                img, corners[i], corners[i + 4], camera_config, linewidth, planes, colors[2][::-1]
            )

        # Draw front (first 4 corners) in blue
        draw_rect(corners[:4], colors[0][::-1])
        # Draw rear (last 4 corners) in red
        draw_rect(corners[4:], colors[1][::-1])

        # Draw blue line indicating the front half
        center_bottom_forward = np.mean(corners[2:4], axis=0)
        center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
        draw_clipped_line_segment(
            img, center_bottom, center_bottom_forward, camera_config, linewidth, planes, colors[0][::-1]
        )

        return img


def form_obj_label_from_json(label: Dict[str, Any]) -> Tuple[np.array, str]:
    """Construct object from loaded json.

    The dictionary loaded from saved json file is expected to have the
    following fields::

        ['frame_index', 'center', 'rotation', 'length', 'width', 'height',
        'track_label_uuid', 'occlusion', 'on_driveable_surface', 'key_frame',
        'stationary', 'label_class']

   Args:
        label: Python dictionary that was loaded from saved json file

    Returns:
        Tuple of (bbox_ego_frame, color); bbox is a numpy array of shape (4,3); color is "g" or "r"
    """
    tr_x = label["center"]["x"]
    tr_y = label["center"]["y"]
    tr_z = label["center"]["z"]
    translation = np.array([tr_x, tr_y, tr_z])

    rot_w = label["rotation"]["w"]
    rot_x = label["rotation"]["x"]
    rot_y = label["rotation"]["y"]
    rot_z = label["rotation"]["z"]
    quaternion = np.array([rot_w, rot_x, rot_y, rot_z])

    obj_label_rec = ObjectLabelRecord(
        quaternion=quaternion,
        translation=translation,
        length=label["length"],
        width=label["width"],
        height=label["height"],
        occlusion=label["occlusion"],
    )
    bbox_ego_frame = obj_label_rec.as_2d_bbox()
    if label["occlusion"] == 0:
        color = "g"
    else:
        color = "r"
    return bbox_ego_frame, color


def json_label_dict_to_obj_record(label: Dict[str, Any]) -> ObjectLabelRecord:
    """Convert a label dict (from JSON) to an ObjectLabelRecord.

    NB: "Shrink-wrapped" objects don't have the occlusion field, but
    other other objects do.

    Args:
        label: Python dictionary with relevant info about a cuboid, loaded from json

    Returns:
        ObjectLabelRecord object
    """
    tr_x = label["center"]["x"]
    tr_y = label["center"]["y"]
    tr_z = label["center"]["z"]
    translation = np.array([tr_x, tr_y, tr_z])

    rot_w = label["rotation"]["w"]
    rot_x = label["rotation"]["x"]
    rot_y = label["rotation"]["y"]
    rot_z = label["rotation"]["z"]
    quaternion = np.array([rot_w, rot_x, rot_y, rot_z])

    length = label["length"]
    width = label["width"]
    height = label["height"]

    if "occlusion" in label:
        occlusion = label["occlusion"]
    else:
        occlusion = 0

    if "label_class" in label:
        label_class = label["label_class"]
        if "name" in label_class:
            label_class = label_class["name"]
    else:
        label_class = None

    if "track_label_uuid" in label:
        track_id = label["track_label_uuid"]
    else:
        track_id = None
    obj_rec = ObjectLabelRecord(quaternion, translation, length, width, height, occlusion, label_class, track_id)
    return obj_rec


def read_label(label_filename: str) -> List[ObjectLabelRecord]:
    """Read label from the json file.

    Args:
        label_filename: label filename,

    Returns:
        List of ObjectLabelRecords constructed from the file.
    """
    if not os.path.exists(label_filename):
        return []
    with open(label_filename, "r") as f:
        labels = json.load(f)

    objects = [json_label_dict_to_obj_record(item) for item in labels]
    return objects
