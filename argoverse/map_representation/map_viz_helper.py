# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Map visualization helper functions."""
import copy
import math
import warnings
from typing import Mapping, Tuple

import cv2
import numpy as np
from colour import Color
from typing_extensions import Protocol

from argoverse.utils.cv2_plotting_utils import draw_polygon_cv2, draw_polyline_cv2
from argoverse.utils.datetime_utils import generate_datetime_string
from argoverse.utils.mesh_grid import get_mesh_grid_as_point_cloud
from argoverse.utils.se2 import SE2

from .lane_segment import LaneSegment

__all__ = ["render_global_city_map_bev"]

LaneCenterline = Mapping[int, LaneSegment]


def _find_min_coords_das(
    driveable_areas: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float
) -> Tuple[float, float, float, float]:
    for da in driveable_areas:
        xmin = min(da[:, 0].min(), xmin)
        ymin = min(da[:, 1].min(), ymin)
        xmax = max(da[:, 0].max(), xmax)
        ymax = max(da[:, 1].max(), ymax)

    return xmin, ymin, xmax, ymax


def _find_min_coords_centerlines(
    lane_centerlines: LaneCenterline, xmin: float, ymin: float, xmax: float, ymax: float
) -> Tuple[float, float, float, float]:
    for lane_id, lane_obj in lane_centerlines.items():
        centerline_2d = lane_obj.centerline
        xmin = min(centerline_2d[:, 0].min(), xmin)
        ymin = min(centerline_2d[:, 1].min(), ymin)
        xmax = max(centerline_2d[:, 0].max(), xmax)
        ymax = max(centerline_2d[:, 1].max(), ymax)

    return xmin, ymin, xmax, ymax


def _get_opencv_green_to_red_colormap(num_color_bins: int) -> np.ndarray:
    """Create a red to green BGR colormap with a finite number of steps.

    Args:
        num_color_bins: Number of color steps

    Returns:
        The colormap
    """
    color_range = Color("green").range_to(Color("red"), num_color_bins)
    colors_arr = np.array([[color.rgb] for color in color_range]).squeeze()
    # RGB -> BGR for OpenCV's sake
    return np.fliplr(colors_arr)


def _get_int_image_bounds_from_city_coords(
    driveable_areas: np.ndarray, lane_centerlines: LaneCenterline
) -> Tuple[int, int, int, int, int, int]:
    """Get the internal iamge bounds based on the city coordinates

    Args:
        driveable_areas: Driveable area data
        lane_centerlines: Line centerline data

    Returns:
        A tuple containing: image height, image width, and x and y coordinate bounds.
    """
    xmin = float("inf")
    ymin = float("inf")
    xmax = -float("inf")
    ymax = -float("inf")

    xmin, ymin, xmax, ymax = _find_min_coords_das(driveable_areas, xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = _find_min_coords_centerlines(lane_centerlines, xmin, ymin, xmax, ymax)

    xmin = int(math.floor(xmin))
    ymin = int(math.floor(ymin))
    xmax = int(math.ceil(xmax))
    ymax = int(math.ceil(ymax))

    im_h = int(math.ceil(ymax - ymin))
    im_w = int(math.ceil(xmax - xmin))
    return im_h, im_w, xmin, xmax, ymin, ymax


class MapProtocol(Protocol):
    def remove_non_driveable_area_points(self, point_cloud: np.ndarray, city_name: str) -> np.ndarray:
        ...


def render_global_city_map_bev(
    lane_centerlines: LaneCenterline,
    driveable_areas: np.ndarray,
    city_name: str,
    avm: MapProtocol,
    plot_rasterized_roi: bool = True,
    plot_vectormap_roi: bool = False,
    centerline_color_scheme: str = "constant",
) -> None:
    """
    Assume indegree and outdegree are always less than 7.
    Since 1 pixel-per meter resolution is very low, we upsample the resolution
    by some constant factor.

    Args:
        lane_centerlines: Line centerline data
        driveable_areas: Driveable area data
        city_name: The city name (eg. "PIT")
        avm: The map
        plot_rasterized_roi: Whether to show the rasterized ROI
        plot_vectormap_roi: Whether to show the vector map ROI
        centerline_color_scheme: `"indegree"`, `"outdegree"`, or `"constant"`
    """
    UPSAMPLE_FACTOR = 2
    GRID_NUMBER_INTERVAL = 500
    NUM_COLOR_BINS = 7

    warnings.filterwarnings("error")
    im_h, im_w, xmin, xmax, ymin, ymax = _get_int_image_bounds_from_city_coords(driveable_areas, lane_centerlines)
    rendered_image = np.ones((im_h * UPSAMPLE_FACTOR, im_w * UPSAMPLE_FACTOR, 3))
    image_to_city_se2 = SE2(rotation=np.eye(2), translation=np.array([-xmin, -ymin]))

    if plot_rasterized_roi:
        grid_2d_pts = get_mesh_grid_as_point_cloud(xmin + 1, xmax - 1, ymin + 1, ymax - 1)
        pink = np.array([180.0, 105.0, 255.0]) / 255
        roi_2d_pts = avm.remove_non_driveable_area_points(grid_2d_pts, city_name)
        roi_2d_pts = image_to_city_se2.transform_point_cloud(roi_2d_pts)
        roi_2d_pts *= UPSAMPLE_FACTOR
        roi_2d_pts = np.round(roi_2d_pts).astype(np.int32)
        for pt in roi_2d_pts:
            row = pt[1]
            col = pt[0]
            rendered_image[row, col, :] = pink

    if plot_vectormap_roi:
        driveable_areas = copy.deepcopy(driveable_areas)
        for da_idx, da in enumerate(driveable_areas):
            da = image_to_city_se2.transform_point_cloud(da[:, :2])
            rendered_image = draw_polygon_cv2(da * UPSAMPLE_FACTOR, rendered_image, pink)

    for x in range(0, im_w * UPSAMPLE_FACTOR, GRID_NUMBER_INTERVAL):
        for y in range(0, im_h * UPSAMPLE_FACTOR, GRID_NUMBER_INTERVAL):
            coords_str = f"{x}_{y}"
            cv2.putText(
                rendered_image,
                coords_str,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
                bottomLeftOrigin=True,
            )

    # Color the graph
    max_outdegree = 0
    max_indegree = 0

    colors_arr = _get_opencv_green_to_red_colormap(NUM_COLOR_BINS)

    blue = [0, 0, 1][::-1]
    for lane_id, lane_obj in lane_centerlines.items():
        centerline_2d = lane_obj.centerline
        centerline_2d = image_to_city_se2.transform_point_cloud(centerline_2d)
        centerline_2d = np.round(centerline_2d).astype(np.int32)

        preds = lane_obj.predecessors
        succs = lane_obj.successors

        indegree = 0 if preds is None else len(preds)
        outdegree = 0 if succs is None else len(succs)

        if indegree > max_indegree:
            max_indegree = indegree
            print("max indegree", indegree)

        if outdegree > max_outdegree:
            max_outdegree = outdegree
            print("max outdegree", outdegree)

        if centerline_color_scheme == "indegree":
            color = colors_arr[indegree]
        elif centerline_color_scheme == "outdegree":
            color = colors_arr[outdegree]
        elif centerline_color_scheme == "constant":
            color = blue

        draw_polyline_cv2(
            centerline_2d * UPSAMPLE_FACTOR,
            rendered_image,
            color,
            im_h * UPSAMPLE_FACTOR,
            im_w * UPSAMPLE_FACTOR,
        )

    # provide colormap in corner
    for i in range(NUM_COLOR_BINS):
        rendered_image[0, i, :] = colors_arr[i]

    rendered_image = np.flipud(rendered_image)
    rendered_image *= 255
    rendered_image = rendered_image.astype(np.uint8)

    cur_datetime = generate_datetime_string()
    img_save_fpath = f"{city_name}_{centerline_color_scheme}_{cur_datetime}.png"
    cv2.imwrite(filename=img_save_fpath, img=rendered_image)

    warnings.resetwarnings()
