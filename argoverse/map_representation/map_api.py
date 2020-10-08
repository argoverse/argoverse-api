# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

from argoverse.data_loading.vector_map_loader import load_lane_segments_from_xml
from argoverse.utils.centerline_utils import (
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    lane_waypt_to_query_dist,
    remove_overlapping_lane_seq,
)
from argoverse.utils.cv2_plotting_utils import get_img_contours
from argoverse.utils.dilation_utils import dilate_by_l2
from argoverse.utils.geometry import point_inside_polygon
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.manhattan_search import (
    compute_polygon_bboxes,
    find_all_polygon_bboxes_overlapping_query_bbox,
    find_local_polygons,
)
from argoverse.utils.mpl_plotting_utils import plot_lane_segment_patch, visualize_centerline
from argoverse.utils.pkl_utils import load_pkl_dictionary
from argoverse.utils.se2 import SE2

from .lane_segment import LaneSegment
from .map_viz_helper import render_global_city_map_bev

GROUND_HEIGHT_THRESHOLD = 0.3  # 30 centimeters
MAX_LABEL_DIST_TO_LANE = 20  # meters
OUT_OF_RANGE_LANE_DIST_THRESHOLD = 5.0  # 5 meters
ROI_ISOCONTOUR = 5.0

# known City IDs from newest to oldest
MIAMI_ID = 10316
PITTSBURGH_ID = 10314

ROOT = Path(__file__).resolve().parent.parent.parent  # ../../..
MAP_FILES_ROOT = ROOT / "map_files"

# Any numeric type
Number = Union[int, float]


class ArgoverseMap:
    """
    This class provides the interface to our vector maps and rasterized maps. Exact lane boundaries
    are not provided, but can be hallucinated if one considers an average lane width.
    """

    def __init__(self) -> None:
        """ Initialize the Argoverse Map. """
        self.city_name_to_city_id_dict = {"PIT": PITTSBURGH_ID, "MIA": MIAMI_ID}
        self.render_window_radius = 150
        self.im_scale_factor = 50

        self.city_lane_centerlines_dict = self.build_centerline_index()
        (
            self.city_halluc_bbox_table,
            self.city_halluc_tableidx_to_laneid_map,
        ) = self.build_hallucinated_lane_bbox_index()
        self.city_rasterized_da_roi_dict = self.build_city_driveable_area_roi_index()
        self.city_rasterized_ground_height_dict = self.build_city_ground_height_index()

        # get hallucinated lane extends and driveable area from binary img
        self.city_to_lane_polygons_dict: Mapping[str, np.ndarray] = {}
        self.city_to_driveable_areas_dict: Mapping[str, np.ndarray] = {}
        self.city_to_lane_bboxes_dict: Mapping[str, np.ndarray] = {}
        self.city_to_da_bboxes_dict: Mapping[str, np.ndarray] = {}

        for city_name in self.city_name_to_city_id_dict.keys():
            lane_polygons = np.array(self.get_vector_map_lane_polygons(city_name))
            driveable_areas = np.array(self.get_vector_map_driveable_areas(city_name))
            lane_bboxes = compute_polygon_bboxes(lane_polygons)
            da_bboxes = compute_polygon_bboxes(driveable_areas)

            self.city_to_lane_polygons_dict[city_name] = lane_polygons
            self.city_to_driveable_areas_dict[city_name] = driveable_areas
            self.city_to_lane_bboxes_dict[city_name] = lane_bboxes
            self.city_to_da_bboxes_dict[city_name] = da_bboxes

    def get_vector_map_lane_polygons(self, city_name: str) -> List[np.ndarray]:
        """
        Get list of lane polygons for a specified city

        Args:
           city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
           Numpy array of polygons
        """
        lane_polygons = []
        lane_segments = self.city_lane_centerlines_dict[city_name]
        for lane_id, lane_segment in lane_segments.items():
            lane_polygon_xyz = self.get_lane_segment_polygon(lane_segment.id, city_name)
            lane_polygons.append(lane_polygon_xyz)

        return lane_polygons

    def get_vector_map_driveable_areas(self, city_name: str) -> List[np.hstack]:
        """
        Get driveable area for a specified city

        Args:
           city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
           das: driveable areas as n-d array of NumPy objects of shape (n,3)

        Note:
         'z_min', 'z_max' were removed
        """
        return self.get_da_contours(city_name)

    def get_da_contours(self, city_name: str) -> List[np.hstack]:
        """
        We threshold the binary driveable area or ROI image and obtain contour lines. These
        contour lines represent the boundary.

        Args:
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            Drivable area contours
        """
        da_imgray = self.city_rasterized_da_roi_dict[city_name]["da_mat"]
        contours = get_img_contours(da_imgray)

        # pull out 3x3 matrix parameterizing the SE(2) transformation from city coords -> npy image
        npyimage_T_city = self.city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"]
        R = npyimage_T_city[:2, :2]
        t = npyimage_T_city[:2, 2]
        npyimage_SE2_city = SE2(rotation=R, translation=t)
        city_SE2_npyimage = npyimage_SE2_city.inverse()

        city_contours: List[np.ndarray] = []
        for i, contour_im_coords in enumerate(contours):
            contour_im_coords = contour_im_coords.squeeze()
            contour_im_coords = contour_im_coords.astype(np.float64)

            contour_city_coords = city_SE2_npyimage.transform_point_cloud(contour_im_coords)
            city_contours.append(self.append_height_to_2d_city_pt_cloud(contour_city_coords, city_name))

        return city_contours

    def build_centerline_index(self) -> Mapping[str, Mapping[int, LaneSegment]]:
        """
        Build dictionary of centerline for each city, with lane_id as key

        Returns:
            city_lane_centerlines_dict:  Keys are city names, values are dictionaries
                                        (k=lane_id, v=lane info)
        """
        city_lane_centerlines_dict = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            xml_fpath = MAP_FILES_ROOT / f"pruned_argoverse_{city_name}_{city_id}_vector_map.xml"
            city_lane_centerlines_dict[city_name] = load_lane_segments_from_xml(xml_fpath)

        return city_lane_centerlines_dict

    def build_city_driveable_area_roi_index(
        self,
    ) -> Mapping[str, Mapping[str, np.ndarray]]:
        """
        Load driveable area files from disk. Dilate driveable area to get ROI (takes about 1/2 second).

        Returns:
            city_rasterized_da_dict: a dictionary of dictionaries. Key is city_name, and
                    value is a dictionary with driveable area info. For example, includes da_matrix: Numpy array of
                    shape (M,N) representing binary values for driveable area
                    city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        city_rasterized_da_roi_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            city_id = self.city_name_to_city_id_dict[city_name]
            city_rasterized_da_roi_dict[city_name] = {}
            npy_fpath = MAP_FILES_ROOT / f"{city_name}_{city_id}_driveable_area_mat_2019_05_28.npy"
            city_rasterized_da_roi_dict[city_name]["da_mat"] = np.load(npy_fpath)

            se2_npy_fpath = MAP_FILES_ROOT / f"{city_name}_{city_id}_npyimage_to_city_se2_2019_05_28.npy"
            city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"] = np.load(se2_npy_fpath)
            da_mat = copy.deepcopy(city_rasterized_da_roi_dict[city_name]["da_mat"])
            city_rasterized_da_roi_dict[city_name]["roi_mat"] = dilate_by_l2(da_mat, dilation_thresh=ROI_ISOCONTOUR)

        return city_rasterized_da_roi_dict

    def build_city_ground_height_index(self) -> Mapping[str, Mapping[str, np.ndarray]]:
        """
        Build index of rasterized ground height.

        Returns:
            city_ground_height_index: a dictionary of dictionaries. Key is city_name, and values
                    are dictionaries that store the "ground_height_matrix" and also the
                    city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city
                    coordinates, e.g. p_city = city_Transformation_pklimage * p_pklimage
        """
        city_rasterized_ground_height_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            city_rasterized_ground_height_dict[city_name] = {}
            npy_fpath = MAP_FILES_ROOT / f"{city_name}_{city_id}_ground_height_mat_2019_05_28.npy"

            # load the file with rasterized values
            city_rasterized_ground_height_dict[city_name]["ground_height"] = np.load(npy_fpath)

            se2_npy_fpath = MAP_FILES_ROOT / f"{city_name}_{city_id}_npyimage_to_city_se2_2019_05_28.npy"
            city_rasterized_ground_height_dict[city_name]["npyimage_to_city_se2"] = np.load(se2_npy_fpath)

        return city_rasterized_ground_height_dict

    def get_rasterized_driveable_area(self, city_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the driveable area.

        Args:
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        da_mat = self.city_rasterized_da_roi_dict[city_name]["da_mat"]
        return (
            da_mat,
            self.city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"],
        )

    def get_rasterized_roi(self, city_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the driveable area.

        Args:
            city_name: string, either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        roi_mat = self.city_rasterized_da_roi_dict[city_name]["roi_mat"]
        return (
            roi_mat,
            self.city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"],
        )

    def build_hallucinated_lane_bbox_index(
        self,
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Populate the pre-computed hallucinated extent of each lane polygon, to allow for fast
        queries.

        Returns:
            city_halluc_bbox_table
            city_id_to_halluc_tableidx_map
        """

        city_halluc_bbox_table = {}
        city_halluc_tableidx_to_laneid_map = {}

        for city_name, city_id in self.city_name_to_city_id_dict.items():
            json_fpath = MAP_FILES_ROOT / f"{city_name}_{city_id}_tableidx_to_laneid_map.json"
            city_halluc_tableidx_to_laneid_map[city_name] = read_json_file(json_fpath)

            npy_fpath = MAP_FILES_ROOT / f"{city_name}_{city_id}_halluc_bbox_table.npy"
            city_halluc_bbox_table[city_name] = np.load(npy_fpath)

        return city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map

    def render_city_centerlines(self, city_name: str) -> None:
        """
        Draw centerlines for the entire city_name

        Args:
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        """
        lane_centerlines = self.city_lane_centerlines_dict[city_name]
        das = self.city_to_driveable_areas_dict[city_name]
        render_global_city_map_bev(lane_centerlines, das, city_name, self, centerline_color_scheme="indegree")

    def get_rasterized_ground_height(self, city_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ground height matrix along with se2 that convert to city coordinate

        Args:
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            ground_height_matrix
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        ground_height_mat = self.city_rasterized_ground_height_dict[city_name]["ground_height"]
        return (
            ground_height_mat,
            self.city_rasterized_ground_height_dict[city_name]["npyimage_to_city_se2"],
        )

    def remove_ground_surface(
        self, point_cloud: np.ndarray, city_name: str, return_logicals: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get a lidar point, snap it to the grid, perform the O(1) raster map query.
        If our z-height is within THRESHOLD of that grid's z-height, then we keep it; otherwise, discard it

        Args:
            point_cloud: NumPy n-d array of shape (n,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            return_logicals: whether to return pointwise boolean of function result
        Returns:
            subset of original point cloud, with ground points removed
            optionally, pass boolean array where `True` indicates point was not part of "ground"
        """
        is_ground_boolean_arr = self.get_ground_points_boolean(point_cloud, city_name)
        not_ground_logicals = np.logical_not(is_ground_boolean_arr)
        not_ground_indxs = np.where(not_ground_logicals)[0]

        if return_logicals:
            return point_cloud[not_ground_indxs], not_ground_logicals

        return point_cloud[not_ground_indxs]

    def remove_non_driveable_area_points(self, point_cloud: np.ndarray, city_name: str) -> np.ndarray:
        """
        Get a lidar point, snap it to the grid, perform the O(1) raster map query.
        If our z-height is within THRESHOLD of that grid's z-height, then we keep it; otherwise, discard it

        Decimate the point cloud to the driveable area only.

        Args:
            point_cloud: NumPy n-d array of shape (n,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        Returns:
            lidar_point_cloud: subset of original point cloud, with non-driveable area removed
        """
        is_da_boolean_arr = self.get_raster_layer_points_boolean(point_cloud, city_name, "driveable_area")
        return point_cloud[is_da_boolean_arr]

    def remove_non_roi_points(self, point_cloud: np.ndarray, city_name: str) -> np.ndarray:
        """
        Remove any points that does't fall within the region of interest (ROI)

        Args:
            point_cloud: NumPy n-d array of shape (n,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            lidar_point_cloud: subset of original point cloud, with ROI points removed
        """
        is_roi_boolean_arr = self.get_raster_layer_points_boolean(point_cloud, city_name, "roi")
        return point_cloud[is_roi_boolean_arr]

    def get_ground_points_boolean(self, point_cloud: np.ndarray, city_name: str) -> np.ndarray:
        """
        Check whether each point is likely to be from the ground surface

        Args:
            point_cloud: Numpy array of shape (N,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            is_ground_boolean_arr: Numpy array of shape (N,) where ith entry is True if the LiDAR return
                is likely a hit from the ground surface.
        """
        ground_height_values = self.get_ground_height_at_xy(point_cloud, city_name)
        is_ground_boolean_arr = (np.absolute(point_cloud[:, 2] - ground_height_values) <= GROUND_HEIGHT_THRESHOLD) | (
            np.array(point_cloud[:, 2] - ground_height_values) < 0
        )
        return is_ground_boolean_arr

    def get_ground_height_at_xy(self, point_cloud: np.ndarray, city_name: str) -> np.ndarray:
        """
        Get ground height for each of the xy location in point_cloud

        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            ground_height_values: Numpy array of shape (k,)
        """
        ground_height_mat, npyimage_to_city_se2_mat = self.get_rasterized_ground_height(city_name)
        city_coords = np.round(point_cloud[:, :2]).astype(np.int64)

        se2_rotation = npyimage_to_city_se2_mat[:2, :2]
        se2_trans = npyimage_to_city_se2_mat[:2, 2]

        npyimage_to_city_se2 = SE2(rotation=se2_rotation, translation=se2_trans)
        npyimage_coords = npyimage_to_city_se2.transform_point_cloud(city_coords)
        npyimage_coords = npyimage_coords.astype(np.int64)

        ground_height_values = np.full((npyimage_coords.shape[0]), np.nan)
        ind_valid_pts = (npyimage_coords[:, 1] < ground_height_mat.shape[0]) * (
            npyimage_coords[:, 0] < ground_height_mat.shape[1]
        )

        ground_height_values[ind_valid_pts] = ground_height_mat[
            npyimage_coords[ind_valid_pts, 1], npyimage_coords[ind_valid_pts, 0]
        ]

        return ground_height_values

    def append_height_to_2d_city_pt_cloud(self, pt_cloud_xy: np.ndarray, city_name: str) -> np.ndarray:
        """
        Accept 2d point cloud in xy plane and return 3d point cloud (xyz)

        Args:
            pt_cloud_xy: Numpy array of shape (N,2)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            pt_cloud_xyz: Numpy array of shape (N,3)
        """
        pts_z = self.get_ground_height_at_xy(pt_cloud_xy, city_name)
        return np.hstack([pt_cloud_xy, pts_z[:, np.newaxis]])

    def get_raster_layer_points_boolean(self, point_cloud: np.ndarray, city_name: str, layer_name: str) -> np.ndarray:
        """
        driveable area is "da"

        Args:
            point_cloud: Numpy array of shape (N,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            layer_name: indicating layer name, either "roi" or "driveable area"

        Returns:
            is_ground_boolean_arr: Numpy array of shape (N,) where ith entry is True if the LiDAR return
                is likely a hit from the ground surface.
        """
        if layer_name == "roi":
            layer_raster_mat, npyimage_to_city_se2_mat = self.get_rasterized_roi(city_name)
        elif layer_name == "driveable_area":
            (
                layer_raster_mat,
                npyimage_to_city_se2_mat,
            ) = self.get_rasterized_driveable_area(city_name)
        else:
            raise ValueError("layer_name should be wither roi or driveable_area.")

        city_coords = np.round(point_cloud[:, :2]).astype(np.int64)

        se2_rotation = npyimage_to_city_se2_mat[:2, :2]
        se2_trans = npyimage_to_city_se2_mat[:2, 2]

        npyimage_to_city_se2 = SE2(rotation=se2_rotation, translation=se2_trans)
        npyimage_coords = npyimage_to_city_se2.transform_point_cloud(city_coords)
        npyimage_coords = npyimage_coords.astype(np.int64)

        # index in at (x,y) locations, which are (y,x) in the image
        layer_values = np.full((npyimage_coords.shape[0]), 0.0)
        ind_valid_pts = (
            (npyimage_coords[:, 1] > 0)
            * (npyimage_coords[:, 1] < layer_raster_mat.shape[0])
            * (npyimage_coords[:, 0] > 0)
            * (npyimage_coords[:, 0] < layer_raster_mat.shape[1])
        )
        layer_values[ind_valid_pts] = layer_raster_mat[
            npyimage_coords[ind_valid_pts, 1], npyimage_coords[ind_valid_pts, 0]
        ]
        is_layer_boolean_arr = layer_values == 1.0
        return is_layer_boolean_arr

    def get_nearest_centerline(
        self, query_xy_city_coords: np.ndarray, city_name: str, visualize: bool = False
    ) -> Tuple[LaneSegment, float, np.ndarray]:
        """
        KD Tree with k-closest neighbors or a fixed radius search on the lane centroids
        is unreliable since (1) there is highly variable density throughout the map and (2)
        lane lengths differ enormously, meaning the centroid is not indicative of nearby points.
        If no lanes are found with MAX_LABEL_DIST_TO_LANE, we increase the search radius.

        A correct approach is to compare centerline-to-query point distances, e.g. as done
        in Shapely. Instead of looping over all points, we precompute the bounding boxes of
        each lane.

        We use the closest_waypoint as our criterion. Using the smallest sum to waypoints
        does not work in many cases with disproportionately shaped lane segments.

        and then choose the lane centerline with the smallest sum of 3-5
        closest waypoints.

        Args:
            query_xy_city_coords: Numpy array of shape (2,) representing xy position of query in city coordinates
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            visualize:

        Returns:
            lane_object: Python dictionary with fields describing a lane.
                Keys include: 'centerline', 'predecessor', 'successor', 'turn_direction',
                             'is_intersection', 'has_traffic_control', 'is_autonomous', 'is_routable'
            conf: real-valued confidence. less than 0.85 is almost always unreliable
            dense_centerline: numpy array
        """
        query_x = query_xy_city_coords[0]
        query_y = query_xy_city_coords[1]

        lane_centerlines_dict = self.city_lane_centerlines_dict[city_name]

        search_radius = MAX_LABEL_DIST_TO_LANE
        while True:
            nearby_lane_ids = self.get_lane_ids_in_xy_bbox(
                query_x, query_y, city_name, query_search_range_manhattan=search_radius
            )
            if not nearby_lane_ids:
                search_radius *= 2  # double search radius
            else:
                break

        nearby_lane_objs = [lane_centerlines_dict[lane_id] for lane_id in nearby_lane_ids]

        cache = lane_waypt_to_query_dist(query_xy_city_coords, nearby_lane_objs)
        per_lane_dists, min_dist_nn_indices, dense_centerlines = cache

        closest_lane_obj = nearby_lane_objs[min_dist_nn_indices[0]]
        dense_centerline = dense_centerlines[min_dist_nn_indices[0]]

        # estimate confidence
        conf = 1.0 - (per_lane_dists.min() / OUT_OF_RANGE_LANE_DIST_THRESHOLD)
        conf = max(0.0, conf)  # clip to ensure positive value

        if visualize:
            # visualize dists to nearby centerlines
            fig = plt.figure(figsize=(22.5, 8))
            ax = fig.add_subplot(111)

            (query_x, query_y) = query_xy_city_coords.squeeze()
            ax.scatter([query_x], [query_y], 100, color="k", marker=".")
            # make another plot now!

            self.plot_nearby_halluc_lanes(ax, city_name, query_x, query_y)

            for i, line in enumerate(dense_centerlines):
                ax.plot(line[:, 0], line[:, 1], color="y")
                ax.text(line[:, 0].mean(), line[:, 1].mean(), str(per_lane_dists[i]))

            ax.axis("equal")
            plt.show()
            plt.close("all")
        return closest_lane_obj, conf, dense_centerline

    def get_lane_direction(
        self, query_xy_city_coords: np.ndarray, city_name: str, visualize: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Get vector direction of the lane you're in.
        We ignore the sparse version of the centerline that we could
        trivially pull from lane_obj['centerline'].

        Args:
            query_xy_city_coords: Numpy array of shape (2,) representing (x,y) position in city coordinates
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            visualize: to also visualize the result

        Returns:
            lane_dir_vector: Numpy array of shape (2,) representing the direction (as a vector) of the closest
                lane to the provided position in city coordinates
            conf: real-valued confidence. less than 0.85 is almost always unreliable

        We have access to all of the following fields in "lane_obj":
            'centerline', 'predecessor', 'successor', 'turn_direction',
            'is_intersection', 'has_traffic_control'
        """
        cache = self.get_nearest_centerline(query_xy_city_coords, city_name)
        lane_obj, confidence, dense_centerline = cache
        centerline = dense_centerline

        waypoint_dists = np.linalg.norm(centerline - query_xy_city_coords, axis=1)
        closest_waypt_indxs = np.argsort(waypoint_dists)[:2]

        prev_waypoint_id = closest_waypt_indxs.min()
        next_waypoint_id = closest_waypt_indxs.max()

        prev_waypoint = centerline[prev_waypoint_id]
        next_waypoint = centerline[next_waypoint_id]

        lane_dir_vector = next_waypoint - prev_waypoint
        if visualize:
            plt.plot(centerline[:, 0], centerline[:, 1], color="y")
            plt.scatter(
                query_xy_city_coords[0],
                query_xy_city_coords[1],
                200,
                marker=".",
                color="b",
            )
            dx = lane_dir_vector[0] * 10
            dy = lane_dir_vector[1] * 10
            plt.arrow(
                query_xy_city_coords[0],
                query_xy_city_coords[1],
                dx,
                dy,
                color="r",
                width=0.3,
                zorder=2,
            )
            centerline_length = centerline.shape[0]
            for i in range(centerline_length):
                plt.scatter(centerline[i, 0], centerline[i, 1], i / 5.0, marker=".", color="k")
            plt.axis("equal")
            plt.show()
            plt.close("all")

        return lane_dir_vector, confidence

    def get_lane_ids_in_xy_bbox(
        self,
        query_x: float,
        query_y: float,
        city_name: str,
        query_search_range_manhattan: float = 5.0,
    ) -> List[int]:
        """
        Prune away all lane segments based on Manhattan distance. We vectorize this instead
        of using a for-loop. Get all lane IDs within a bounding box in the xy plane.
        This is a approximation of a bubble search for point-to-polygon distance.

        The bounding boxes of small point clouds (lane centerline waypoints) are precomputed in the map.
        We then can perform an efficient search based on manhattan distance search radius from a
        given 2D query point.

        We pre-assign lane segment IDs to indices inside a big lookup array, with precomputed
        hallucinated lane polygon extents.

        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            query_search_range_manhattan: search radius along axes

        Returns:
            lane_ids: lane segment IDs that live within a bubble
        """
        query_min_x = query_x - query_search_range_manhattan
        query_max_x = query_x + query_search_range_manhattan
        query_min_y = query_y - query_search_range_manhattan
        query_max_y = query_y + query_search_range_manhattan

        overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
            self.city_halluc_bbox_table[city_name],
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        if len(overlap_indxs) == 0:
            return []

        neighborhood_lane_ids: List[int] = []
        for overlap_idx in overlap_indxs:
            lane_segment_id = self.city_halluc_tableidx_to_laneid_map[city_name][str(overlap_idx)]
            neighborhood_lane_ids.append(lane_segment_id)

        return neighborhood_lane_ids

    def get_lane_segment_predecessor_ids(self, lane_segment_id: int, city_name: str) -> List[int]:
        """
        Get land id for the lane predecessor of the specified lane_segment_id

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            predecessor_ids: list of integers, representing lane segment IDs of predecessors
        """
        predecessor_ids = self.city_lane_centerlines_dict[city_name][lane_segment_id].predecessors
        return predecessor_ids

    def get_lane_segment_successor_ids(self, lane_segment_id: int, city_name: str) -> Optional[List[int]]:
        """
        Get land id for the lane sucessor of the specified lane_segment_id

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            successor_ids: list of integers, representing lane segment IDs of successors
        """
        successor_ids = self.city_lane_centerlines_dict[city_name][lane_segment_id].successors
        return successor_ids

    def get_lane_segment_adjacent_ids(self, lane_segment_id: int, city_name: str) -> List[Optional[int]]:
        """
        Get land id for the lane adjacent left/right neighbor of the specified lane_segment_id

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            adjacent_ids: list of integers, representing lane segment IDs of adjacent
                            left/right neighbor lane segments
        """
        r_neighbor = self.city_lane_centerlines_dict[city_name][lane_segment_id].r_neighbor_id
        l_neighbor = self.city_lane_centerlines_dict[city_name][lane_segment_id].l_neighbor_id
        adjacent_ids = [r_neighbor, l_neighbor]
        return adjacent_ids

    def get_lane_segment_centerline(self, lane_segment_id: int, city_name: str) -> np.ndarray:
        """
        We return a 3D centerline for any particular lane segment.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            lane_centerline: Numpy array of shape (N,3)
        """
        lane_centerline = self.city_lane_centerlines_dict[city_name][lane_segment_id].centerline
        if len(lane_centerline[0]) == 2:
            lane_centerline = self.append_height_to_2d_city_pt_cloud(lane_centerline, city_name)

        return lane_centerline

    def get_lane_segment_polygon(self, lane_segment_id: int, city_name: str) -> np.ndarray:
        """
        Hallucinate a 3d lane polygon based around the centerline. We rely on the average
        lane width within our cities to hallucinate the boundaries. We rely upon the
        rasterized maps to provide heights to points in the xy plane.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            lane_polygon: Array of polygon boundary (K,3), with identical and last boundary points
        """
        lane_centerline = self.city_lane_centerlines_dict[city_name][lane_segment_id].centerline
        lane_polygon = centerline_to_polygon(lane_centerline[:, :2])
        return self.append_height_to_2d_city_pt_cloud(lane_polygon, city_name)

    def lane_is_in_intersection(self, lane_segment_id: int, city_name: str) -> bool:
        """
        Check if the specified lane_segment_id falls within an intersection

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            is_intersection: indicating if lane segment falls within an
                intersection
        """
        return self.city_lane_centerlines_dict[city_name][lane_segment_id].is_intersection

    def get_lane_turn_direction(self, lane_segment_id: int, city_name: str) -> str:
        """
        Get left/right/none direction of the specified lane_segment_id

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            turn_direction: string, can be 'RIGHT', 'LEFT', or 'NONE'
        """
        return self.city_lane_centerlines_dict[city_name][lane_segment_id].turn_direction

    def lane_has_traffic_control_measure(self, lane_segment_id: int, city_name: str) -> bool:
        """
        You can have an intersection without a control measure.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            has_traffic_control: indicating if lane segment has a
                traffic control measure
        """
        return self.city_lane_centerlines_dict[city_name][lane_segment_id].has_traffic_control

    def remove_extended_predecessors(
        self, lane_seqs: List[List[int]], xy: np.ndarray, city_name: str
    ) -> List[List[int]]:
        """
        Remove lane_ids which are obtained by finding way too many predecessors from lane sequences.
        If any lane id is an occupied lane id for the first coordinate of the trajectory, ignore all the
        lane ids that occured before that

        Args:
            lane_seqs: List of list of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
            xy: trajectory coordinates
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            filtered_lane_seq (list of list of integers): List of list of lane ids obtained after filtering
        """
        filtered_lane_seq = []
        occupied_lane_ids = self.get_lane_segments_containing_xy(xy[0, 0], xy[0, 1], city_name)
        for lane_seq in lane_seqs:
            for i in range(len(lane_seq)):
                if lane_seq[i] in occupied_lane_ids:
                    new_lane_seq = lane_seq[i:]
                    break
                new_lane_seq = lane_seq
            filtered_lane_seq.append(new_lane_seq)
        return filtered_lane_seq

    def get_cl_from_lane_seq(self, lane_seqs: Iterable[List[int]], city_name: str) -> List[np.ndarray]:
        """Get centerlines corresponding to each lane sequence in lane_sequences

        Args:
            lane_seqs: Iterable of sequence of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            candidate_cl: list of numpy arrays for centerline corresponding to each lane sequence
        """

        candidate_cl = []
        for lanes in lane_seqs:
            curr_candidate_cl = np.empty((0, 2))
            for curr_lane in lanes:
                curr_candidate = self.get_lane_segment_centerline(curr_lane, city_name)[:, :2]
                curr_candidate_cl = np.vstack((curr_candidate_cl, curr_candidate))
            candidate_cl.append(curr_candidate_cl)
        return candidate_cl

    def get_candidate_centerlines_for_traj(
        self,
        xy: np.ndarray,
        city_name: str,
        viz: bool = False,
        max_search_radius: float = 50.0,
    ) -> List[np.ndarray]:
        """Get centerline candidates upto a threshold. .

        Algorithm:
        1. Take the lanes in the bubble of last obs coordinate
        2. Extend before and after considering all possible candidates
        3. Get centerlines with max distance along centerline

        Args:
            xy: trajectory of shape (N, 2).
            city_name
            viz: Visualize

        Returns:
            candidate_centerlines: List of candidate centerlines
        """

        # Get all lane candidates within a bubble
        manhattan_threshold = 2.5
        curr_lane_candidates = self.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name, manhattan_threshold)

        # Keep expanding the bubble until at least 1 lane is found
        while len(curr_lane_candidates) < 1 and manhattan_threshold < max_search_radius:
            manhattan_threshold *= 2
            curr_lane_candidates = self.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name, manhattan_threshold)

        assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

        # Set dfs threshold
        displacement = np.sqrt((xy[0, 0] - xy[-1, 0]) ** 2 + (xy[0, 1] - xy[-1, 1]) ** 2)
        dfs_threshold = displacement * 2.0

        # DFS to get all successor and predecessor candidates
        obs_pred_lanes: List[List[int]] = []
        for lane in curr_lane_candidates:
            candidates_future = self.dfs(lane, city_name, 0, dfs_threshold)
            candidates_past = self.dfs(lane, city_name, 0, dfs_threshold, True)

            # Merge past and future
            for past_lane_seq in candidates_past:
                for future_lane_seq in candidates_future:
                    assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate lanes past and future"
                    obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # Remove unnecessary extended predecessors
        obs_pred_lanes = self.remove_extended_predecessors(obs_pred_lanes, xy, city_name)

        # Getting candidate centerlines
        candidate_cl = self.get_cl_from_lane_seq(obs_pred_lanes, city_name)

        # Reduce the number of candidates based on distance travelled along the centerline
        candidate_centerlines = filter_candidate_centerlines(xy, candidate_cl)

        # If no candidate found using above criteria, take the onces along with travel is the maximum
        if len(candidate_centerlines) < 1:
            candidate_centerlines = get_centerlines_most_aligned_with_trajectory(xy, candidate_cl)

        if viz:
            plt.figure(0, figsize=(8, 7))
            for centerline_coords in candidate_centerlines:
                visualize_centerline(centerline_coords)
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "-",
                color="#d33e4c",
                alpha=1,
                linewidth=1,
                zorder=15,
            )

            final_x = xy[-1, 0]
            final_y = xy[-1, 1]

            plt.plot(final_x, final_y, "o", color="#d33e4c", alpha=1, markersize=7, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("Number of candidates = {}".format(len(candidate_centerlines)))
            plt.show()

        return candidate_centerlines

    def dfs(
        self,
        lane_id: int,
        city_name: str,
        dist: float = 0,
        threshold: float = 30,
        extend_along_predecessor: bool = False,
    ) -> List[List[int]]:
        """
        Perform depth first search over lane graph up to the threshold.

        Args:
            lane_id: Starting lane_id (Eg. 12345)
            city_name
            dist: Distance of the current path
            threshold: Threshold after which to stop the search
            extend_along_predecessor: if true, dfs over predecessors, else successors

        Returns:
            lanes_to_return (list of list of integers): List of sequence of lane ids
                Eg. [[12345, 12346, 12347], [12345, 12348]]

        """
        if dist > threshold:
            return [[lane_id]]
        else:
            traversed_lanes = []
            child_lanes = (
                self.get_lane_segment_predecessor_ids(lane_id, city_name)
                if extend_along_predecessor
                else self.get_lane_segment_successor_ids(lane_id, city_name)
            )
            if child_lanes is not None:
                for child in child_lanes:
                    centerline = self.get_lane_segment_centerline(child, city_name)
                    cl_length = LineString(centerline).length
                    curr_lane_ids = self.dfs(
                        child,
                        city_name,
                        dist + cl_length,
                        threshold,
                        extend_along_predecessor,
                    )
                    traversed_lanes.extend(curr_lane_ids)
            if len(traversed_lanes) == 0:
                return [[lane_id]]
            lanes_to_return = []
            for lane_seq in traversed_lanes:
                lanes_to_return.append(lane_seq + [lane_id] if extend_along_predecessor else [lane_id] + lane_seq)
            return lanes_to_return

    def draw_lane(self, lane_segment_id: int, city_name: str, legend: bool = False) -> None:
        """Draw the given lane.

        Args:
            lane_segment_id: lane ID
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            legend: True if legends specifying lane IDs are to shown as well

        """
        lane_segment_polygon = self.get_lane_segment_polygon(lane_segment_id, city_name)
        if legend:
            plt.plot(
                lane_segment_polygon[:, 0],
                lane_segment_polygon[:, 1],
                color="dimgray",
                label=lane_segment_id,
            )
        else:
            plt.plot(
                lane_segment_polygon[:, 0],
                lane_segment_polygon[:, 1],
                color="lightgrey",
            )
        plt.axis("equal")

    def get_lane_segments_containing_xy(self, query_x: float, query_y: float, city_name: str) -> List[int]:
        """

        Get the occupied lane ids, i.e. given (x,y), list those lane IDs whose hallucinated
        lane polygon contains this (x,y) query point.

        This function performs a "point-in-polygon" test.

        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            occupied_lane_ids: list of integers, representing lane segment IDs containing (x,y)
        """
        neighborhood_lane_ids = self.get_lane_ids_in_xy_bbox(query_x, query_y, city_name)

        occupied_lane_ids: List[int] = []
        if neighborhood_lane_ids is not None:
            for lane_id in neighborhood_lane_ids:
                lane_polygon = self.get_lane_segment_polygon(lane_id, city_name)
                inside = point_inside_polygon(
                    lane_polygon.shape[0],
                    lane_polygon[:, 0],
                    lane_polygon[:, 1],
                    query_x,
                    query_y,
                )
                if inside:
                    occupied_lane_ids += [lane_id]
        return occupied_lane_ids

    def plot_nearby_halluc_lanes(
        self,
        ax: plt.Axes,
        city_name: str,
        query_x: float,
        query_y: float,
        patch_color: str = "r",
        radius: float = 20,
    ) -> None:
        """
        Plot lane segment for nearby lanes of the specified x, y location

        Args:
            query_bbox: An array of shape (4,) representing a 2d axis-aligned bounding box, with order
                        [xmin,xmax,ymin,ymax]
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        """
        nearby_lane_ids = self.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, radius)
        for nearby_lane_id in nearby_lane_ids:
            halluc_lane_polygon = self.get_lane_segment_polygon(nearby_lane_id, city_name)
            plot_lane_segment_patch(halluc_lane_polygon, ax, color=patch_color, alpha=0.3)

    def find_local_lane_polygons(self, query_bbox: Tuple[float, float, float, float], city_name: str) -> np.ndarray:
        """
        Find land polygons within specified area

        Args:
            query_bbox: An array of shape (4,) representing a 2d axis-aligned bounding box, with order
                        [xmin,xmax,ymin,ymax]
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns
            local_lane_polygons: Array of arrays, representing local hallucinated lane polygons
        """
        lane_polygons = self.city_to_lane_polygons_dict[city_name]
        lane_bboxes = self.city_to_lane_bboxes_dict[city_name]
        xmin, xmax, ymin, ymax = query_bbox
        local_lane_polygons, _ = find_local_polygons(
            copy.deepcopy(lane_polygons),
            copy.deepcopy(lane_bboxes),
            xmin,
            xmax,
            ymin,
            ymax,
        )
        return local_lane_polygons

    def find_local_driveable_areas(self, query_bbox: Tuple[float, float, float, float], city_name: str) -> np.ndarray:
        """
        Find local driveable areas within specified area

        Args:
            query_bbox: An array of shape (4,) representing a 2d axis-aligned bounding box, with order
                        [xmin,xmax,ymin,ymax]
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns
            local_das: Array of arrays, representing local driveable area polygons
        """
        driveable_areas = self.city_to_driveable_areas_dict[city_name]
        da_bboxes = self.city_to_da_bboxes_dict[city_name]
        xmin, xmax, ymin, ymax = query_bbox
        local_das, _ = find_local_polygons(
            copy.deepcopy(driveable_areas),
            copy.deepcopy(da_bboxes),
            xmin,
            xmax,
            ymin,
            ymax,
        )
        return local_das

    def find_local_lane_centerlines(
        self,
        query_x: float,
        query_y: float,
        city_name: str,
        query_search_range_manhattan: float = 80.0,
    ) -> np.ndarray:
        """
        Find local lane centerline to the specified x,y location

        Args:
            query_x: x-coordinate of map query
            query_y: x-coordinate of map query
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns
            local_lane_centerlines: Array of arrays, representing an array of lane centerlines, each a polyline
        """
        lane_ids = self.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, query_search_range_manhattan)
        local_lane_centerlines = [self.get_lane_segment_centerline(lane_id, city_name) for lane_id in lane_ids]
        return np.array(local_lane_centerlines)
