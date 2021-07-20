# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>

"""
API for loading and Argoverse 2.0 maps. These include left and right lane boundaries,
instead of only lane centerlines, as was the case in Argoverse 1.0 and 1.1.

Separate map data (files) is provided for each log/scenario. This local map data represents
map entities that fall within some distance according to l-infinity norm from the trajectory
of the egovehicle (AV).
"""

import copy
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import Final

import numpy as np
from dataclasses import dataclass

from argoverse.data_loading.vector_map_v2_loader import point_arr_from_points_list_dict
from argoverse.utils.centerline_utils import convert_lane_boundaries3d_to_polygon3d
from argoverse.utils.dilation_utils import dilate_by_l2
from argoverse.utils.interpolate import interp_arc
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.sim2 import Sim2

_PathLike = Union[str, "os.PathLike[str]"]

ROI_ISOCONTOUR: Final[float] = 5.0  # in meters

logger = logging.getLogger(__name__)


class PedCrossing(NamedTuple):
    """Represents a pedestrian crossing (i.e. crosswalk) as two edges along its principal axis.

    Args:
        edge1: array of shape (N,3) representing one edge of the crosswalk.
        edge2: array of shape (N,3) representing the other edge of the crosswalk.
    """
    edge1: np.ndarray
    edge2: np.ndarray

    def get_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the two principal edges of the crosswalk, in 2d."""
        return (self.edge1[:, :2], self.edge2[:, :2])

    def __eq__(self, other: "PedCrossing") -> bool:
        """Check if two pedestrian crossing objects are equal, up to a tolerance."""
        return np.allclose(self.edge1, other.edge1) and np.allclose(self.edge2, other.edge2)


class LocalLaneMarking(NamedTuple):
    """Information about a lane marking, representing either the left or right boundary of a lane segment.

    Args:
        bound_type: type of marking that represents the lane boundary, e.g. "SOLID_WHITE" or "DASHED_YELLOW".
        src_lane_id: id of lane segment to which this lane marking belongs.
        bound_side: string representing which side of a lane segment this marking represents, i.e. "left" or "right".
        polyline: array of shape (N,3) representing the waypoints of the lane segment's marked boundary.
    """

    bound_type: str
    src_lane_id: int
    bound_side: str
    polyline: np.ndarray


@dataclass(frozen=False)
class VectorLaneSegment:
    """
    Args:
        id: unique identifier for this lane segment (guaranteed to be unique only within this local map).
        right_ln_bound: array of shape (M,3) representing the right lane boundary.
        right_bound_type: type of marking that represents the right lane boundary.
        r_neighbor_id: unique identifier of the lane segment representing this object's right neighbor.
        left_ln_bound: array of shape (N,3) representing the right lane boundary
        left_bound_type: type of marking that represents the left lane boundary.
        l_neighbor_id: unique identifier of the lane segment representing this object's left neighbor.
        predecessors: unique identifiers of lane segments that are predecessors of this object.
        successors: unique identifiers of lane segments that represent successor of this object.
        lane_type:
        polygon_boundary: array of shape (N,3) 
        is_intersection: boolean value representing whether or not this lane segment lies within an intersection.
        render_l_bound: boolean flag for visualization, indicating whether to render the left lane boundary.
        render_r_bound: boolean flag for visualization, indicating whether to render the right lane boundary.
    """

    id: Optional[int] = None
    right_ln_bound: Optional[np.ndarray] = None
    right_bound_type: Optional[str] = None
    r_neighbor_id: Optional[int] = None
    left_ln_bound: Optional[np.ndarray] = None
    left_bound_type: Optional[str] = None
    l_neighbor_id: Optional[int] = None
    predecessors: Optional[List[int]] = None
    successors: Optional[List[int]] = None
    lane_type: Optional[str] = None
    polygon_boundary: Optional[np.ndarray] = None
    is_intersection: Optional[bool] = None
    render_l_bound: Optional[bool] = True
    render_r_bound: Optional[bool] = True

    def get_left_lane_marking(self):
        """ """
        return LocalLaneMarking(self.left_bound_type, self.id, "left", self.left_ln_bound)

    def get_right_lane_marking(self):
        """ """
        return LocalLaneMarking(self.right_bound_type, self.id, "right", self.right_ln_bound)


class ArgoverseMapV2:
    """API to interact with a local map for a single log (within a single city).

    Nodes in the lane graph are lane segments. Edges in the lane graph provided the lane segment connectivity, via
    left and right neighbors and successors.

    Lane segments are parameterized by 3d waypoints representing their left and right boundaries.
    """

    def __init__(self, log_map_dirpath: _PathLike) -> None:
        """Initialize the Argoverse Map.

        Args:
           log_map_dirpath: e.g. "log_map_archive_gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__2020-07-21-Z1F0076____city_72406.json"
        """
        self._log_map_dirpath = log_map_dirpath
        log_id = Path(log_map_dirpath).stem
        logger.info("Loaded map for %s", log_id)
        vector_data_fnames = glob.glob(os.path.join(log_map_dirpath, "log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError("JSON file containing vector map data is missing.")
        vector_data_fname = vector_data_fnames[0]

        log_vector_map_fpath = os.path.join(log_map_dirpath, vector_data_fname)
        self._vector_data = read_json_file(log_vector_map_fpath)

        self._lane_segments_dict = self.__build_lane_segment_index()
        self._ped_crossings_list = self.__build_ped_crossing_index()
        self._drivable_areas_list = self.__build_drivable_area_index()

        # TODO: check if the ground surface file exists (will not exist for the forecasting data). Group into map dir per log.

        self.rasterized_da_roi_dict = self.__build_rasterized_driveable_area_roi_index()
        # self.city_rasterized_ground_height_dict = self.__build_city_ground_height_index()

    def __build_lane_segment_index(self) -> Dict[int, VectorLaneSegment]:
        """Build a lookup index of all lane segments that are local to this log/scenario.

        Note: predecessors are implicit and available by reversing the directed graph dictated by successors.

        Returns:
            vls_dict: mapping from lane segment ID to vector lane segment object, parameterized in 3d.
        """
        vls_dict: Dict[int, VectorLaneSegment] = {}
        for lane_segment in self._vector_data["lane_segments"]:
            right_ln_bound = point_arr_from_points_list_dict(lane_segment["right_lane_boundary"]["points"])
            left_ln_bound = point_arr_from_points_list_dict(lane_segment["left_lane_boundary"]["points"])
            lane_polygon = convert_lane_boundaries3d_to_polygon3d(right_ln_bound, left_ln_bound)

            if not (right_ln_bound.shape[1] == 3 and left_ln_bound.shape[1] == 3):
                raise RuntimeError("Boundary waypoints should be 3-dimensional.")

            # TODO: reverse directed graph on-the-fly to generate predecessors

            vls_id = lane_segment["id"]
            vls_dict[vls_id] = VectorLaneSegment(
                id=vls_id,
                right_ln_bound=right_ln_bound,
                right_bound_type=lane_segment["right_lane_mark_type"],
                r_neighbor_id=lane_segment["right_neighbor"],
                left_ln_bound=left_ln_bound,
                left_bound_type=lane_segment["left_lane_mark_type"],
                l_neighbor_id=lane_segment["left_neighbor"],
                successors=lane_segment["successors"],
                lane_type=lane_segment["lane_type"],
                polygon_boundary=lane_polygon,  # uses x,y,z
                is_intersection=lane_segment["is_intersection"],
            )

        return vls_dict

    def __build_ped_crossing_index(self) -> List[PedCrossing]:
        """Build a lookup index of all pedestrian crossings (i.e. crosswalks) that are local to this log/scenario.

        Note: the lookup index is simply a list, rather than a dictionary-based mapping, since pedestrian crossings
        are not part of a larger graph.

        Returns:
            lpcs: local pedestrian crossings.
        """
        lpcs: List[PedCrossing] = []

        for ped_crossing in self._vector_data["pedestrian_crossings"]:
            edge1 = point_arr_from_points_list_dict(ped_crossing["edge1"]["points"])
            edge2 = point_arr_from_points_list_dict(ped_crossing["edge2"]["points"])

            lpc = PedCrossing(edge1, edge2)
            lpcs.append(lpc)

        return lpcs

    def __build_drivable_area_index(self) -> List[np.ndarray]:
        """Build a lookup index of all drivable area polygons. Each polygon is represented by a Nx3 array of its vertices.

        Note: the first and last polygon vertex are identical (i.e. the first index is repeated).
        """
        das: List[np.ndarray] = []
        for da_data in self._vector_data["drivable_areas"]:
            da = point_arr_from_points_list_dict(da_data["area_boundary"]["points"])

            # append the first vertex to the end of vertex list
            da = np.vstack([da, da[0]])

            das.append(da)

        return das
        
    def get_scenario_vector_drivable_areas(self) -> List[np.ndarray]:
        """Fetch a list of polygons, whose union represents the drivable area for the log/scenario.

        Note: this function provides drivable areas in vector, not raster, format).
        """
        return self._drivable_areas_list


    def get_lane_segment_successor_ids(self, lane_segment_id: int) -> Optional[List[int]]:
        """Get lane id for the lane sucessor of the specified lane_segment_id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            successor_ids: list of integers, representing lane segment IDs of successors
        """
        successor_ids = self._lane_segments_dict[lane_segment_id].successors
        return successor_ids

    def get_lane_segment_left_neighbor_id(self, lane_segment_id: int) -> Optional[int]:
        """Get id of lane segment that is the left neighbor (if any exists) to the query lane segment id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            integer representing id of left neighbor to the query lane segment id, or None if no such neighbor exists.
        """
        return self._lane_segments_dict[lane_segment_id].l_neighbor_id

    def get_lane_segment_right_neighbor_id(self, lane_segment_id: int) -> Optional[int]:
        """Get id of lane segment that is the right neighbor (if any exists) to the query lane segment id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            integer representing id of right neighbor to the query lane segment id, or None if no such neighbor exists.
        """
        return self._lane_segments_dict[lane_segment_id].r_neighbor_id

    def get_scenario_lane_segment_ids(self) -> List[int]:
        """Get ids of all lane segments that are local to this log/scenario (according to l-infinity norm).

        Returns:
            list containing ids of local lane segments
        """
        return list(self._lane_segments_dict.keys())

    def get_lane_segment_centerline(self, lane_segment_id: int, city_name: str) -> np.ndarray:
        """We return an inferred 3D centerline for any particular lane segment by forming a ladder of left and right waypoints.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            lane_centerline: Numpy array of shape (N,3)
        """
        left_ln_bound = self._lane_segments_dict[lane_segment_id].left_ln_bound
        right_ln_bound = self._lane_segments_dict[lane_segment_id].right_ln_bound

        lane_centerline = ""  # TODO: add 3d linear interpolation fn

        return lane_centerline

    def get_lane_segment_polygon(self, lane_segment_id: int) -> np.ndarray:
        """Return an array contained coordinates of vertices that represent the polygon's boundary.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            lane_polygon: Array of polygon boundary (K,3), with identical and last boundary points
        """
        return self._lane_segments_dict[lane_segment_id].polygon_boundary

    def lane_is_in_intersection(self, lane_segment_id: int) -> bool:
        """
        Check if the specified lane_segment_id falls within an intersection

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            is_intersection: boolean indicating if the lane segment falls within an intersection
        """
        return self._lane_segments_dict[lane_segment_id].is_intersection

    def get_scenario_ped_crossings(self) -> List[PedCrossing]:
        """Return a list of all pedestrian crossing objects that are local to this log/scenario (by l-infity norm).

        Returns:
            lpcs: local pedestrian crossings
        """
        return self._ped_crossings_list

    def get_scenario_lane_segments(self) -> List[VectorLaneSegment]:
        """Return a list of all lane segments objects that are local to this log/scenario.

        Returns:
            vls_list: local lane segments
        """
        return list(self._lane_segments_dict.values())

    def __build_rasterized_driveable_area_roi_index(self) -> Dict[str, np.ndarray]:
        """Rasterize and return 3d vector drivable area as a 2d array, and dilate it by 5 meters, to return a region of interest mask.

        Note: This function provides "drivable area" and "region of interest" as binary segmentation masks in the bird's eye view.

        Returns:
            rasterized_da_dict: A dictionary with driveable area info. For example, includes da_matrix: Numpy array of
                    shape (M,N) representing binary values for driveable area
                    npyimage_Sim2_city: Sim(2) that produces takes point in city coordinates to numpy image/array coordinates:
                       p_npyimage  = npyimage_Sim2_city * p_city
        """
        rasterized_da_roi_dict: Dict[str, np.ndarray] = {}

        # TODO: just load this from disk, instead of having to compute on the fly.
        import math
        xmin = math.floor(min([ da[:,0].min() for da in self._drivable_areas_list]))
        ymin = math.floor(min([ da[:,1].min() for da in self._drivable_areas_list]))
        xmax = math.ceil(max([ da[:,0].max() for da in self._drivable_areas_list]))
        ymax = math.ceil(max([ da[:,1].max() for da in self._drivable_areas_list]))
        
        # TODO: choose the resolution of the rasterization, will affect image dimensions
        img_h = ymax - ymin + 1
        img_w = xmax - xmin + 1

        npyimg_Sim2_city = Sim2(R=np.eye(2), t=np.array([-xmin,-ymin]), s=1.0)

        # convert vertices for each polygon from a 3d array in city coordinates, to a 2d array in image/array coordinates.
        da_polygons_img = []
        for da_polygon_city in self._drivable_areas_list:

            da_polygon_img = npyimg_Sim2_city.transform_from(da_polygon_city[:,:2])
            da_polygon_img = np.round(da_polygon_img).astype(np.int32)
            da_polygons_img.append(da_polygon_img)

        # import pdb; pdb.set_trace()
        rasterized_da_roi_dict["da_mat"] = get_mask_from_polygons(da_polygons_img, img_h, img_w)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.flipud(mask))
        # plt.show()

        # npy_fpath = f"{self.extracted_map_dir}/{self.city_name}_{self.city_id}_driveable_area_mat_2020_07_13.npy"
        # rasterized_da_roi_dict["da_mat"] = np.load(npy_fpath)

        # initialize ROI as zero-level isocontour of drivable area, and the dilate to 5-meter isocontour
        roi_mat_init = copy.deepcopy(rasterized_da_roi_dict["da_mat"])
        rasterized_da_roi_dict["roi_mat"] = dilate_by_l2(roi_mat_init, dilation_thresh=ROI_ISOCONTOUR)

        return rasterized_da_roi_dict

    def __build_city_ground_height_index(self) -> Dict[str, np.ndarray]:
        """
        Build index of rasterized ground height.

        Returns:
            city_ground_height_index: a dictionary of dictionaries. A dictionary that stores the "ground_height_matrix" and also the
                    city_se2_pkl_image: SE(2) that produces takes point in pkl image to city
                    coordinates, e.g. p_city = city_Transformation_pklimage * p_pklimage
        """
        city_rasterized_ground_height_dict: Dict[str, np.ndarray] = {}

        Sim2_json_fpaths = glob.glob(os.path.join(self._log_map_dirpath, "*driveable_area_npyimage_Sim2_city*.json"))
        if not len(Sim2_json_fpaths) == 1:
            raise RuntimeError("Sim(2) mapping from city to image coords is missing")
        # city_rasterized_da_roi_dict["npyimage_Sim2_city"] = Sim2.from_json(sim2_json_fpath)

        npy_fpath = f"{self.extracted_map_dir}/{self.city_name}_{self.city_id}_ground_surface_mat_2020_07_13.npy"

        # load the file with rasterized values
        city_rasterized_ground_height_dict["ground_height"] = np.load(npy_fpath)

        sim2_json_fpath = f"{self.log_map_dirpath}/{self.city_name}_{self.city_id}_ground_surface_npyimage_Sim2_city_2020_07_13.json"
        city_rasterized_ground_height_dict["npyimage_Sim2_city"] = Sim2.from_json(sim2_json_fpath)

        return city_rasterized_ground_height_dict

    def get_rasterized_driveable_area(self) -> Tuple[np.ndarray, Sim2]:
        """Get the driveable area.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            city_se2_pkl_image: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        da_mat = self.city_rasterized_da_roi_dict["da_mat"]
        return (da_mat, self.city_rasterized_da_roi_dict["npyimage_Sim2_city"])

    def get_rasterized_roi(self) -> Tuple[np.ndarray, Sim2]:
        """Get the driveable area.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        roi_mat = self.city_rasterized_da_roi_dict["roi_mat"]
        return (roi_mat, self.city_rasterized_da_roi_dict["npyimage_Sim2_city"])

    def get_rasterized_ground_height(self) -> Tuple[np.ndarray, Sim2]:
        """
        Get ground height matrix along with se2 that convert to city coordinate

        Returns:
            ground_height_matrix
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        ground_height_mat = self.city_rasterized_ground_height_dict["ground_height"]
        return (ground_height_mat, self.city_rasterized_ground_height_dict["npyimage_Sim2_city"])


def get_mask_from_polygons(polygons: List[np.ndarray], img_h:int, img_w: int) -> np.ndarray:
    """Rasterize multiple polygons onto a single 2d array.

    Args:
        polygons: 
        img_h: height of the image to generate, in pixels
        img_w: width of the image to generate, in pixels

    Returns:
        mask: 2d array with 0/1 values representing a binary segmentation mask
    """
    from PIL import Image, ImageDraw
    mask_img = Image.new("L", size=(img_w, img_h), color=0)
    for polygon in polygons:
        polygon = [ tuple([x,y]) for (x,y) in polygon]

        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    
    mask = np.array(mask_img)
    return mask

