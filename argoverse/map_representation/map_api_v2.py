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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import Final

import numpy as np
from dataclasses import dataclass

from argoverse.utils.centerline_utils import convert_lane_boundaries_to_polygon
from argoverse.utils.dilation_utils import dilate_by_l2
from argoverse.utils.interpolate import interp_arc
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.mask_utils import get_mask_from_polygons
from argoverse.utils.sim2 import Sim2

_PathLike = Union[str, "os.PathLike[str]"]

GROUND_HEIGHT_THRESHOLD = 0.3  # 30 centimeters
ROI_ISOCONTOUR: Final[float] = 5.0  # in meters

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """Represents a single 3-d point."""

    x: float
    y: float
    z: float

    @property
    def xyz(self) -> np.ndarray:
        """Return (3,) vector"""
        return np.array([self.x, self.y, self.z])


@dataclass
class Polyline:
    """Represents an ordered point set with consecutive adjacency."""

    waypoints: List[Point]

    @property
    def xyz(self) -> np.ndarray:
        """Return (N,3) array representing ordered waypoint coordinates."""
        return np.vstack([wpt.xyz for wpt in self.waypoints])

    @classmethod
    def from_dict(cls, points_dict: List[Dict[str, float]]) -> "Polyline":
        """Generate object instance from dictionary read from JSON data.

        TODO: should we rename this as from_points_list_dict() ?
        """
        return cls(waypoints=[Point(x=v["x"], y=v["y"], z=v["z"]) for v in points_dict])


@dataclass
class DrivableArea:
    """Represents a single polygon, not a polyline."""

    id: int
    area_boundary: List[Point]

    @property
    def xyz(self) -> np.ndarray:
        """Return (N,3) array representing the ordered 3d coordinates of the polygon vertices."""
        return np.vstack([wpt.xyz for wpt in self.area_boundary])

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> "DrivableArea":
        """Generate object instance from dictionary read from JSON data."""
        point_list = [Point(x=v["x"], y=v["y"], z=v["z"]) for v in json_data["area_boundary"]["points"]]
        # append the first vertex to the end of vertex list
        point_list.append(point_list[0])

        return cls(id=json_data["id"], area_boundary=point_list)


class LaneType(str, Enum):
    VEHICLE: str = "VEHICLE"
    BIKE: str = "BIKE"
    BUS: str = "BUS"
    NON_VEHICLE: str = "NON_VEHICLE"


# TODO: determine clearer definition for `NON_VEHICLE` lane type.


class LaneMarkType(str, Enum):
    """Color and pattern of a painted lane marking, located on either the left or ride side of a lane segment."""

    DASH_SOLID_YELLOW: str = "DASH_SOLID_YELLOW"
    DASH_SOLID_WHITE: str = "DASH_SOLID_WHITE"
    DASHED_WHITE: str = "DASHED_WHITE"
    DASHED_YELLOW: str = "DASHED_YELLOW"
    DOUBLE_SOLID_YELLOW: str = "DOUBLE_SOLID_YELLOW"
    DOUBLE_SOLID_WHITE: str = "DOUBLE_SOLID_WHITE"
    DOUBLE_DASH_YELLOW: str = "DOUBLE_DASH_YELLOW"
    DOUBLE_DASH_WHITE: str = "DOUBLE_DASH_WHITE"
    SOLID_YELLOW: str = "SOLID_YELLOW"
    SOLID_WHITE: str = "SOLID_WHITE"
    SOLID_DASH_WHITE: str = "SOLID_DASH_WHITE"
    SOLID_DASH_YELLOW: str = "SOLID_DASH_YELLOW"
    SOLID_BLUE: str = "SOLID_BLUE"
    NONE: str = "NONE"


@dataclass
class PedestrianCrossing:
    """Represents a pedestrian crossing (i.e. crosswalk) as two edges along its principal axis.

    Args:
        edge1: array of shape (N,3) representing one edge of the crosswalk.
        edge2: array of shape (N,3) representing the other edge of the crosswalk.
    """

    id: int
    edge1: Polyline
    edge2: Polyline

    def get_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the two principal edges of the crosswalk, in 2d."""
        return (self.edge1.xyz[:, :2], self.edge2.xyz[:, :2])

    def __eq__(self, other: "PedestrianCrossing") -> bool:
        """Check if two pedestrian crossing objects are equal, up to a tolerance."""
        return np.allclose(self.edge1.xyz, other.edge1.xyz) and np.allclose(self.edge2.xyz, other.edge2.xyz)

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> "PedestrianCrossing":
        """Converts JSON data to a PedestrianCrossing object."""

        edge1 = Polyline.from_dict(json_data["edge1"]["points"])
        edge2 = Polyline.from_dict(json_data["edge2"]["points"])

        return PedestrianCrossing(id=json_data["id"], edge1=edge1, edge2=edge2)


class LocalLaneMarking(NamedTuple):
    """Information about a lane marking, representing either the left or right boundary of a lane segment.

    Args:
        bound_type: type of marking that represents the lane boundary, e.g. "SOLID_WHITE" or "DASHED_YELLOW".
        src_lane_id: id of lane segment to which this lane marking belongs.
        bound_side: string representing which side of a lane segment this marking represents, i.e. "left" or "right".
        polyline: array of shape (N,3) representing the waypoints of the lane segment's marked boundary.
    """

    bound_type: LaneMarkType
    src_lane_id: int
    bound_side: str
    polyline: np.ndarray


# TODO: (willqi) we should not name it left_lane_mark_type, just _left_mark_type. implicit it is a about a lane
# TODO: (willqi) DOUBLE_DASH_YELLOW was missing, SOLID_BLUE
# TODO: (willqi) re-dump maps, with pedestrian crossings as empty dict or list, instead of missing key
# TODO: add in lane turn direction attribute when re-dump the maps.
# TODO: decide if we want to retain the `predecessors` field.


@dataclass(frozen=False)
class LaneSegment:
    """Represents a singe lane segments within the Argoverse 2.0 maps.

    Args:
        id: unique identifier for this lane segment (guaranteed to be unique only within this local map).
        is_intersection: boolean value representing whether or not this lane segment lies within an intersection.
        lane_type: designation of which vehicle types may legally utilize this lane for travel.
        right_lane_boundary: array of shape (M,3) representing the right lane boundary.
        left_lane_boundary: array of shape (N,3) representing the right lane boundary
        right_mark_type: type of marking that represents the right lane boundary.
        left_mark_type: type of marking that represents the left lane boundary.
        predecessors: unique identifiers of lane segments that are predecessors of this object.
        successors: unique identifiers of lane segments that represent successor of this object.
            Note: this list will be empty if no successors exist.
        right_neighbor_id: unique identifier of the lane segment representing this object's right neighbor.
        left_neighbor_id: unique identifier of the lane segment representing this object's left neighbor.

        render_l_bound: boolean flag for visualization, indicating whether to render the left lane boundary.
        render_r_bound: boolean flag for visualization, indicating whether to render the right lane boundary.
    """

    id: int
    is_intersection: bool
    lane_type: LaneType
    right_lane_boundary: Polyline
    left_lane_boundary: Polyline
    right_mark_type: LaneMarkType
    left_mark_type: LaneMarkType
    successors: List[int]
    predecessors: Optional[List[int]] = None
    right_neighbor_id: Optional[int] = None
    left_neighbor_id: Optional[int] = None

    # polygon_boundary: Optional[np.ndarray] = None

    # render_l_bound: Optional[bool] = True
    # render_r_bound: Optional[bool] = True

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> "LaneSegment":
        """Convert JSON to a LaneSegment instance."""
        return cls(
            id=json_data["id"],
            lane_type=LaneType(json_data["lane_type"]),
            right_lane_boundary=Polyline.from_dict(json_data["right_lane_boundary"]["points"]),
            left_lane_boundary=Polyline.from_dict(json_data["left_lane_boundary"]["points"]),
            right_mark_type=LaneMarkType(json_data["right_lane_mark_type"]),
            left_mark_type=LaneMarkType(json_data["left_lane_mark_type"]),
            right_neighbor_id=json_data["right_neighbor"],
            left_neighbor_id=json_data["left_neighbor"],
            successors=json_data["successors"],
            is_intersection=json_data["is_intersection"],
        )

    def get_left_lane_marking(self):
        """ """
        return LocalLaneMarking(self.left_mark_type, self.id, "left", self.left_lane_boundary)

    def get_right_lane_marking(self):
        """ """
        return LocalLaneMarking(self.right_mark_type, self.id, "right", self.right_lane_boundary)

    @property
    def polygon_boundary(self) -> np.ndarray:
        """Extract coordinates of the polygon formed by the lane segment's left and right boundaries.

        Returns:
            polygon_boundary: array of shape (N,3)
        """
        return convert_lane_boundaries_to_polygon(self.right_lane_boundary.xyz, self.left_lane_boundary.xyz)


# TODO: (willqi) should be dictionaries for lane_segments, not lists, so we can do O(1) lookup


@dataclass
class RasterMapLayer:
    """Data sampled at points along a regular grid, and a mapping from grid array coordinates to city coordinates."""

    array: np.ndarray
    city_Sim2_array: Sim2


class GroundHeightLayer(RasterMapLayer):
    """Rasterized ground height map layer.

    Stores the "ground_height_matrix" and also the city_Sim2_npyimage: Sim(2) that produces takes point in numpy
    image to city coordinates, e.g. p_city = city_Transformation_pklimage * p_pklimage
    """

    @classmethod
    def from_file(cls, log_map_dirpath) -> "GroundHeightLayer":
        """ """
        # Sim2_json_fpaths = glob.glob(os.path.join(log_map_dirpath, "*driveable_area_npyimage_Sim2_city*.json"))
        # if not len(Sim2_json_fpaths) == 1:
        #     raise RuntimeError("Sim(2) mapping from city to image coords is missing")
        # # city_rasterized_da_roi_dict["npyimage_Sim2_city"] = Sim2.from_json(sim2_json_fpath)

        # npy_fpath = f"{self.extracted_map_dir}/{self.city_name}_{self.city_id}_ground_surface_mat_2020_07_13.npy"

        # # load the file with rasterized values
        # city_rasterized_ground_height_dict["ground_height"] = np.load(npy_fpath)

        # sim2_json_fpath = (
        #     f"{log_map_dirpath}/{self.city_name}_{self.city_id}_ground_surface_npyimage_Sim2_city_2020_07_13.json"
        # )
        # npyimage_Sim2_city = Sim2.from_json(sim2_json_fpath)

        return cls(array=None, city_Sim2_array=None)

    def get_ground_points_boolean(self, point_cloud: np.ndarray) -> np.ndarray:
        """Check whether each point is likely to be from the ground surface.

        Args:
            point_cloud: Numpy array of shape (N,3)

        Returns:
            is_ground_boolean_arr: Numpy array of shape (N,) where ith entry is True if the 3d point
                (e.g. a LiDAR return) is likely located on the ground surface.
        """
        ground_height_values = self.get_ground_height_at_xy(point_cloud)
        z = point_cloud[:, 2]
        near_ground = np.absolute(z - ground_height_values) <= GROUND_HEIGHT_THRESHOLD
        underground = z < ground_height_values
        is_ground_boolean_arr = near_ground | underground
        return is_ground_boolean_arr

    def get_ground_height_at_xy(self, point_cloud: np.ndarray) -> np.ndarray:
        """Get ground height for each of the xy locations for all points {(x,y,z)} in a point cloud.

        Args:
            point_cloud: Numpy array of shape (K,2) or (K,3)

        Returns:
            ground_height_values: Numpy array of shape (K,)
        """
        ground_height_mat, city_Sim2_array = self.get_rasterized_ground_height()

        # TODO: should not be rounded here, because we need to enforce scaled discretization.
        city_coords = np.round(point_cloud[:, :2]).astype(np.int64)

        npyimage_coords = city_Sim2_array.inverse().transform_point_cloud(city_coords)
        npyimage_coords = npyimage_coords.astype(np.int64)

        # TODO: verify if the code below is still needed.

        ground_height_values = np.full((npyimage_coords.shape[0]), np.nan)
        ind_valid_pts = (npyimage_coords[:, 1] < ground_height_mat.shape[0]) * (
            npyimage_coords[:, 0] < ground_height_mat.shape[1]
        )

        ground_height_values[ind_valid_pts] = ground_height_mat[
            npyimage_coords[ind_valid_pts, 1], npyimage_coords[ind_valid_pts, 0]
        ]

        return ground_height_values


class DrivableAreaMapLayer(RasterMapLayer):
    """Rasterized drivable area map layer."""

    @classmethod
    def from_vector_data(cls, drivable_areas: List[DrivableArea]) -> "DrivableAreaMapLayer":
        """
        TODO: should it accept Polygon instead?
        """

        # TODO: just load this from disk, instead of having to compute on the fly.
        xmin, ymin, xmax, ymax = compute_data_bounds(drivable_areas)

        # TODO: choose the resolution of the rasterization, will affect image dimensions
        img_h = ymax - ymin + 1
        img_w = xmax - xmin + 1

        npyimg_Sim2_city = Sim2(R=np.eye(2), t=np.array([-xmin, -ymin]), s=1.0)

        # convert vertices for each polygon from a 3d array in city coordinates, to a 2d array
        # in image/array coordinates.
        da_polygons_img = []
        for da_polygon_city in drivable_areas:

            da_polygon_img = npyimg_Sim2_city.transform_from(da_polygon_city.xyz[:, :2])
            da_polygon_img = np.round(da_polygon_img).astype(np.int32)
            da_polygons_img.append(da_polygon_img)

        da_mask = get_mask_from_polygons(da_polygons_img, img_h, img_w)

        return cls(array=da_mask, city_Sim2_array=npyimg_Sim2_city.inverse())


class RoiMapLayer(RasterMapLayer):
    """Rasterized Region of Interest (RoI) map layer."""

    @classmethod
    def from_drivable_area_layer(cls, drivable_area_layer: DrivableAreaMapLayer) -> "RoiMapLayer":
        """Rasterize and return 3d vector drivable area as a 2d array, and dilate it by 5 meters, to return a ROI mask.

        Note: This function provides "drivable area" and "region of interest" as binary segmentation masks in the
        bird's eye view.

        Returns:
            Driveable area info. includes da_matrix: Numpy array of shape (M,N) representing binary values for
                driveable area
                npyimage_Sim2_city: Sim(2) that produces takes point in city coordinates to numpy image/array
                    coordinates:
                    p_npyimage  = npyimage_Sim2_city * p_city
        """
        # initialize ROI as zero-level isocontour of drivable area, and the dilate to 5-meter isocontour
        roi_mat_init = copy.deepcopy(drivable_area_layer.array)
        roi_mask = dilate_by_l2(roi_mat_init, dilation_thresh=ROI_ISOCONTOUR)

        return cls(array=roi_mask, city_Sim2_array=drivable_area_layer.city_Sim2_array)


def compute_data_bounds(drivable_areas: List[DrivableArea]) -> Tuple[int, int, int, int]:
    """ """
    import math

    xmin = math.floor(min([da.xyz[:, 0].min() for da in drivable_areas]))
    ymin = math.floor(min([da.xyz[:, 1].min() for da in drivable_areas]))
    xmax = math.ceil(max([da.xyz[:, 0].max() for da in drivable_areas]))
    ymax = math.ceil(max([da.xyz[:, 1].max() for da in drivable_areas]))

    return xmin, ymin, xmax, ymax


@dataclass
class ArgoverseStaticMapV2:
    """API to interact with a local map for a single log (within a single city).

    Nodes in the lane graph are lane segments. Edges in the lane graph provided the lane segment connectivity, via
    left and right neighbors and successors.

    Lane segments are parameterized by 3d waypoints representing their left and right boundaries.
        Note: predecessors are implicit and available by reversing the directed graph dictated by successors.

    Args:
        log_id: unique identifier for log/scenario.
        vector_drivable_areas: drivable area polygons. Each polygon is represented by a Nx3 array of its vertices.
            Note: the first and last polygon vertex are identical (i.e. the first index is repeated).
        vector_lane_segments: lane segments that are local to this log/scenario. Consists of a mapping from
            lane segment ID to vector lane segment object, parameterized in 3d.
        vector_pedestrian_crossings: all pedestrian crossings (i.e. crosswalks) that are local to this log/scenario.
            Note: the lookup index is simply a list, rather than a dictionary-based mapping, since pedestrian crossings
            are not part of a larger graph.
        raster_drivable_area_layer: raster representation pf
        raster_roi_layer:
        raster_ground_height_layer: not provided for Motion Forecasting-specific scenarios/logs.
    """

    # TODO: make them all Dict[]
    # handle out-of-bounds lane segment ids with ValueError

    log_id: str
    vector_drivable_areas: List[DrivableArea]
    vector_lane_segments: Dict[int, LaneSegment]
    vector_pedestrian_crossings: List[PedestrianCrossing]
    raster_drivable_area_layer: Optional[DrivableAreaMapLayer]
    raster_roi_layer: Optional[RoiMapLayer]
    raster_ground_height_layer: Optional[GroundHeightLayer]

    # TODO (will): reverse directed graph on-the-fly to generate predecessors?

    @classmethod
    def from_json(cls, log_map_dirpath: _PathLike, build_raster: bool = False) -> "ArgoverseV2StaticMap":
        """Initialize the Argoverse Map for a specific log from JSON data.

        Args:
           log_map_dirpath: path to directory containing log-scenario-specific maps,
               e.g. "log_map_archive_gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__2020-07-21-Z1F0076____city_72406.json"
            build_raster: whether to rasterize drivable areas, compute region of interest BEV binary segmentation,
                and to load raster ground height (when available).
        """
        log_id = Path(log_map_dirpath).stem

        logger.info("Loaded map for %s", log_id)
        vector_data_fnames = glob.glob(os.path.join(log_map_dirpath, "log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError("JSON file containing vector map data is missing.")
        vector_data_fname = vector_data_fnames[0]

        log_vector_map_fpath = os.path.join(log_map_dirpath, vector_data_fname)

        vector_data = read_json_file(log_vector_map_fpath)

        vector_drivable_areas = [DrivableArea.from_dict(da) for da in vector_data["drivable_areas"]]
        vector_lane_segments = {ls["id"]: LaneSegment.from_dict(ls) for ls in vector_data["lane_segments"]}

        if "pedestrian_crossings" not in vector_data:
            logger.error("Missing Pedestrian crossings!")
            vector_pedestrian_crossings = None
        else:
            vector_pedestrian_crossings = [
                PedestrianCrossing.from_dict(pc) for pc in vector_data["pedestrian_crossings"]
            ]

        if build_raster:
            # TODO: check if the ground surface file exists (will not exist for the forecasting data).
            # Group into map dir per log.
            raster_da_map_layer = DrivableAreaMapLayer.from_vector_data(vector_drivable_areas)
            raster_roi_layer = RoiMapLayer.from_drivable_area_layer(da_map_layer)
            raster_ground_height_layer = GroundHeightLayer.from_file(log_map_dirpath)

        else:
            raster_da_map_layer = None
            raster_roi_layer = None
            raster_ground_height_layer = None

        return cls(
            log_id=log_id,
            vector_drivable_areas=vector_drivable_areas,
            vector_lane_segments=vector_lane_segments,
            vector_pedestrian_crossings=vector_pedestrian_crossings,
            raster_drivable_area_layer=raster_da_map_layer,
            raster_roi_layer=raster_roi_layer,
            raster_ground_height_layer=raster_ground_height_layer,
        )

    def get_scenario_vector_drivable_areas(self) -> List[np.ndarray]:
        """Fetch a list of polygons, whose union represents the drivable area for the log/scenario.

        Note: this function provides drivable areas in vector, not raster, format).
        """
        return self.vector_drivable_areas

    def get_lane_segment_successor_ids(self, lane_segment_id: int) -> Optional[List[int]]:
        """Get lane id for the lane sucessor of the specified lane_segment_id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            successor_ids: list of integers, representing lane segment IDs of successors
        """
        successor_ids = self.vector_lane_segments[lane_segment_id].successors
        return successor_ids

    def get_lane_segment_left_neighbor_id(self, lane_segment_id: int) -> Optional[int]:
        """Get id of lane segment that is the left neighbor (if any exists) to the query lane segment id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            integer representing id of left neighbor to the query lane segment id, or None if no such neighbor exists.
        """
        return self.vector_lane_segments[lane_segment_id].left_neighbor_id

    def get_lane_segment_right_neighbor_id(self, lane_segment_id: int) -> Optional[int]:
        """Get id of lane segment that is the right neighbor (if any exists) to the query lane segment id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            integer representing id of right neighbor to the query lane segment id, or None if no such neighbor exists.
        """
        return self.vector_lane_segments[lane_segment_id].right_neighbor_id

    def get_scenario_lane_segment_ids(self) -> List[int]:
        """Get ids of all lane segments that are local to this log/scenario (according to l-infinity norm).

        Returns:
            list containing ids of local lane segments
        """
        return list(self.vector_lane_segments.keys())

    def get_lane_segment_centerline(self, lane_segment_id: int, city_name: str) -> np.ndarray:
        """Infer a 3D centerline for any particular lane segment by forming a ladder of left and right waypoints.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            lane_centerline: Numpy array of shape (N,3)
        """
        left_ln_bound = self.vector_lane_segments[lane_segment_id].left_lane_boundary
        right_ln_bound = self.vector_lane_segments[lane_segment_id].right_lane_boundary

        lane_centerline = ""  # TODO: add 3d linear interpolation fn

        return lane_centerline

    def get_lane_segment_polygon(self, lane_segment_id: int) -> np.ndarray:
        """Return an array contained coordinates of vertices that represent the polygon's boundary.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            lane_polygon: Array of polygon boundary (K,3), with identical and last boundary points
        """
        return self.vector_lane_segments[lane_segment_id].polygon_boundary

    def lane_is_in_intersection(self, lane_segment_id: int) -> bool:
        """
        Check if the specified lane_segment_id falls within an intersection

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            is_intersection: boolean indicating if the lane segment falls within an intersection
        """
        return self.vector_lane_segments[lane_segment_id].is_intersection

    def get_scenario_ped_crossings(self) -> List[PedestrianCrossing]:
        """Return a list of all pedestrian crossing objects that are local to this log/scenario (by l-infity norm).

        Returns:
            lpcs: local pedestrian crossings
        """
        return self.vector_pedestrian_crossings

    def get_scenario_lane_segments(self) -> List[LaneSegment]:
        """Return a list of all lane segments objects that are local to this log/scenario.

        Returns:
            vls_list: local lane segments
        """
        return list(self.vector_lane_segments.values())

    def remove_ground_surface(self) -> np.ndarray:
        """ """
        raise NotImplementedError("")

    def remove_non_driveable_area_points(self) -> np.ndarray:
        """ """
        raise NotImplementedError("")

    def remove_non_roi_points(self) -> np.ndarray:
        """ """
        raise NotImplementedError("")

    def get_ground_points_boolean(self) -> np.ndarray:
        """ """
        raise NotImplementedError("")

    def get_rasterized_driveable_area(self) -> Tuple[np.ndarray, Sim2]:
        """Get the driveable area along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            city_Sim2_npyimage: Sim(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        return self.raster_drivable_area_layer.array, self.raster_drivable_area_layer.city_Sim2_array

    def get_rasterized_roi(self) -> Tuple[np.ndarray, Sim2]:
        """Get the driveable area along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            city_Sim2_npyimage: Sim(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        return self.raster_roi_layer.array, self.raster_roi_layer.city_Sim2_array

    def get_rasterized_ground_height(self) -> Tuple[np.ndarray, Sim2]:
        """Get ground height matrix along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            ground_height_matrix
            city_Sim2_npyimage: Sim(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        return self.raster_ground_height_layer.array, self.raster_ground_height_layer.city_Sim2_array
