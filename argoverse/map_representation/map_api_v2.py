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
import math
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import Final

import numpy as np
from dataclasses import dataclass

import argoverse.utils.dilation_utils as dilation_utils
import argoverse.utils.interpolate as interp_utils
import argoverse.utils.json_utils as json_utils
from argoverse.utils.centerline_utils import convert_lane_boundaries_to_polygon
from argoverse.utils.mask_utils import get_mask_from_polygons
from argoverse.utils.sim2 import Sim2

_PathLike = Union[str, "os.PathLike[str]"]

GROUND_HEIGHT_THRESHOLD_M: Final[float] = 0.3  # 30 centimeters
ROI_ISOCONTOUR_M: Final[float] = 5.0  # in meters

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

    def __eq__(self, other: "Point") -> bool:
        """Check for equality with another Point object."""
        return self.x == other.x and self.y == other.y and self.z == other.z


@dataclass
class Polyline:
    """Represents an ordered point set with consecutive adjacency."""

    waypoints: List[Point]

    @property
    def xyz(self) -> np.ndarray:
        """Return (N,3) array representing ordered waypoint coordinates."""
        return np.vstack([wpt.xyz for wpt in self.waypoints])

    @classmethod
    def from_dict_list(cls, json_data: List[Dict[str, float]]) -> "Polyline":
        """Generate object instance from list of dictionaries, read from JSON data.

        TODO: should we rename this as from_points_list_dict() ?
        """
        return cls(waypoints=[Point(x=v["x"], y=v["y"], z=v["z"]) for v in json_data])

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Polyline":
        """Generate object instance from a (N,3) Numpy array.

        Args:
            array: array of shape (N,3) representing N ordered three-dimensional points.
        """
        return cls(waypoints=[Point(*pt.tolist()) for pt in array])

    def __eq__(self, other: "Polyline") -> bool:
        """Check for equality with another Polyline object."""
        if len(self.waypoints) != len(other.waypoints):
            return False

        return all([wpt == wpt_ for wpt, wpt_ in zip(self.waypoints, other.waypoints)])

    def __len__(self) -> int:
        """Returns the number of waypoints in the polyline."""
        return len(self.waypoints)


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
    """Describes the sorts of objects that may use the lane for travel."""

    VEHICLE: str = "VEHICLE"
    BIKE: str = "BIKE"
    BUS: str = "BUS"
    NON_VEHICLE: str = "NON_VEHICLE"


# TODO: determine clearer definition for `NON_VEHICLE` lane type.


class LaneMarkType(str, Enum):
    """Color and pattern of a painted lane marking, located on either the left or ride side of a lane segment.

    The `NONE` type indicates that lane boundary is not marked by any paint; its extent should be implicitly inferred.
    """

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
        edge1: 3d polyline representing one edge of the crosswalk, with 2 waypoints.
        edge2: 3d polyline representing the other edge of the crosswalk, with 2 waypoints.
    """

    id: int
    edge1: Polyline
    edge2: Polyline

    def get_edges_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the two principal edges of the crosswalk, in 2d.

        Returns:
            edge1: array of shape (2,2)
            edge2: array of shape (2,2)
        """
        return (self.edge1.xyz[:, :2], self.edge2.xyz[:, :2])

    def __eq__(self, other: "PedestrianCrossing") -> bool:
        """Check if two pedestrian crossing objects are equal, up to a tolerance."""
        return np.allclose(self.edge1.xyz, other.edge1.xyz) and np.allclose(self.edge2.xyz, other.edge2.xyz)

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> "PedestrianCrossing":
        """Generate a PedestrianCrossing object from a dictionary read from JSON data."""
        edge1 = Polyline.from_dict_list(json_data["edge1"]["points"])
        edge2 = Polyline.from_dict_list(json_data["edge2"]["points"])

        return PedestrianCrossing(id=json_data["id"], edge1=edge1, edge2=edge2)

    @property
    def polygon(self) -> np.ndarray:
        """Return the vertices of the polygon representing the pedestrian crossing.

        Returns:
            array of shape (N,3) representing vertices. The first and last vertex that are provided are identical.
        """
        v0, v1 = self.edge1.xyz
        v2, v3 = self.edge2.xyz
        return np.array([v0, v1, v3, v2, v0])


class LocalLaneMarking(NamedTuple):
    """Information about a lane marking, representing either the left or right boundary of a lane segment.

    Args:
        mark_type: type of marking that represents the lane boundary, e.g. "SOLID_WHITE" or "DASHED_YELLOW".
        src_lane_id: id of lane segment to which this lane marking belongs.
        bound_side: string representing which side of a lane segment this marking represents, i.e. "left" or "right".
        polyline: array of shape (N,3) representing the waypoints of the lane segment's marked boundary.
    """

    mark_type: LaneMarkType
    src_lane_id: int
    bound_side: str
    polyline: np.ndarray


# TODO: decide if we want to retain the `predecessors` field.


@dataclass(frozen=False)
class LaneSegment:
    """Vector representation of a single lane segment within a log-specific Argoverse 2.0 map.

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

    # for rendering
    render_l_bound: Optional[bool] = True
    render_r_bound: Optional[bool] = True

    # polygon_boundary: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> "LaneSegment":
        """Convert JSON to a LaneSegment instance."""
        return cls(
            id=json_data["id"],
            lane_type=LaneType(json_data["lane_type"]),
            right_lane_boundary=Polyline.from_dict_list(json_data["right_lane_boundary"]["points"]),
            left_lane_boundary=Polyline.from_dict_list(json_data["left_lane_boundary"]["points"]),
            right_mark_type=LaneMarkType(json_data["right_lane_mark_type"]),
            left_mark_type=LaneMarkType(json_data["left_lane_mark_type"]),
            right_neighbor_id=json_data["right_neighbor"],
            left_neighbor_id=json_data["left_neighbor"],
            successors=json_data["successors"],
            is_intersection=json_data["is_intersection"],
        )

    def get_left_lane_marking(self) -> LocalLaneMarking:
        """ """
        return LocalLaneMarking(
            mark_type=self.left_mark_type, src_lane_id=self.id, bound_side="left", polyline=self.left_lane_boundary.xyz
        )

    def get_right_lane_marking(self) -> LocalLaneMarking:
        """ """
        return LocalLaneMarking(
            mark_type=self.right_mark_type,
            src_lane_id=self.id,
            bound_side="right",
            polyline=self.right_lane_boundary.xyz,
        )

    @property
    def polygon_boundary(self) -> np.ndarray:
        """Extract coordinates of the polygon formed by the lane segment's left and right boundaries.

        Returns:
            polygon_boundary: array of shape (N,3)
        """
        return convert_lane_boundaries_to_polygon(self.right_lane_boundary.xyz, self.left_lane_boundary.xyz)

    def within_l_infinity_norm_radius(self, query_center: np.ndarray, search_radius: float) -> bool:
        """Whether any waypoint of lane boundaries falls within search_radius meters of query center, by l-infty norm.

        Could have very long segment, with endpoints and all waypoints outside of radius, therefore cannot check just
        its endpoints.

        Args:
            center: array of shape (3,) representing 3d coordinates of query center.
            search_radius:

        Returns:
            whether the lane segment has any waypoint within search_radius meters of the query center.
        """
        from argoverse.utils.infinity_norm_utils import has_pts_in_infty_norm_radius

        WPT_INFTY_NORM_INTERP_NUM = 50

        try:
            right_ln_bnd_interp = interp_utils.interp_arc(
                t=WPT_INFTY_NORM_INTERP_NUM,
                px=self.right_lane_boundary.xyz[:, 0],
                py=self.right_lane_boundary.xyz[:, 1],
            )
            left_ln_bnd_interp = interp_utils.interp_arc(
                t=WPT_INFTY_NORM_INTERP_NUM, px=self.left_lane_boundary.xyz[:, 0], py=self.left_lane_boundary.xyz[:, 1]
            )
        except Exception as e:
            print("Interpolation attempt failed!")
            logging.exception(f"Interpolation failed")
            # 1-point line segments will cause trouble later
            right_ln_bnd_interp = self.right_lane_boundary.xyz[:, :2]
            left_ln_bnd_interp = self.left_lane_boundary.xyz[:, :2]

        left_in_bounds = has_pts_in_infty_norm_radius(right_ln_bnd_interp, query_center, search_radius)
        right_in_bounds = has_pts_in_infty_norm_radius(left_ln_bnd_interp, query_center, search_radius)
        return left_in_bounds or right_in_bounds


@dataclass
class RasterMapLayer:
    """Data sampled at points along a regular grid, and a mapping from city coordinates to grid array coordinates."""

    array: np.ndarray
    array_Sim2_city: Sim2

    def get_raster_values_at_coords(self, point_cloud: np.ndarray, fill_value: float) -> np.ndarray:
        """Index into a raster grid and extract values corresponding to city coordinates.

        Note: a conversion is required between city coordinates and raster grid coordinates, via Sim(2).

        Args:
            point_cloud: array of shape (N,2) or (N,3) representing coordinates in the city coordinate frame.
            fill_value: float representing default "raster" return value for out-of-bounds queries.

        Returns:
            raster_values: array of shape (N,) representing raster values at the N query coordinates.
        """
        # Note: we do NOT round here, because we need to enforce scaled discretization.
        city_coords = point_cloud[:, :2]

        npyimage_coords = self.array_Sim2_city.transform_point_cloud(city_coords)
        npyimage_coords = npyimage_coords.astype(np.int64)

        # out of bounds values will default to the fill value, and will not be indexed into the array.
        # index in at (x,y) locations, which are (y,x) in the image
        raster_values = np.full((npyimage_coords.shape[0]), fill_value)
        # generate boolean array indicating whether the value at each index represents a valid coordinate.
        ind_valid_pts = (
            (npyimage_coords[:, 1] >= 0)
            * (npyimage_coords[:, 1] < self.array.shape[0])
            * (npyimage_coords[:, 0] >= 0)
            * (npyimage_coords[:, 0] < self.array.shape[1])
        )
        raster_values[ind_valid_pts] = self.array[npyimage_coords[ind_valid_pts, 1], npyimage_coords[ind_valid_pts, 0]]
        return raster_values


class GroundHeightLayer(RasterMapLayer):
    """Rasterized ground height map layer.

    Stores the "ground_height_matrix" and also the array_Sim2_city: Sim(2) that produces takes point in city
    coordinates to numpy image/matrix coordinates, e.g. p_npyimage = array_Transformation_city * p_city
    """

    @classmethod
    def from_file(cls, log_map_dirpath: _PathLike) -> "GroundHeightLayer":
        """Load ground height values (w/ values at 30 cm resolution) from .npy file, and associated Sim(2) mapping.

        Note: ground height values are stored on disk as a float16 2d-array, but cast to float32 once loaded for
        compatibility with matplotlib.

        Args:
            log_map_dirpath: path to directory which contains map files associated with one specific log/scenario.
        """
        ground_height_npy_fpaths = glob.glob(os.path.join(log_map_dirpath, "*_ground_height_surface____*.npy"))
        if not len(ground_height_npy_fpaths) == 1:
            raise RuntimeError("Raster ground height layer file is missing")

        Sim2_json_fpaths = glob.glob(os.path.join(log_map_dirpath, "*___img_Sim2_city.json"))
        if not len(Sim2_json_fpaths) == 1:
            raise RuntimeError("Sim(2) mapping from city to image coordinates is missing")

        # load the file with rasterized values
        ground_height_array = np.load(ground_height_npy_fpaths[0])

        # TODO: do we prefer name `npyimage_Sim2_city` ?
        array_Sim2_city = Sim2.from_json(Sim2_json_fpaths[0])

        return cls(array=ground_height_array.astype(np.float32), array_Sim2_city=array_Sim2_city)

    def get_ground_points_boolean(self, point_cloud: np.ndarray) -> np.ndarray:
        """Check whether each 3d point is likely to be from the ground surface.

        Args:
            point_cloud: Numpy array of shape (N,3)

        Returns:
            is_ground_boolean_arr: Numpy array of shape (N,) where ith entry is True if the 3d point
                (e.g. a LiDAR return) is likely located on the ground surface.
        """
        if point_cloud.shape[1] != 3:
            raise ValueError("3-dimensional points must be provided to classify them as `ground` with the map.")

        ground_height_values = self.get_ground_height_at_xy(point_cloud)
        z = point_cloud[:, 2]
        near_ground = np.absolute(z - ground_height_values) <= GROUND_HEIGHT_THRESHOLD_M
        underground = z < ground_height_values
        is_ground_boolean_arr = near_ground | underground
        return is_ground_boolean_arr

    def get_rasterized_ground_height(self) -> Tuple[np.ndarray, Sim2]:
        """Get ground height matrix along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            ground_height_matrix:
            array_Sim2_city: Sim(2) that produces takes point in city coordinates to image coordinates, e.g.
                    p_image = image_Transformation_city * p_city
        """
        return self.array, self.array_Sim2_city

    def get_ground_height_at_xy(self, point_cloud: np.ndarray) -> np.ndarray:
        """Get ground height for each of the xy locations for all points {(x,y,z)} in a point cloud.

        Args:
            point_cloud: Numpy array of shape (K,2) or (K,3)

        Returns:
            ground_height_values: Numpy array of shape (K,)
        """
        ground_height_values = self.get_raster_values_at_coords(point_cloud, fill_value=np.nan)
        return ground_height_values


class DrivableAreaMapLayer(RasterMapLayer):
    """Rasterized drivable area map layer."""

    @classmethod
    def from_vector_data(cls, drivable_areas: List[DrivableArea]) -> "DrivableAreaMapLayer":
        """
        TODO: should it accept List[Polygon] instead?
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

        return cls(array=da_mask, array_Sim2_city=npyimg_Sim2_city)


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
        roi_mask = dilation_utils.dilate_by_l2(roi_mat_init, dilation_thresh=ROI_ISOCONTOUR_M)

        return cls(array=roi_mask, array_Sim2_city=drivable_area_layer.array_Sim2_city)


def compute_data_bounds(drivable_areas: List[DrivableArea]) -> Tuple[int, int, int, int]:
    """Find the minimum and maximum coordinates along the x and y axes for a set of drivable areas.

    Args:
        drivable_areas: list of drivable area objects, defined in the city coordinate frame.

    Returns:
        xmin: float representing minimum x-coordinate of any vertex of any provided drivable area.
        ymin: float representing minimum y-coordinate, as above.
        xmax: float representing maximum x-coordinate, as above.
        ymax: float representing maximum y-coordinate, as above.
    """
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
        raster_drivable_area_layer: 2d raster representation of drivable area segmentation.
        raster_roi_layer: 2d raster representation of region of interest segmentation.
        raster_ground_height_layer: not provided for Motion Forecasting-specific scenarios/logs.
    """

    # TODO: make them all Dict[], instead of List[], for consistency?
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

        Note: the ground height surface file and associated coordinate mapping is not provided for the
        2.0 Motion Forecasting dataset, so `build_raster` defaults to False. If raster functionality is
        desired, users should pass `build_raster` to True (e.g. for the Sensor Datasets and Map Change Datasets).

        Args:
           log_map_dirpath: path to directory containing log-scenario-specific maps,
               e.g. "log_map_archive_gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__2020-07-21-Z1F0076____city_72406.json"
            build_raster: whether to rasterize drivable areas, compute region of interest BEV binary segmentation,
                and to load raster ground height from disk (when available).
        """
        log_id = Path(log_map_dirpath).stem

        logger.info("Loaded map for %s", log_id)
        vector_data_fnames = glob.glob(os.path.join(log_map_dirpath, "log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError("JSON file containing vector map data is missing.")
        vector_data_fname = vector_data_fnames[0]

        log_vector_map_fpath = os.path.join(log_map_dirpath, vector_data_fname)

        vector_data = json_utils.read_json_file(log_vector_map_fpath)

        vector_drivable_areas = [DrivableArea.from_dict(da) for da in vector_data["drivable_areas"]]
        vector_lane_segments = {ls["id"]: LaneSegment.from_dict(ls) for ls in vector_data["lane_segments"]}

        if "pedestrian_crossings" not in vector_data:
            logger.error("Missing Pedestrian crossings!")
            vector_pedestrian_crossings = []
        else:
            vector_pedestrian_crossings = [
                PedestrianCrossing.from_dict(pc) for pc in vector_data["pedestrian_crossings"]
            ]

        # avoid file I/O and polygon rasterization when not needed
        raster_da_map_layer = DrivableAreaMapLayer.from_vector_data(vector_drivable_areas) if build_raster else None
        raster_roi_layer = RoiMapLayer.from_drivable_area_layer(raster_da_map_layer) if build_raster else None
        raster_ground_height_layer = GroundHeightLayer.from_file(log_map_dirpath) if build_raster else None

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
            successor_ids: list of integers, representing lane segment IDs of successors. If there are no
                successor lane segments, then the list will be empty.
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
        left_ln_bound = self.vector_lane_segments[lane_segment_id].left_lane_boundary.xyz
        right_ln_bound = self.vector_lane_segments[lane_segment_id].right_lane_boundary.xyz

        # TODO (willqi): determine if we would like to use a fixed distance-based resolution for waypoint spacing.
        lane_centerline = interp_utils.compute_midpoint_line(
            left_ln_bnds=left_ln_bound,
            right_ln_bnds=right_ln_bound,
            num_interp_pts=interp_utils.NUM_CENTERLINE_INTERP_PTS,
        )
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

    def get_nearby_ped_crossings(self, search_radius: float, norm_type) -> List[PedestrianCrossing]:
        """
        Search radius defined in l-infinity norm or l2 norm
        """
        raise NotImplementedError("Yet to implement...")

    def get_scenario_lane_segments(self) -> List[LaneSegment]:
        """Return a list of all lane segments objects that are local to this log/scenario.

        Returns:
            vls_list: lane segments local to this scenario (any waypoint within 100m by L2 distance)
        """
        return list(self.vector_lane_segments.values())

    def get_nearby_lane_segments(self, query_center: np.ndarray, search_radius: float) -> List[LaneSegment]:
        """
        Args:
            query_center: numpy array of shape (2,) representing query_center

        Returns:
            vls_list: lane segments
        """
        scenario_lane_segments = self.get_scenario_lane_segments()
        return [ls for ls in scenario_lane_segments if ls.within_l_infinity_norm_radius(query_center, search_radius)]

    def remove_ground_surface(self) -> np.ndarray:
        """ """
        raise NotImplementedError("")

    def remove_non_driveable_area_points(self) -> np.ndarray:
        """ """
        raise NotImplementedError("")

    def remove_non_roi_points(self) -> np.ndarray:
        """ """
        raise NotImplementedError("")

    def get_rasterized_driveable_area(self) -> Tuple[np.ndarray, Sim2]:
        """Get the driveable area along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            npyimage_Sim2_city: Sim(2) that produces takes point in city coordinates to image coordinates, e.g.
                    p_pklimage = pklimage_Transformation_city * p_city
        """
        return self.raster_drivable_area_layer.array, self.raster_drivable_area_layer.array_Sim2_city

    def get_rasterized_roi(self) -> Tuple[np.ndarray, Sim2]:
        """Get the driveable area along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for driveable area
            array_Sim2_city: Sim(2) that produces takes point in city coordinates to numpy image, e.g.
                    p_npyimage = npyimage_Transformation_city * p_city
        """
        return self.raster_roi_layer.array, self.raster_roi_layer.array_Sim2_city

    def get_raster_layer_points_boolean(self, point_cloud: np.ndarray, layer_name: str) -> np.ndarray:
        """Query the binary segmentation layers (driveable area and ROI) at specific coordinates, to check values.

        Note:

        Args:
            point_cloud: Numpy array of shape (N,3)
            layer_name: indicating layer name, either "roi" or "driveable area"

        Returns:
            is_layer_boolean_arr: Numpy array of shape (N,) where i'th entry is True if binary segmentation is
                equal to 1 at the i'th point coordinate (i.e. is within the ROI, or within the driveable area,
                depending upon `layer_name` argument).
        """
        if layer_name == "roi":
            layer_values = self.raster_roi_layer.get_raster_values_at_coords(point_cloud, fill_value=0)
        elif layer_name == "driveable_area":
            layer_values = self.raster_drivable_area_layer.get_raster_values_at_coords(point_cloud, fill_value=0)
        else:
            raise ValueError("layer_name should be either `roi` or `driveable_area`.")

        is_layer_boolean_arr = layer_values == 1.0
        return is_layer_boolean_arr

    def append_height_to_2d_city_pt_cloud(self, pt_cloud_xy: np.ndarray) -> np.ndarray:
        """Accept 2d point cloud in xy plane and returns a 3d point cloud (xyz) by querying map for ground height.

        Args:
            pt_cloud_xy: Numpy array of shape (N,2) representing 2d coordinates of N query locations.

        Returns:
            pt_cloud_xyz: Numpy array of shape (N,3) representing 3d coordinates on the ground surface at
               N (x,y) query locations.
        """
        pts_z = self.raster_ground_height_layer.get_ground_height_at_xy(pt_cloud_xy)
        return np.hstack([pt_cloud_xy, pts_z[:, np.newaxis]])
