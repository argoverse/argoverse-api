"""
Tests for the Argoverse 2.0 map API.

Uses a simplified map with 2 pedestrian crossings, and 3 lane segments.
"""
import unittest
from pathlib import Path

import numpy as np

from argoverse.map_representation.map_api_v2 import (
    ArgoverseStaticMapV2,
    DrivableArea,
    LaneMarkType,
    LaneSegment,
    LaneType,
    PedestrianCrossing,
    Point,
    Polyline,
)

TEST_DATA_ROOT = Path(__file__).parent.resolve() / "test_data"


class TestPolyline(unittest.TestCase):
    def test_from_list(self) -> None:
        """Ensure object is generated correctly from a list of dictionaries."""
        points_dict_list = [{"x": 874.01, "y": -105.15, "z": -19.58}, {"x": 890.58, "y": -104.26, "z": -19.58}]
        polyline = Polyline.from_dict_list(points_dict_list)

        assert isinstance(polyline, Polyline)

        assert len(polyline.waypoints) == 2
        assert polyline.waypoints[0] == Point(874.01, -105.15, -19.58)
        assert polyline.waypoints[1] == Point(890.58, -104.26, -19.58)


class TestDrivableArea(unittest.TestCase):
    def test_from_dict(self) -> None:
        """Ensure object is generated correctly from a dictionary.

        Note: 3 arbitrary waypoints taken from the dummy log map file.
        """
        json_data = {
            "area_boundary": {
                "points": [
                    {"x": 905.09, "y": -148.95, "z": -19.19},
                    {"x": 919.48, "y": -150.0, "z": -18.86},
                    {"x": 905.14, "y": -150.0, "z": -19.18},
                ]
            },
            "id": 4499430,
        }

        drivable_area = DrivableArea.from_dict(json_data)

        assert isinstance(drivable_area, DrivableArea)
        assert drivable_area.id == 4499430
        assert len(drivable_area.area_boundary) == 4  # first vertex is repeated as the last vertex


class TestPedestrianCrossing(unittest.TestCase):
    def test_from_dict(self) -> None:
        """Ensure object is generated correctly from a dictionary."""
        json_data = {
            "id": 6310421,
            "edge1": {"points": [{"x": 899.17, "y": -91.52, "z": -19.58}, {"x": 915.68, "y": -93.93, "z": -19.53}]},
            "edge2": {"points": [{"x": 899.44, "y": -95.37, "z": -19.48}, {"x": 918.25, "y": -98.05, "z": -19.4}]},
        }
        pedestrian_crossing = PedestrianCrossing.from_dict(json_data)

        isinstance(pedestrian_crossing, PedestrianCrossing)


class TestLaneSegment(unittest.TestCase):
    def test_from_dict(self) -> None:
        """Ensure object is generated correctly from a dictionary."""
        json_data = {
            "id": 93269421,
            "is_intersection": False,
            "lane_type": "VEHICLE",
            "left_lane_boundary": {
                "points": [
                    {"x": 873.97, "y": -101.75, "z": -19.7},
                    {"x": 880.31, "y": -101.44, "z": -19.7},
                    {"x": 890.29, "y": -100.56, "z": -19.66},
                ]
            },
            "left_lane_mark_type": "SOLID_YELLOW",
            "left_neighbor": None,
            "right_lane_boundary": {
                "points": [{"x": 874.01, "y": -105.15, "z": -19.58}, {"x": 890.58, "y": -104.26, "z": -19.58}]
            },
            "right_lane_mark_type": "SOLID_WHITE",
            "right_neighbor": 93269520,
            "successors": [93269500],
        }
        lane_segment = LaneSegment.from_dict(json_data)

        assert isinstance(lane_segment, LaneSegment)

        assert lane_segment.id == 93269421
        assert lane_segment.is_intersection == False
        assert lane_segment.lane_type == LaneType("VEHICLE")
        # fmt: off
        assert lane_segment.right_lane_boundary == Polyline(
            waypoints=[
                Point(874.01, -105.15, -19.58),
                Point(890.58, -104.26, -19.58)
            ]
        )
        assert lane_segment.left_lane_boundary == Polyline(
            waypoints=[
                Point(873.97, -101.75, -19.7),
                Point(880.31, -101.44, -19.7),
                Point(890.29, -100.56, -19.66)
            ]
        )
        # fmt: on
        assert lane_segment.right_mark_type == LaneMarkType("SOLID_WHITE")
        assert lane_segment.left_mark_type == LaneMarkType("SOLID_YELLOW")
        assert lane_segment.successors == [93269500]
        assert lane_segment.right_neighbor_id == 93269520
        assert lane_segment.left_neighbor_id is None


class TestArgoverseStaticMapV2(unittest.TestCase):
    """Unit test for the Argoverse 2.0 per-log map."""

    def setUp(self) -> None:
        super().setUp()

        log_map_dirpath = (
            TEST_DATA_ROOT / "v2_maps" / "dummy_log_map_v2_gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__2020-07-21-Z1F0076"
        )
        self.v2_map = ArgoverseStaticMapV2.from_json(log_map_dirpath, build_raster=True)

    def test_get_lane_segment_successor_ids(self) -> None:
        """Ensure lane segment successors are fetched properly."""

        lane_segment_id = 93269421
        successor_ids = self.v2_map.get_lane_segment_successor_ids(lane_segment_id)
        expected_successor_ids = [93269500]
        assert successor_ids == expected_successor_ids

        lane_segment_id = 93269500
        successor_ids = self.v2_map.get_lane_segment_successor_ids(lane_segment_id)
        expected_successor_ids = [93269554]
        assert successor_ids == expected_successor_ids

        lane_segment_id = 93269520
        successor_ids = self.v2_map.get_lane_segment_successor_ids(lane_segment_id)
        expected_successor_ids = [93269526]
        assert successor_ids == expected_successor_ids

    def test_lane_is_in_intersection(self) -> None:
        """Ensure the attribute describing if a lane segment is located with an intersection is fetched properly."""

        lane_segment_id = 93269421
        in_intersection = self.v2_map.lane_is_in_intersection(lane_segment_id)
        assert isinstance(in_intersection, bool)
        assert in_intersection == False

        lane_segment_id = 93269500
        in_intersection = self.v2_map.lane_is_in_intersection(lane_segment_id)
        assert isinstance(in_intersection, bool)
        assert in_intersection == True

        lane_segment_id = 93269520
        in_intersection = self.v2_map.lane_is_in_intersection(lane_segment_id)
        assert isinstance(in_intersection, bool)
        assert in_intersection == False

    def test_get_lane_segment_left_neighbor_id(self) -> None:
        """Ensure id of lane segment (if any) that is the left neighbor to the query lane segment can be fetched properly."""

        lane_segment_id = 93269421
        l_neighbor_id = self.v2_map.get_lane_segment_left_neighbor_id(lane_segment_id)
        assert l_neighbor_id is None

        lane_segment_id = 93269500
        l_neighbor_id = self.v2_map.get_lane_segment_left_neighbor_id(lane_segment_id)
        assert l_neighbor_id is None

        lane_segment_id = 93269520
        l_neighbor_id = self.v2_map.get_lane_segment_left_neighbor_id(lane_segment_id)
        assert l_neighbor_id == 93269421

    def test_get_lane_segment_right_neighbor_id(self) -> None:
        """Ensure id of lane segment (if any) that is the right neighbor to the query lane segment can be fetched properly."""

        lane_segment_id = 93269421
        r_neighbor_id = self.v2_map.get_lane_segment_right_neighbor_id(lane_segment_id)
        assert r_neighbor_id == 93269520

        lane_segment_id = 93269500
        r_neighbor_id = self.v2_map.get_lane_segment_right_neighbor_id(lane_segment_id)
        assert r_neighbor_id == 93269526

        lane_segment_id = 93269520
        r_neighbor_id = self.v2_map.get_lane_segment_right_neighbor_id(lane_segment_id)
        assert r_neighbor_id == 93269458

    def test_get_scenario_lane_segment_ids(self) -> None:
        """Ensure ids of all lane segments in the local map can be fetched properly."""
        lane_segment_ids = self.v2_map.get_scenario_lane_segment_ids()

        expected_lane_segment_ids = [93269421, 93269500, 93269520]
        assert lane_segment_ids == expected_lane_segment_ids

    def test_get_lane_segment_polygon(self) -> None:
        """Ensure lane segment polygons are fetched properly."""
        lane_segment_id = 93269421

        ls_polygon = self.v2_map.get_lane_segment_polygon(lane_segment_id)
        assert isinstance(ls_polygon, np.ndarray)

        expected_ls_polygon = np.array(
            [
                [874.01, -105.15, -19.58],
                [890.58, -104.26, -19.58],
                [890.29, -100.56, -19.66],
                [880.31, -101.44, -19.7],
                [873.97, -101.75, -19.7],
                [874.01, -105.15, -19.58],
            ]
        )
        np.testing.assert_allclose(ls_polygon, expected_ls_polygon)

    def test_lane_is_in_intersection(self) -> None:
        """Ensure boolean attribute describing if lane segment falls into an intersection is fetched properly."""

        lane_segment_id = 93269421
        in_intersection = self.v2_map.lane_is_in_intersection(lane_segment_id)
        assert in_intersection == False

        lane_segment_id = 93269500
        in_intersection = self.v2_map.lane_is_in_intersection(lane_segment_id)
        assert in_intersection == True

        lane_segment_id = 93269520
        in_intersection = self.v2_map.lane_is_in_intersection(lane_segment_id)
        assert in_intersection == False

    # def test_get_lane_segment_centerline(self) -> None:
    #     """Ensure lane segment centerlines can be inferred and fetched properly."""
    #     lane_segment_id = 1234

    #     centerline =self.v2_map. get_lane_segment_centerline(lane_segment_id)
    #     assert isinstance(centerline, np.ndarray)

    #     expected_centerline = np.array([])
    #     np.testing.assert_allclose(centerline, expected_centerline)

    def test_get_scenario_lane_segments(self) -> None:
        """Ensure that all LaneSegment objects in the local map can be returned as a list."""
        vector_lane_segments = self.v2_map.get_scenario_lane_segments()
        assert isinstance(vector_lane_segments, list)
        assert all([isinstance(vls, LaneSegment) for vls in vector_lane_segments])
        assert len(vector_lane_segments) == 3

    # def test_get_scenario_ped_crossings(self) -> None:
    #     """Ensure that all PedCrossing objects in the local map can be returned as a list."""
    #     ped_crossings = self.v2_map.get_scenario_ped_crossings()
    #     assert isinstance(ped_crossings, list)
    #     assert all([isinstance(pc, PedCrossing) for pc in ped_crossings])

    #     # fmt: off
    #     expected_ped_crossings = [
    #         PedCrossing(
    #             edge1=np.array(
    #                 [
    #                     [ 892.17,  -99.44,  -19.59],
    #                     [ 893.47, -115.4 ,  -19.45]
    #                 ]
    #             ), edge2=np.array(
    #                 [
    #                     [ 896.06,  -98.95,  -19.52],
    #                     [ 897.43, -116.58,  -19.42]
    #                 ]
    #             )
    #         ), PedCrossing(
    #             edge1=np.array(
    #                 [
    #                     [899.17, -91.52, -19.58],
    #                     [915.68, -93.93, -19.53]
    #                 ]
    #             ),
    #             edge2=np.array(
    #                 [
    #                     [899.44, -95.37, -19.48],
    #                     [918.25, -98.05, -19.4 ]
    #                 ]
    #             ),
    #         )
    #     ]
    #     # fmt: on
    #     assert len(ped_crossings) == len(expected_ped_crossings)
    #     assert all([pc == expected_pc for pc, expected_pc in zip(ped_crossings, expected_ped_crossings)])

    def test_get_scenario_vector_drivable_areas(self) -> None:
        """ """
        vector_das = self.v2_map.get_scenario_vector_drivable_areas()
        assert isinstance(vector_das, list)
        assert len(vector_das) == 1

        # examine just one sample
        vector_da = vector_das[0]
        assert vector_da.xyz.shape == (172, 3)

        # compare first and last vertex, for equality
        np.testing.assert_allclose(vector_da.xyz[0], vector_da.xyz[171])

        # fmt: off
        # compare first 4 vertices
        expected_first4_vertices = np.array(
            [
                [ 905.09, -148.95,  -19.19 ],
                [ 904.85, -141.95,  -19.25],
                [ 904.64, -137.25,  -19.28],
                [ 904.37, -132.55,  -19.32]
            ]
        )
        # fmt: on
        np.testing.assert_allclose(vector_da.xyz[:4], expected_first4_vertices)

    def test_get_raster_layer_points_boolean(self) -> None:
        """Ensure that region-of-interest (ROI) binary segmentation at (x,y) locations can be retrieved properly."""
        point_cloud = np.array(
            [
                [770.6398, -105.8351, -19.4105], # ego-vehicle pose at one timestamp
                [943.5386,  -49.6295, -19.3291], # ego-vehicle pose at one timestamp
                [918.0960,   82.5588, -20.5742], # ego-vehicle pose at one timestamp
                [9999999, 999999, 0], # obviously out of bounds value for city coordinate system
                [-999999, -999999, 0], # obviously out of bounds value for city coordinate system
            ])

        import pdb; pdb.set_trace()
        is_roi = self.v2_map.get_raster_layer_points_boolean(point_cloud, layer_name="roi")
        
        assert point_cloud.shape[0] == is_roi.shape[0]
        assert is_roi.dtype == bool

    def test_get_ground_height_at_xy(self) -> None:
        """Ensure that ground height at (x,y) locations can be retrieved properly.
        """
        point_cloud = np.array(
            [
                [770.6398, -105.8351, -19.4105], # ego-vehicle pose at one timestamp
                [943.5386,  -49.6295, -19.3291], # ego-vehicle pose at one timestamp
                [918.0960,   82.5588, -20.5742], # ego-vehicle pose at one timestamp
                [9999999, 999999, 0], # obviously out of bounds value for city coordinate system
                [-999999, -999999, 0], # obviously out of bounds value for city coordinate system
            ])
        ground_height_z = self.v2_map.raster_ground_height_layer.get_ground_height_at_xy(point_cloud)

        assert ground_height_z.shape[0] == point_cloud.shape[0]
        assert ground_height_z.dtype == np.float64

        # last 2 indices should be filled with dummy values (NaN) because obviously out of bounds.
        assert np.all(np.isnan(ground_height_z[-2:]))

        # based on grid resolution, ground should be within 10 centimeters of 30cm under back axle.
        expected_ground = point_cloud[:3,2] - 0.30
        assert np.allclose(np.absolute(expected_ground - ground_height_z[:3]), 0, atol=0.1)


    def test_get_ground_points_boolean(self) -> None:
        """Ensure that points close to the ground surface are correctly classified as `ground` category."""

        point_cloud = np.array(
            [
                [770.6398, -105.8351, -19.4105], # ego-vehicle pose at one timestamp
                [943.5386,  -49.6295, -19.3291], # ego-vehicle pose at one timestamp
                [918.0960,   82.5588, -20.5742], # ego-vehicle pose at one timestamp
                [9999999, 999999, 0], # obviously out of bounds value for city coordinate system
                [-999999, -999999, 0], # obviously out of bounds value for city coordinate system
            ])

        # first 3 points correspond to city_SE3_egovehicle, i.e. height of rear axle in city frame
        # ~30 cm below the axle should be the ground surface.
        point_cloud -= 0.30

        is_ground_pt = self.v2_map.raster_ground_height_layer.get_ground_points_boolean(point_cloud)
        expected_is_ground_pt = np.array([True, True, True, False, False])
        assert is_ground_pt.dtype == bool

