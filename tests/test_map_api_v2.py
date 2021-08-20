"""
Tests for the Argoverse 2.0 map API.

Uses a simplified map with 2 pedestrian crossings, and 3 lane segments.
"""
import unittest
from pathlib import Path

import numpy as np

from argoverse.map_representation.map_api_v2 import (
    ArgoverseV2StaticMap,
    PedestrianCrossing,
    VectorLaneSegment,
)

TEST_DATA_ROOT = Path(__file__).parent.resolve() / "test_data"


class TestArgoverseV2StaticMap(unittest.TestCase):
    """Unit test for the multi-view optimizer."""

    def setUp(self) -> None:
        super().setUp()

        log_map_dirpath = (
            TEST_DATA_ROOT / "v2_maps" / "dummy_log_map_v2_gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__2020-07-21-Z1F0076"
        )
        self.v2_map = ArgoverseV2StaticMap.from_json(log_map_dirpath)

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
        """Ensure that all VectorLaneSegment objects in the local map can be returned as a list."""
        vector_lane_segments = self.v2_map.get_scenario_lane_segments()
        assert isinstance(vector_lane_segments, list)
        assert all([isinstance(vls, VectorLaneSegment) for vls in vector_lane_segments])
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
        assert vector_da.xyz.shape == (172,3)

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

