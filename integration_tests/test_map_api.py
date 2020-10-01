# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Map API unit tests"""

import glob
from typing import Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.datetime_utils import generate_datetime_string
from argoverse.utils.geometry import point_inside_polygon
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.mpl_plotting_utils import plot_lane_segment_patch


def add_lane_segment_to_ax(
    ax: plt.axes.Axis,
    lane_centerline: np.ndarray,
    lane_polygon: np.ndarray,
    patch_color: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    """"""
    plot_lane_segment_patch(lane_polygon, ax, color=patch_color, alpha=0.3)


def find_lane_segment_bounds_in_table(
    avm: ArgoverseMap, city_name: str, lane_segment_id: int
) -> Tuple[float, float, float, float]:
    """"""
    match_found = False
    # find the lane segment inside the table
    for table_idx, table_lane_id in avm.city_halluc_tableidx_to_laneid_map[city_name].items():
        if lane_segment_id == table_lane_id:
            match_found = True
            break
    if not match_found:
        print("Failure -- Lane ID not found!")
        quit()
    (xmin, ymin, xmax, ymax) = avm.city_halluc_bbox_table[city_name][table_idx]
    return xmin, ymin, xmax, ymax


def verify_halluc_lane_extent_index(enable_lane_boundaries: bool = False) -> None:
    """"""
    avm = ArgoverseMap()

    city_names = ["MIA", "PIT"]

    for city_name in city_names:
        # get all lane segment IDs inside of this city
        lane_segment_ids = list(avm.city_lane_centerlines_dict[city_name].keys())
        for lane_segment_id in lane_segment_ids:
            xmin, ymin, xmax, ymax = find_lane_segment_bounds_in_table(avm, city_name, lane_segment_id)

            predecessor_ids = avm.get_lane_segment_predecessor_ids(lane_segment_id, city_name)
            successor_ids = avm.get_lane_segment_successor_ids(lane_segment_id, city_name)
            (r_neighbor_id, l_neighbor_id) = avm.get_lane_segment_adjacent_ids(lane_segment_id, city_name)
            lane_centerline = avm.get_lane_segment_centerline(lane_segment_id, city_name)
            halluc_lane_polygon = avm.get_lane_segment_polygon(lane_segment_id, city_name)

            fig = plt.figure(figsize=(22.5, 8))
            ax = fig.add_subplot(111)

            # add the lane of interest
            add_lane_segment_to_ax(ax, lane_centerline, halluc_lane_polygon, "y", xmin, xmax, ymin, ymax)

            if predecessor_ids is not None:
                # add predecessors
                for predecessor_id in predecessor_ids:
                    lane_centerline = avm.get_lane_segment_centerline(predecessor_id, city_name)
                    halluc_lane_polygon = avm.get_lane_segment_polygon(predecessor_id, city_name)
                    xmin, ymin, xmax, ymax = find_lane_segment_bounds_in_table(avm, city_name, predecessor_id)
                    add_lane_segment_to_ax(
                        ax,
                        lane_centerline,
                        halluc_lane_polygon,
                        "r",
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                    )

            if successor_ids is not None:
                # add successors
                for successor_id in successor_ids:
                    lane_centerline = avm.get_lane_segment_centerline(successor_id, city_name)
                    halluc_lane_polygon = avm.get_lane_segment_polygon(successor_id, city_name)
                    xmin, ymin, xmax, ymax = find_lane_segment_bounds_in_table(avm, city_name, successor_id)
                    add_lane_segment_to_ax(
                        ax,
                        lane_centerline,
                        halluc_lane_polygon,
                        "b",
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                    )

            # add left neighbor
            if l_neighbor_id is not None:
                lane_centerline = avm.get_lane_segment_centerline(l_neighbor_id, city_name)
                halluc_lane_polygon = avm.get_lane_segment_polygon(l_neighbor_id, city_name)
                xmin, ymin, xmax, ymax = find_lane_segment_bounds_in_table(avm, city_name, l_neighbor_id)
                add_lane_segment_to_ax(
                    ax,
                    lane_centerline,
                    halluc_lane_polygon,
                    "g",
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )

            # add right neighbor
            if r_neighbor_id is not None:
                lane_centerline = avm.get_lane_segment_centerline(r_neighbor_id, city_name)
                halluc_lane_polygon = avm.get_lane_segment_polygon(r_neighbor_id, city_name)
                xmin, ymin, xmax, ymax = find_lane_segment_bounds_in_table(avm, city_name, r_neighbor_id)
                add_lane_segment_to_ax(
                    ax,
                    lane_centerline,
                    halluc_lane_polygon,
                    "m",
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )

            if enable_lane_boundaries:
                # Compare with Argo's proprietary, ground truth lane boundaries
                gt_lane_polygons = avm.city_to_lane_polygons_dict[city_name]
                for gt_lane_polygon in gt_lane_polygons:
                    dist = np.linalg.norm(gt_lane_polygon.mean(axis=0)[:2] - np.array([xmin, ymin]))
                    if dist < 30:
                        ax.plot(
                            gt_lane_polygon[:, 0],
                            gt_lane_polygon[:, 1],
                            color="k",
                            alpha=0.3,
                            zorder=1,
                        )

            ax.axis("equal")
            plt.show()
            datetime_str = generate_datetime_string()
            plt.savefig(f"lane_segment_id_{lane_segment_id}_@_{datetime_str}.jpg")
            plt.close("all")


# def verify_manhattan_search_functionality() -> None:
#     """
#         Minimal example where we
#         """
#     avm = ArgoverseMap()
#     # query_x = 254.
#     # query_y = 1778.

#     ref_query_x = 422.0
#     ref_query_y = 1005.0

#     city_name = "PIT"  # 'MIA'
#     for trial_idx in range(10):
#         query_x = ref_query_x + (np.random.rand() - 0.5) * 10
#         query_y = ref_query_y + (np.random.rand() - 0.5) * 10

#         # query_x,query_y = (3092.49845414,1798.55426805)
#         query_x, query_y = (3112.80160113, 1817.07585338)

#         lane_segment_ids = avm.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, 5000)

#         fig = plt.figure(figsize=(22.5, 8))
#         ax = fig.add_subplot(111)
#         # ax.scatter([query_x], [query_y], 500, color='k', marker='.')

#         plot_lane_segment_patch(pittsburgh_bounds, ax, color="m", alpha=0.1)

#         if len(lane_segment_ids) > 0:
#             for i, lane_segment_id in enumerate(lane_segment_ids):
#                 patch_color = "y"  # patch_colors[i % 4]
#                 lane_centerline = avm.get_lane_segment_centerline(lane_segment_id, city_name)

#                 test_x, test_y = lane_centerline.mean(axis=0)
#                 inside = point_inside_polygon(
#                     n_poly_vertices, pittsburgh_bounds[:, 0], pittsburgh_bounds[:, 1], test_x, test_y
#                 )

#                 if inside:
#                     halluc_lane_polygon = avm.get_lane_segment_polygon(lane_segment_id, city_name)
#                     xmin, ymin, xmax, ymax = find_lane_segment_bounds_in_table(avm, city_name, lane_segment_id)
#                     add_lane_segment_to_ax(
#                         ax, lane_centerline, halluc_lane_polygon, patch_color, xmin, xmax, ymin, ymax
#                     )

#         ax.axis("equal")
#         plt.show()
#         datetime_str = generate_datetime_string()
#         plt.savefig(f"{trial_idx}_{datetime_str}.jpg")
#         plt.close("all")


def verify_point_in_polygon_for_lanes() -> None:
    """"""
    avm = ArgoverseMap()

    # ref_query_x = 422.
    # ref_query_y = 1005.

    ref_query_x = -662
    ref_query_y = 2817

    city_name = "MIA"
    for trial_idx in range(10):
        query_x = ref_query_x + (np.random.rand() - 0.5) * 10
        query_y = ref_query_y + (np.random.rand() - 0.5) * 10

        fig = plt.figure(figsize=(22.5, 8))
        ax = fig.add_subplot(111)
        ax.scatter([query_x], [query_y], 100, color="k", marker=".")

        occupied_lane_ids = avm.get_lane_segments_containing_xy(query_x, query_y, city_name)
        for occupied_lane_id in occupied_lane_ids:
            halluc_lane_polygon = avm.get_lane_segment_polygon(occupied_lane_id, city_name)
            plot_lane_segment_patch(halluc_lane_polygon, ax, color="y", alpha=0.3)

        nearby_lane_ids = avm.get_lane_ids_in_xy_bbox(query_x, query_y, city_name)
        nearby_unoccupied_lane_ids: Set[int] = set(nearby_lane_ids) - set(occupied_lane_ids)
        for nearby_lane_id in nearby_unoccupied_lane_ids:
            halluc_lane_polygon = avm.get_lane_segment_polygon(nearby_lane_id, city_name)
            plot_lane_segment_patch(halluc_lane_polygon, ax, color="r", alpha=0.3)

        ax.axis("equal")
        plt.show()
        plt.close("all")


def plot_nearby_halluc_lanes(
    ax: plt.axes.Axis,
    city_name: str,
    avm: ArgoverseMap,
    query_x: float,
    query_y: float,
    patch_color: str = "r",
    radius: float = 20.0,
) -> None:
    """"""
    nearby_lane_ids = avm.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, radius)
    for nearby_lane_id in nearby_lane_ids:
        halluc_lane_polygon = avm.get_lane_segment_polygon(nearby_lane_id, city_name)
        plot_lane_segment_patch(halluc_lane_polygon, ax, color=patch_color, alpha=0.3)
        plt.text(
            halluc_lane_polygon[:, 0].mean(),
            halluc_lane_polygon[:, 1].mean(),
            str(nearby_lane_id),
        )


def verify_lane_tangent_vector() -> None:
    """
    debug low confidence lane tangent predictions

    I noticed that the confidence score of lane direction is
    pretty low (almost zero) in some logs
    """
    POSE_FILE_DIR = "../debug_lane_tangent"

    # both of these are Pittsburgh logs
    log_ids = [
        "033669d3-3d6b-3d3d-bd93-7985d86653ea",
        "028d5cb1-f74d-366c-85ad-84fde69b0fd3",
    ]

    avm = ArgoverseMap()
    city_name = "PIT"
    for log_id in log_ids:
        print(f"On {log_id}")
        pose_fpaths = glob.glob(f"{POSE_FILE_DIR}/{log_id}/poses/city_SE3_egovehicle_*.json")
        num_poses = len(pose_fpaths)
        egovehicle_xy_arr = np.zeros((num_poses, 2))
        for i, pose_fpath in enumerate(pose_fpaths):
            json_data = read_json_file(pose_fpath)
            egovehicle_xy_arr[i, 0] = json_data["translation"][0]
            egovehicle_xy_arr[i, 1] = json_data["translation"][1]

        for i, query_xy_city_coords in enumerate(egovehicle_xy_arr[::10, :]):

            query_xy_city_coords = np.array([3116.8282170094944, 1817.1269613456188])

            query_xy_city_coords = np.array([3304.7072308190845, 1993.1670162837597])

            # start = time.time()
            lane_dir_vector, confidence = avm.get_lane_direction(query_xy_city_coords, city_name, visualize=False)
            # end = time.time()
            # duration = end - start
            # print(f'query took {duration} s')
            # if confidence < 0.5:
            print(f"\t{i}: {confidence}")
            # if confidence == 0.:
            #   pdb.set_trace()
            # This was an absolute failure case!
            # lane_dir_vector, confidence = avm.get_lane_direction(query_xy_city_coords, city_name, visualize=True)

            visualize = True
            if visualize:
                fig = plt.figure(figsize=(22.5, 8))
                ax = fig.add_subplot(111)

                dx = lane_dir_vector[0] * 20
                dy = lane_dir_vector[1] * 20
                plt.arrow(
                    query_xy_city_coords[0],
                    query_xy_city_coords[1],
                    dx,
                    dy,
                    color="r",
                    width=0.3,
                    zorder=2,
                )

                query_x, query_y = query_xy_city_coords
                ax.scatter([query_x], [query_y], 100, color="k", marker=".")
                # make another plot now!

                plot_nearby_halluc_lanes(ax, city_name, avm, query_x, query_y)

                ax.axis("equal")
                plt.show()
                plt.close("all")


def test_remove_extended_predecessors() -> None:
    """Test remove_extended_predecessors() for map_api"""

    lane_seqs = [
        [9621385, 9619110, 9619209, 9631133],
        [9621385, 9619110, 9619209],
        [9619209, 9631133],
    ]
    xy = np.array([[-130.0, 2315.0], [-129.0, 2315.0], [-128.0, 2315.0]])  # 9619209 comntains xy[0]
    city_name = "MIA"

    avm = ArgoverseMap()
    filtered_lane_seq = avm.remove_extended_predecessors(lane_seqs, xy, city_name)

    assert np.array_equal(
        filtered_lane_seq, [[9619209, 9631133], [9619209], [9619209, 9631133]]
    ), "remove_extended_predecessors() failed!"


def test_get_candidate_centerlines_for_traj() -> None:
    """Test get_candidate_centerlines_for_traj()

    -180        .  .  .  .  .                   -100
                                                            2340
                            v
                            |                                 .
                            |                                 .
                            *  (CL1)                          .
                             \
                              \
                        (CL2)  \
                      >---------*-------------------->
                        s x x x x x x x x x e                 .
        >-------------------------------------------->        .
                        (CL3)                               2310
    """
    xy = np.array(
        [
            [-130.0, 2315.0],
            [-129.0, 2315.0],
            [-128.0, 2315.0],
            [-127, 2315],
            [-126, 2315],
            [-125, 2315],
            [-124, 2315],
        ]
    )
    city_name = "MIA"
    avm = ArgoverseMap()
    # import pdb; pdb.set_trace()
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(xy, city_name)

    assert len(candidate_centerlines) == 3, "Number of candidates wrong!"

    expected_centerlines = [
        np.array(
            [
                [-131.88540689, 2341.87225878],
                [-131.83054027, 2340.33723194],
                [-131.77567365, 2338.8022051],
                [-131.72080703, 2337.26717826],
                [-131.66594041, 2335.73215142],
                [-131.61107379, 2334.19712458],
                [-131.55620718, 2332.66209774],
                [-131.50134056, 2331.1270709],
                [-131.44647394, 2329.59204406],
                [-131.39160732, 2328.05701721],
                [-131.39160732, 2328.05701721],
                [-131.37997138, 2327.72338427],
                [-131.36833545, 2327.38975132],
                [-131.35669951, 2327.05611837],
                [-131.34506358, 2326.72248542],
                [-131.33342764, 2326.38885247],
                [-131.32179171, 2326.05521952],
                [-131.31015577, 2325.72158657],
                [-131.29851984, 2325.38795362],
                [-131.2868839, 2325.05432067],
                [-131.2868839, 2325.05432067],
                [-131.19279519, 2322.55119928],
                [-130.98376304, 2320.05690639],
                [-130.24692629, 2317.70490846],
                [-128.37426431, 2316.09358878],
                [-125.9878693, 2315.38876171],
                [-123.48883479, 2315.29784077],
                [-120.98715427, 2315.43423973],
                [-118.48467829, 2315.55478278],
                [-115.9822023, 2315.67532583],
                [-115.9822023, 2315.67532583],
                [-114.27604136, 2315.74436169],
                [-112.56988042, 2315.81339756],
                [-110.86371948, 2315.88243342],
                [-109.15755854, 2315.95146928],
                [-107.4513976, 2316.02050515],
                [-105.74523665, 2316.08954101],
                [-104.03907571, 2316.15857687],
                [-102.33291477, 2316.22761274],
                [-100.62675383, 2316.2966486],
            ]
        ),
        np.array(
            [
                [-139.13361714, 2314.54725812],
                [-136.56123771, 2314.67259898],
                [-133.98885829, 2314.79793983],
                [-131.41647886, 2314.92328069],
                [-128.84409943, 2315.04862155],
                [-126.27172001, 2315.1739624],
                [-123.69934058, 2315.29930326],
                [-121.12696116, 2315.42464412],
                [-118.55458173, 2315.54998497],
                [-115.9822023, 2315.67532583],
                [-115.9822023, 2315.67532583],
                [-114.27604136, 2315.74436169],
                [-112.56988042, 2315.81339756],
                [-110.86371948, 2315.88243342],
                [-109.15755854, 2315.95146928],
                [-107.4513976, 2316.02050515],
                [-105.74523665, 2316.08954101],
                [-104.03907571, 2316.15857687],
                [-102.33291477, 2316.22761274],
                [-100.62675383, 2316.2966486],
            ]
        ),
        np.array(
            [
                [-178.94773558, 2309.75038731],
                [-175.73132051, 2309.8800903],
                [-172.51490545, 2310.00979328],
                [-169.29849039, 2310.13949626],
                [-166.08207532, 2310.26919925],
                [-162.86566026, 2310.39890223],
                [-159.64924519, 2310.52860522],
                [-156.43283013, 2310.6583082],
                [-153.21641506, 2310.78801118],
                [-150.0, 2310.91771417],
                [-150.0, 2310.91771417],
                [-148.77816698, 2310.97013154],
                [-147.55633396, 2311.0225489],
                [-146.33450094, 2311.07496627],
                [-145.11266792, 2311.12738364],
                [-143.89083489, 2311.17980101],
                [-142.66900187, 2311.23221837],
                [-141.44716885, 2311.28463574],
                [-140.22533583, 2311.33705311],
                [-139.00350281, 2311.38947048],
                [-139.00350281, 2311.38947048],
                [-136.42679274, 2311.51113082],
                [-133.85008268, 2311.63279117],
                [-131.27337261, 2311.75445152],
                [-128.69666254, 2311.87611187],
                [-126.11995247, 2311.99777222],
                [-123.54324241, 2312.11943257],
                [-120.96653234, 2312.24109292],
                [-118.38982227, 2312.36275327],
                [-115.8131122, 2312.48441361],
                [-115.8131122, 2312.48441361],
                [-114.11040334, 2312.54102742],
                [-112.40815545, 2312.6106056],
                [-110.70605773, 2312.68440659],
                [-109.00396, 2312.75820759],
                [-107.30186227, 2312.83200858],
                [-105.59976454, 2312.90580958],
                [-103.89766681, 2312.97961057],
                [-102.19556909, 2313.05341156],
                [-100.49347136, 2313.12721256],
            ]
        ),
    ]

    for i in range(len(expected_centerlines)):
        assert np.allclose(expected_centerlines[i], candidate_centerlines[i]), "Centerline coordinates wrong!"


def test_dfs() -> None:
    """Test dfs for lane graph

    Lane Graph:
                9629626
               /       \
              /         \
          9620336    9632589
          (10.77)     (8.33)
             |          |
             |          |
          9628835    9621228
           (31.9)    (31.96)
             |          |
             |          |
          9629406    9626257
           (7.9)      (7.81)

    """

    lane_id = 9629626
    city_name = "MIA"
    dist = 0.0
    threshold = 30.0
    extend_along_predecessor = False

    avm = ArgoverseMap()
    lane_seq = avm.dfs(lane_id, city_name, dist, threshold, extend_along_predecessor)

    expected_lane_seq = [[9629626, 9620336, 9628835], [9629626, 9632589, 9621228]]
    assert np.array_equal(lane_seq, expected_lane_seq), "dfs over lane graph failed!"
