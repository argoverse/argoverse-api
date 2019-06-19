# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np
from argoverse.utils.geometry import filter_point_cloud_to_polygon, point_inside_polygon, rotate_polygon_about_pt


"""
Unit tests for argoverse/utils/geometry.py
"""


def test_rotate_polygon_about_pt_2d_triangle_0deg_origin():
    """
        Rotate a triangle 0 degrees counterclockwise, about the origin
        """
    polygon_pts = np.array([[0, 0], [4, 0], [4, 4]])
    theta = 0  # radians, which is 0 degrees
    c = np.cos(theta)
    s = np.sin(theta)
    rotmat = np.array([[c, -s], [s, c]])
    center_pt = np.array([0, 0])
    rot_polygon = rotate_polygon_about_pt(polygon_pts.copy(), rotmat, center_pt)
    assert np.allclose(polygon_pts, rot_polygon)


def test_rotate_polygon_about_pt_2d_triangle_90deg_origin():
    """
        Rotate a triangle 90 degrees counterclockwise, about the origin
        """
    polygon_pts = np.array([[0, 0], [4, 0], [4, 4]])
    theta = np.pi / 2  # radians, which is 90 degrees
    c = np.cos(theta)
    s = np.sin(theta)
    rotmat = np.array([[c, -s], [s, c]])
    center_pt = np.array([0, 0])
    rot_polygon = rotate_polygon_about_pt(polygon_pts.copy(), rotmat, center_pt)

    gt_rot_polygon = np.array([[0, 0], [0, 4], [-4, 4]])
    assert np.allclose(gt_rot_polygon, rot_polygon)


def test_rotate_polygon_about_pt_2d_triangle_0deg_nonorigin():
    """
        Rotate a triangle 0 degrees counterclockwise, but this time
        not rotating about the origin.
        """
    polygon_pts = np.array([[0, 0], [4, 0], [4, 4]])
    theta = 0  # radians, which is 0 degrees
    c = np.cos(theta)
    s = np.sin(theta)
    rotmat = np.array([[c, -s], [s, c]])
    center_pt = np.array([1, 1])
    rot_polygon = rotate_polygon_about_pt(polygon_pts.copy(), rotmat, center_pt)
    assert np.allclose(polygon_pts, rot_polygon)


def test_rotate_polygon_about_pt_2d_triangle_90deg_nonorigin():
    """
        Rotate a triangle 90 degrees counterclockwise, but this time
        not rotating about the origin. Instead we rotate about (2,2).
        """
    polygon_pts = np.array([[0, 0], [4, 0], [4, 4]])
    theta = np.pi / 2  # radians, which is 90 degrees
    c = np.cos(theta)
    s = np.sin(theta)
    rotmat = np.array([[c, -s], [s, c]])
    center_pt = np.array([2, 2])
    rot_polygon = rotate_polygon_about_pt(polygon_pts.copy(), rotmat, center_pt)

    gt_rot_polygon = np.array([[4, 0], [4, 4], [0, 4]])
    assert np.allclose(gt_rot_polygon, rot_polygon)


def test_rotate_polygon_about_pt_3d():
    """
    Rotate a point cloud in xy plane, but keep z fixed. in other words,
    perform a 3D rotation about Z-axis (rotation about yaw axis).

    This point cloud represents an extruded triangle.
    """
    pts = np.array([[3.5, 2, -1], [3, 1, -1], [4, 1, -1], [3.5, 2, 2.3], [3, 1, 2.3], [4, 1, 2.3]])

    theta = np.pi
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    center_pt = np.array([1.0, 1.0, 5.0])
    rotated_pts = rotate_polygon_about_pt(pts, R, center_pt)

    gt_rotated_pts = np.array([[-1.5, 0, -1], [-1, 1, -1], [-2, 1, -1], [-1.5, 0, 2.3], [-1, 1, 2.3], [-2, 1, 2.3]])

    assert np.allclose(rotated_pts, gt_rotated_pts)


def test_filter_point_cloud_to_polygon_2d_triangle():
    """
        Test points that fall within a triangle symbol centered at
        the origin. The shape resembles:

            (0,1)
                /\
               /  \
        (-1,0)--.--(1,0)

    By shapely convention, points on the boundary are not
    considered to be located inside of the polygon.
        """
    polygon = np.array([[-1, 0], [1, 0], [0, 1]])

    point_cloud_2d = np.array(
        [
            [-1, 1],  # out
            [-0.5, 0.5],  # on boundary, so out
            [0, 0.5],  # in
            [0.5, 0.5],  # on boundary, so out
            [0.0, 0.0],  # on boundary, so out
            [0.5, -0.1],  # out
            [1, -1],  # out
            [-0.5, -0.1],  # out
            [-1, -1],  # out
            [-0.5, 0.1],  # in
            [0.49999, 0.49999],  # in
            [-0.49999, 0.49999],
        ]
    )  # in

    interior_pts = filter_point_cloud_to_polygon(polygon, point_cloud_2d.copy())
    interior_gt_bool = np.array([False, False, True, False, False, False, False, False, False, True, True, True])
    assert np.allclose(point_cloud_2d[interior_gt_bool], interior_pts)


def test_filter_point_cloud_to_polygon_2d_triangle_and_3d_pointcloud():
    """
    Test points that fall within a triangle symbol centered at
    the origin. The shape resembles:

            (0,1)
            /\
           /  \
    (-1,0)--.--(1,0)

    By shapely convention, points on the boundary are not
    considered to be located inside of the polygon.
    """
    polygon = np.array([[-1, 0], [1, 0], [0, 1]])

    point_cloud_2d = np.array(
        [
            [-1, 1, -3.0],  # out
            [-0.5, 0.5, -2.0],  # on boundary, so out
            [0, 0.5, -1.0],  # in
            [0.5, 0.5, 0.0],  # on boundary, so out
            [0.0, 0.0, 1.0],  # on boundary, so out
            [0.5, -0.1, 2.1],  # out
            [1, -1, 3.2],  # out
            [-0.5, -0.1, 4.3],  # out
            [-1, -1, 5.4],  # out
            [-0.5, 0.1, 0.0],  # in
            [0.49999, 0.49999, 6.0],  # in
            [-0.49999, 0.49999, 7.0],
        ]
    )  # in

    interior_pts = filter_point_cloud_to_polygon(polygon, point_cloud_2d.copy())

    interior_gt_bool = np.array([False, False, True, False, False, False, False, False, False, True, True, True])
    assert np.allclose(point_cloud_2d[interior_gt_bool], interior_pts)


def test_filter_point_cloud_to_polygon_2d_triangle_all_outside():
    """
    Test points that fall within a triangle symbol centered at
    the origin. The shape resembles:

            (0,1)
            /\
           /  \
    (-1,0)--.--(1,0)

    All of these points should fall outside. We test if we can
    break the function in such a case.
    """
    polygon = np.array([[-1, 0], [1, 0], [0, 1]])

    point_cloud_2d = np.array(
        [
            [-1, 1],  # out
            [-0.5, 0.5],  # on boundary, so out
            [0.5, 0.5],  # on boundary, so out
            [0.0, 0.0],  # on boundary, so out
            [0.5, -0.1],  # out
            [1, -1],  # out
            [-0.5, -0.1],  # out
            [-1, -1],
        ]
    )  # out

    interior_pts = filter_point_cloud_to_polygon(polygon, point_cloud_2d.copy())
    assert interior_pts is None


def test_filter_point_cloud_to_polygon_2d_redcross():
    """
    Test points that fall within a red cross symbol centered at
    the origin.

        -----
        |   |
    ----|   |----(2,1)
    |     .     |
    |           |
    ----|   |----(2,-1)
        |   |
        -----

    """
    polygon = np.array(
        [[1, 1], [1, 2], [-1, 2], [-1, 1], [-2, 1], [-2, -1], [-1, -1], [-1, -2], [1, -2], [1, -2], [2, -1], [2, 1]]
    )
    point_cloud_2d = np.array([[0.9, 0.9], [1.1, 1.1], [1, 2.0], [1, 2.1], [0, 1.99]])  # in  # out  # out  # out  # in
    interior_pts = filter_point_cloud_to_polygon(polygon, point_cloud_2d)

    interior_gt_bool = np.array([True, False, False, False, True])
    assert np.allclose(point_cloud_2d[interior_gt_bool], interior_pts)


def point_inside_polygon_interior_sanity_check(n_vertices, poly_x_pts, poly_y_pts, test_x, test_y):
    """
    We use this function to verify shapely.geometry's correctness. This fn only works correctly
    on the interior of an object (not on the boundary).

    Ray casting: Draw a virtual ray from anywhere outside the polygon to your point and count how often it
    hits a side of the polygon. If the number of hits is even, it's outside of the polygon, if it's odd, it's inside.

    Run a semi-infinite ray horizontally (increasing x, fixed y) out from the test point, and count how many edges
    it crosses. At each crossing, the ray switches between inside and outside. This is called the Jordan curve theorem.

    The variable c is switching from 0 to 1 and 1 to 0 each time the horizontal ray crosses any edge.
    It keeps track of whether the number of edges crossed are even or odd. 0 means even and 1 means odd.

    We check if (test_x - x_i)    (x_j - x_i)
                -------------- <  -----------
                (test_y - y_i)    (y_j - y_i)

        Args:
            n_vertices: number of vertices in the polygon
            vert_x_pts, vert_y_pts: lists or NumPy arrays containing the x- and y-coordinates of the polygon's vertices.
            test_x, test_y: the x- and y-coordinate of the test point

        Returns:
            inside: boolean, whether point lies inside polygon
    """
    i = 0
    j = n_vertices - 1
    count = 0
    assert n_vertices == poly_x_pts.shape[0] == poly_y_pts.shape[0]
    for i in range(0, n_vertices, 1):

        if i > 0:
            j = i - 1

        x_i = poly_x_pts[i]
        y_i = poly_y_pts[i]

        x_j = poly_x_pts[j]
        y_j = poly_y_pts[j]

        # checks for intersection!
        both_ends_not_above = (y_i > test_y) != (y_j > test_y)
        if both_ends_not_above and (test_x < (x_j - x_i) * (test_y - y_i) / (y_j - y_i) + x_i):
            count += 1

    return (count % 2) == 1


def test_point_in_polygon_shapely_vs_our_implementation():
    """
    Using 20 points originally sampled in unit square (uniform random),
    ensure that our implementation of point-in-polygon matches
    shapely.geometry's implementation. Using non-convex polygon.
    """
    poly_boundary = [(0, 0), (1, 2), (3, 2), (4, 0), (2, 1), (0, 0)]
    n_vertices = len(poly_boundary) - 1
    vert_x_pts, vert_y_pts = zip(*poly_boundary)

    # skip the last vertex since it is repeating the zero'th vertex
    vert_x_pts = np.array(vert_x_pts)[:-1]
    vert_y_pts = np.array(vert_y_pts)[:-1]

    test_pts = np.array(
        [
            [0.7906206529554805, 3.5247972478750755],
            [2.796929714397516, 1.2869605710087773],
            [2.13736772658553, 0.9354300802539841],
            [0.5533843176857371, 3.5907793820406786],
            [1.067522648741737, 2.107324516856971],
            [0.18413436334503475, 1.999149713775295],
            [1.021532057710976, 1.8409888453463181],
            [1.5852780307128183, 3.3784534320811512],
            [1.808454845296604, 2.2902639353983276],
            [2.3004402727971534, 2.764413577543416],
            [0.22177110806299005, 3.3753677368156096],
            [0.7884126834996188, 3.105915665452909],
            [0.20123469298750463, 2.735888039565455],
            [2.485042055570448, 2.160836673437083],
            [3.2481703626785148, 0.8704179207883942],
            [3.884111027227804, 1.951093430056864],
            [1.4627990780181208, 3.0282856766333315],
            [1.5737052681266528, 1.5394901616575405],
            [3.9923222020570623, 0.05764955338346489],
            [0.19501433730945195, 3.0991165694473595],
        ]
    )

    gt_is_inside = np.array(
        [
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
        ]
    )

    is_inside = np.zeros(test_pts.shape[0], dtype=bool)
    for i in range(20):
        test_x = test_pts[i, 0]
        test_y = test_pts[i, 1]
        inside = point_inside_polygon(n_vertices, vert_x_pts, vert_y_pts, test_x, test_y)
        assert inside == point_inside_polygon_interior_sanity_check(n_vertices, vert_x_pts, vert_y_pts, test_x, test_y)


def test_point_in_polygon_square():
    """
    Ensure point barely inside square boundary is "inside".
    """
    square = np.array([[2, 2], [2, -2], [-2, -2], [-2, 2]])
    n_vertices = 4
    poly_x_pts = square[:, 0]
    poly_y_pts = square[:, 1]

    test_x = 0
    test_y = 1.99999

    inside = point_inside_polygon(n_vertices, poly_x_pts, poly_y_pts, test_x, test_y)
    assert inside
    assert inside == point_inside_polygon_interior_sanity_check(n_vertices, poly_x_pts, poly_y_pts, test_x, test_y)
