import numpy as np

from argoverse.evaluation.competition_util import get_polygon_from_points, poly_to_label


def test_get_polygon_from_points() -> None:
    """Ensure polygon contains only points within the convex hull.

    Point set shape looks like:
    .__.
    |  |
    |  |
    .__.
    """
    # z values between -1 and 2
    # Note: point 1 should be missing in convex hull
    points = np.array(
        [
            # at upper level
            [1, 0, 2],
            [3, 3, 2],
            [2, 3, 2],  # interior as linear combination of points 1 and 3
            [1, 3, 2],
            [3, 1, 2],
            # now, at lower level
            [1, 0, -1],
            [3, 3, -1],
            [2, 3, -1],  # interior as linear combination of points 1 and 3
            [1, 3, -1],
            [3, 1, -1],
        ]
    )
    poly = get_polygon_from_points(points)

    # note: first point is repeated as last point
    expected_exterior_coords = [
        (1.0, 0.0, 2.0),
        (3.0, 3.0, 2.0),
        (1.0, 3.0, 2.0),
        (3.0, 1.0, 2.0),
        (1.0, 0.0, -1.0),
        (3.0, 3.0, -1.0),
        (1.0, 3.0, -1.0),
        (3.0, 1.0, -1.0),
        (1.0, 0.0, 2.0),
    ]

    assert list(poly.exterior.coords) == expected_exterior_coords


def test_poly_to_label() -> None:
    """Make sure we can recover a cuboid, from a point set.
    
    Shape should resemble a slanted bounding box, 2 * sqrt(2) in width, and 3 * sqrt(2) in length
         .
       /  \\
    ./       \\
    \\          \\
       \\      /
         \\  /
            .
    """
    # fmt: off
    points = np.array(
        [
            [4, 6, -1],
            [4, 4, 2],
            [7, 5, 1],
            [7, 3, 0.5],
            [6, 4, 0],
            [6, 2, 0],
            [7, 5, 0],
            [5, 7, -1],
            [8, 4, 0]
        ]
    )
    # fmt: on
    poly = get_polygon_from_points(points)
    object_rec = poly_to_label(poly, category="VEHICLE", track_id="123")

    bbox_verts_2d = object_rec.as_2d_bbox()

    # fmt: off
    expected_bbox_verts_2d = np.array(
        [
            [8, 4, 2],
            [6, 2, 2],
            [5, 7, 2],
            [3, 5, 2]
        ]
    )
    # fmt: on
    assert np.allclose(expected_bbox_verts_2d, bbox_verts_2d)

    expected_length = np.sqrt(2) * 3
    expected_width = np.sqrt(2) * 2
    expected_height = 3.0

    assert np.isclose(object_rec.length, expected_length)
    assert np.isclose(object_rec.width, expected_width)
    assert np.isclose(object_rec.height, expected_height)

    assert object_rec.label_class == "VEHICLE"
    assert object_rec.track_id == "123"
