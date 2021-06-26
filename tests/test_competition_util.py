import numpy as np

from argoverse.evaluation.competition_label import get_polygon_from_points, poly_to_label


def test_get_polygon_from_points() -> None:
    """ """
    points = np.array([[], [], [], []])
    poly = get_polygon_from_points(points)

    assert True


def test_poly_to_label() -> None:
    """ """
    poly = get_polygon_from_points(points)

    object_rec = poly_to_label(poly, category="VEHICLE", track_id="123")

    assert True
