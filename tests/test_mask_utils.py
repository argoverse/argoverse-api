import numpy as np

from argoverse.utils.mask_utils import get_mask_from_polygons


def test_get_mask_from_polygon() -> None:
    """Ensure that a triangle and skinny-column-like rectangle can be correctly rasterized onto a square grid."""
    # fmt: off
    triangle = np.array(
        [
            [1,1],
            [1,3],
            [3,1]
        ]
    )
    rectangle = np.array(
        [
            [5,1],
            [5,4],
            [5.5,4],
            [5.5,1]
        ]
    )
    # fmt: on
    mask = get_mask_from_polygons(polygons=[triangle, rectangle], img_h=7, img_w=7)

    # fmt: off
    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # fmt: on
    assert np.allclose(mask, expected_mask)


def test_get_mask_from_polygon_repeated_coords() -> None:
    """Verify polygon rasterization works correctly when the first coordinate is repeated (as last coordinate).

    Note: same scenario as above, a square grid with 2 polygons: a triangle and skinny-column-like rectangle.
    """
    # fmt: off
    triangle = np.array(
        [
            [1,1],
            [1,3],
            [3,1],
            [1,1]
        ]
    )
    rectangle = np.array(
        [
            [5,1],
            [5,4],
            [5.5,4],
            [5.5,1],
            [5,1]
        ]
    )
    # fmt: on
    mask = get_mask_from_polygons(polygons=[triangle, rectangle], img_h=7, img_w=7)

    # fmt: off
    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # fmt: on
    assert np.allclose(mask, expected_mask)


def test_get_mask_from_polygon_coords_out_of_bounds() -> None:
    """Test rasterization with polygon coordinates outside of the boundaries."""

    # fmt: off
    rectangle = np.array(
        [
            [-2,1],
            [8,1],
            [8,2],
            [-5,2]
        ]
    )
    # fmt: on
    mask = get_mask_from_polygons(polygons=[rectangle], img_h=5, img_w=5)

    # fmt: off
    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    )
    # fmt: on
    assert np.allclose(mask, expected_mask)


if __name__ == "__main__":
    test_get_mask_from_polygon()
    test_get_mask_from_polygon_repeated_coords()
    test_get_mask_from_polygon_coords_out_of_bounds()
