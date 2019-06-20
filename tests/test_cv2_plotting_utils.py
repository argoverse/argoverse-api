# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np

from argoverse.utils.cv2_plotting_utils import (
    draw_point_cloud_in_img_cv2,
    draw_polygon_cv2,
    draw_polyline_cv2,
    plot_bbox_polygon_cv2,
)


def test_draw_point_cloud_in_img_cv2_smokescreen():
    """
        We place 4 red circles in an image (with channel order BGR,
        per the OpenCV convention). We verify that pixel values change accordingly.
        """

    # xy: Numpy array of shape (K,2)
    xy = np.array([[20, 10], [0, 0], [199, 0], [199, 99]])

    for dtype in [np.uint8, np.float32]:
        # img: Numpy array of shape (M,N,3), representing an image
        img = np.ones((100, 200, 3), dtype=dtype)
        color = np.array([0, 0, 255], dtype=dtype)
        num_xy = xy.shape[0]

        # colors: Numpy array of shape (K,3), with BGR values in [0,255]
        colors = np.tile(color, (num_xy, 1))
        img_w_circ = draw_point_cloud_in_img_cv2(img, xy, colors)
        assert isinstance(img_w_circ, np.ndarray)
        assert img_w_circ.dtype == dtype
        for (x, y) in xy:
            assert np.allclose(img_w_circ[y, x, :], color)


def test_draw_polygon_cv2_smokescreen():
    """
        Test ability to fill a nonconvex polygon.

        We don't verify the rendered values since this requires
        scanline rendering computation to find the polygon's
        exact boundaries on a rasterized grid.
        """
    UINT8_MAX = 255
    img_w = 40
    img_h = 20
    for dtype in [np.uint8, np.float32]:

        # (x,y) points: Numpy array of shape (N,2), not (u,v) but (v,u)
        pentagon_pts = np.array([[1, 0], [2, 2], [0, 4], [-2, 2], [-1, 0]])
        # move the pentagon origin to (10,20) so in image center
        pentagon_pts[:, 0] += int(img_w / 2)
        pentagon_pts[:, 1] += int(img_h / 2)
        # img: Numpy array of shape (M,N,3)
        img = np.ones((img_h, img_w, 3), dtype=dtype) * UINT8_MAX
        # color: Numpy array of shape (3,)
        color = np.array([255.0, 0.0, 0.0])

        img_w_polygon = draw_polygon_cv2(pentagon_pts, img.copy(), color)

        assert isinstance(img_w_polygon, np.ndarray)
        assert img_w_polygon.shape == img.shape
        assert img_w_polygon.dtype == dtype


def test_plot_bbox_polygon_cv2_smokescreen():
    """
        Test drawing a green bounding box, with a thin red border, and
        test plotted inside, representing a track ID.

        We don't catch out-of-bounds errors -- that is up to the user.
        """
    for dtype in [np.uint8, np.float32]:
        bbox_h = 8
        bbox_w = 14
        img = np.zeros((bbox_h, bbox_w, 3), dtype=dtype)
        track_id = "test_uuid"
        color = np.array([0, 255, 0])

        # cols
        xmin = 6
        xmax = 11

        # rows
        ymin = 3
        ymax = 5

        bbox = np.array([xmin, ymin, xmax, ymax])
        # get image with a bounding box rendered inside of it.
        img_w_bbox = plot_bbox_polygon_cv2(img.copy(), track_id, color, bbox)

        assert isinstance(img_w_bbox, np.ndarray)
        assert img_w_bbox.shape == img.shape
        assert img_w_bbox.dtype == dtype
