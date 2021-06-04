# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Collection of utilities to explicitly form the camera view frustum and clip line segments to within frustum view.

These utilities use focal lengths and camera image dimensions to create camera view frustums,
and then to clip line segments to live within the frustum only.
"""

import copy
from typing import Any, List, Optional, Tuple

import numpy as np

from argoverse.utils.camera_stats import get_image_dims_for_camera
from argoverse.utils.manhattan_search import compute_point_cloud_bbox


def fit_plane_to_point_cloud(pc: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    """Use SVD with at least 3 points to fit a plane.

    Args:
        pc: Array of shape (N, 3)

    Returns:
        a, b, c, d: float-like values defining ax + by + cz = d for the plane
    """
    center = pc.sum(axis=0) / pc.shape[0]
    u, s, vh = np.linalg.svd(pc - center)

    # Get the unitary normal vector
    u_norm = vh[2, :]
    d = -np.dot(u_norm, center)
    a, b, c = u_norm
    return a, b, c, d


def form_right_clipping_plane(fx: float, img_width: int) -> np.ndarray:
    """Form the right clipping plane for a camera view frustum.

    In the camera coordinate frame, y is down the imager, x is across the imager,
    and z is along the optical axis. The focal length is the distance to the center
    of the image plane. We know that a similar triangle is formed as follows::

        (x,y,z)--(x,y,z)
          |         /
          |        / ->outside of frustum
          |       / ->outside of frustum
          | (w/2)/
          o-----o IMAGE PLANE
          |    /
        fx|   /
          |  /
          | /
          O PINHOLE

    Normal must point into the frustum. The plane moves +fx in z-axis for
    every +w/2 in x-axis, so normal will have negative inverse slope components.

    Then, enforce that x-component of normal points in negative direction.
    The "d" in "ax + by + cz = d" is zero because plane goes through origin.

    Args:
        fx: Horizontal focal length in pixels
        img_width: Image width in pixels

    Returns:
        right_plane: Array of shape (4,) for ax + by + cz = d
    """
    right_plane = np.array([-fx, 0.0, img_width / 2.0, 0.0])
    right_plane /= np.linalg.norm(right_plane)

    return right_plane


def form_left_clipping_plane(fx: float, img_width: int) -> np.ndarray:
    r"""Form the left clipping plane for a camera view frustum.

    In the camera coordinate frame, y is down the imager, x is across the imager,
    and z is along the optical axis. The focal length is the distance to the center
    of the image plane. We know that a similar triangle is formed as follows::

                       (x,y,z)-----(x,y,z)
                          \\          |
     outside of frustum <- \\         |
      outside of frustum <- \\        |
                             \\ (-w/2)|
                               o------o IMAGE PLANE
                               \\     |
                                \\    |
                                 \\   |fx
                                  \\  |
                                   \\ |
                                      O PINHOLE

    Normal must point into the frustum. The plane moves +fx in z-axis for
    every -w/2 in x-axis, so normal will have negative inverse slope components.
    The "d" in "ax + by + cz = d" is zero because plane goes through origin.

    Args:
        fx: Horizontal focal length in pixels
        img_width: Image width in pixels

    Returns:
        left_plane: Array of shape (4,) for ax + by + cz = d
    """
    left_plane = np.array([fx, 0.0, img_width / 2.0, 0.0])
    left_plane /= np.linalg.norm(left_plane)
    return left_plane


def form_top_clipping_plane(fx: float, img_height: int) -> np.ndarray:
    r"""Form the top clipping plane for a camera view frustum.

    In the camera coordinate frame, y is down the imager, x is across the imager,
    and z is along the optical axis. The focal length is the distance to the center
    of the image plane. We know that a similar triangle is formed as follows::

          (x,y,z)               (x,y,z)
              \\=================//
               \\               //
        (-w/h,-h/2,fx)       (w/h,-h/2,fx)
                 o-------------o
                 |\\         //| IMAGE PLANE
                 | \\       // | IMAGE PLANE
                 o--\\-----//--o
                     \\   //
                      \\ //
                        O PINHOLE

    Normal must point into the frustum. The plane moves -h/2 in y-axis for every
    +fx in z-axis, so normal will have negative inverse slope components. The
    x-axis component is zero since constant in x.
    The "d" in "ax + by + cz = d" is zero because plane goes through origin.

    Args:
        fx: Horizontal focal length in pixels
        img_height: Image height in pixels

    Returns:
        top_plane: Array of shape (4,) for ax + by + cz = d
    """
    top_plane = np.array([0.0, fx, img_height / 2.0, 0.0])
    top_plane /= np.linalg.norm(top_plane)
    return top_plane


def form_low_clipping_plane(fx: float, img_height: int) -> np.ndarray:
    r"""Form the low clipping plane for a camera view frustum.

    Use 3 points to fit the low clipping plane. In the camera coordinate frame,
    y is down the imager, x is across the imager, and z is along the optical axis.
    We know that a similar triangle is formed as follows::

                (x,y,z)              (x,y,z)
                   \\                   //
                    \\ o-------------o //
                     \\| IMAGE PLANE |//
                       |             |/
        (-w/h, h/2,fx) o-------------o (w/h, h/2,fx)
                        \\         //
                         \\       //
                          \\     //
                           \\   //
                            \\ //
                              O PINHOLE

    Normal must point into the frustum. The plane moves +h/2 in y-axis for every
    +fx in z-axis, so normal will have negative inverse slope components. The
    x-axis component is zero since constant in x.

    Then enforce that y-coord of normal points in neg y-axis dir(up) on low-clipping plane.
    The z-coord should point in positive z-axis direction (away from camera).
    The "d" in "ax + by + cz = d" is zero because plane goes through origin.

    Args:
        fx: Horizontal focal length in pixels
        img_height: Image height in pixels

    Returns:
        low_plane: Array of shape (4,) for ax + by + cz = d
    """
    low_plane = np.array([0.0, -fx, img_height / 2.0, 0.0])
    low_plane /= np.linalg.norm(low_plane)
    return low_plane


def form_near_clipping_plane(near_clip_dist: float) -> np.ndarray:
    """Form the near clipping plane for a camera view frustum.

    In the camera coordinate frame, y is down the imager, x is across the imager,
    and z is along the optical axis. The near clipping plane should point in
    the positive z-direction (along optical axis).

    We form "ax + by + cz = d", where "d" is a distance from the origin.

    Args:
        near_clip_dist: Near clipping plane distance in meters

    Returns:
        top_plane: Array of shape (4,) for ax + by + cz = d
    """
    return np.array([0.0, 0.0, 1.0, -near_clip_dist])


def generate_frustum_planes(K: np.ndarray, camera_name: str, near_clip_dist: float = 0.5) -> Optional[List[np.ndarray]]:
    """Compute the planes enclosing the field of view (viewing frustum) for a single camera.

    We do this using similar triangles.
    tan(theta/2) = (0.5 * height)/focal_length
    "theta" is the vertical FOV. Similar for horizontal FOV.
    height and focal_length are both in pixels.

    Note that ring cameras and stereo cameras have different image widths
    and heights, affecting the field of view.

    Ring Camera intrinsics K look like (in pixels)::

        [1400,   0, 964]     [fx,skew,cx]
        [   0,1403, 605] for [-,   fy,cy]
        [   0,   0,   1]     [0,    0, 1]

    Args:
        K: Array of shape (3, 3) representing camera intrinsics matrix
        camera_name: String representing the camera name to get the dimensions of and compute the FOV for
        near_clip_dist: The distance for the near clipping plane in meters

    Returns:
        planes: List of length 5, where each list element is an Array of shape (4,)
                representing the equation of a plane, e.g. (a, b, c, d) in ax + by + cz = d
    """
    img_width, img_height = get_image_dims_for_camera(camera_name)
    if img_width is None or img_height is None:
        return None

    # frustum starts at optical center [0,0,0]
    fx = K[0, 0]
    fy = K[1, 1]

    right_plane = form_right_clipping_plane(fx, img_width)
    left_plane = form_left_clipping_plane(fx, img_width)
    near_plane = form_near_clipping_plane(near_clip_dist)

    # The horizontal and vertical focal lengths should be very close to equal,
    # otherwise something went wrong when forming K matrix.
    assert np.absolute(fx - fy) < 10

    low_plane = form_low_clipping_plane(fx, img_height)
    top_plane = form_top_clipping_plane(fx, img_height)

    planes = [left_plane, right_plane, near_plane, low_plane, top_plane]
    return planes


def clip_segment_v3_plane_n(
    p1: np.ndarray, p2: np.ndarray, planes: List[np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Iterate over the frustum planes and intersect them with the segment.

    This  updating the min/max, bailing out early if the min > max.
    We exploit the fact that in a camera frustum, all plane
    normals point inside the frustum volume.

    See section "Line-Plane Intersection" for technical details at: http://geomalgorithms.com/a05-_intersect-1.html

    "t" is the distance we travel along the ray from p1 to p2.
    If "t" exceeds 1.0, then we have exceeded the line segment.

    A similar function, written in C, can be found in the Blender source code at:
    https://fossies.org/dox/blender-2.79b/math__geom_8c_source.html

    Args:
        p1: 3D vector defining a point to constrain a line segment
        p2: 3D vector defining a point to constrain a line segment
        planes: List of length 5, where each list element is an Array of shape (4,)
                representing the equation of a plane, e.g. (a, b, c, d) in ax + by + cz = d
    Returns:
        2 vector triplets (the clipped segment) or (None, None) meaning the segment is entirely outside the frustum.
    """
    dp = p2 - p1

    p1_fac = 0.0
    p2_fac = 1.0

    for p in planes:
        div = p[:3].dot(dp)

        # check if line vector and plane normal are perpendicular
        # if perpendicular, line and plane are parallel
        if div != 0.0:
            # if not perpendicular, find intersection
            t = -plane_point_side_v3(p, p1)
            if div > 0.0:  # clip p1 lower bounds
                if t >= div:
                    return None, None
                if t > 0.0:
                    fac = t / div
                    if fac > p1_fac:
                        p1_fac = fac
                        if p1_fac > p2_fac:
                            # intersection occurs outside of segment
                            return None, None
            elif div < 0.0:  # clip p2 upper bounds
                if t > 0.0:
                    return None, None
                if t > div:
                    fac = t / div
                    if fac < p2_fac:
                        p2_fac = fac
                        if p1_fac > p2_fac:
                            return None, None

    p1_clip = p1 + (dp * p1_fac)
    p2_clip = p1 + (dp * p2_fac)
    return p1_clip, p2_clip


def plane_point_side_v3(p: np.ndarray, v: np.ndarray) -> Any:
    """Get sign of point to plane distance.

    This function does not compute the actual distance.

    Positive denotes that point v is on the same side of the plane as the plane's normal vector.
    Negative if it is on the opposite side.

    Args:
        p: Array of shape (4,) representing a plane in Hessian Normal Form, ax + by + cz + d = 0
        v: A vector/3D point

    Returns:
        sign: A float-like value representing sign of signed distance
    """
    return p[:3].dot(v) + p[3]


def cuboid_to_2d_frustum_bbox(corners: np.ndarray, planes: List[np.ndarray], K: np.ndarray) -> Optional[np.ndarray]:
    """Convert a 3D cuboid to a 2D frustum bounding box.

    We bring the 3D points into each camera, and do the clipping there.

    Args:
        corners: The corners to use as the corners of the frustum bounding box
        planes: List of 4-tuples for ax + by + cz = d representing planes in Hessian Normal Form
        K: 3x3 camera intrinsic matrix

    Returns:
        bbox_2d: Numpy array of shape (4,) with entries [x_min,y_min,x_max,y_max]
    """

    def clip_line_segment(pt_a: np.ndarray, pt_b: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clip a line segment based on two points and the camera instrinc matrix.

        Args:
            pt_a: One 3D point vector constraining a line segment
            pt_b: One 3D point vector constraining a line segment
            K: A 3x3 array representing a camera intrinsic matrix

        Returns:
            a, b: A tuple of the clipped line segment 3D point vectors
        """
        pt_a = K.dot(pt_a)
        pt_a /= pt_a[2]

        pt_b = K.dot(pt_b)
        pt_b /= pt_b[2]

        return np.round(pt_a).astype(np.int32), np.round(pt_b).astype(np.int32)

    def clip_rect(selected_corners: np.ndarray, clipped_uv_verts: np.ndarray) -> np.ndarray:
        """Clip a rectangle based on the selected corners and clipped vertices coordinates.

        Args:
            selected_corners: A list of selected corners
            clipped_uv_verts: A list of clipped vertices

        Returns:
            A new list of clipped vertices based on the selected corners
        """
        prev = selected_corners[-1]
        for corner in selected_corners:
            # interpolate line segments to the image border
            clip_prev, clip_corner = clip_segment_v3_plane_n(
                copy.deepcopy(prev), copy.deepcopy(corner), copy.deepcopy(planes)
            )
            prev = corner
            if clip_prev is None or clip_corner is None:
                continue
            a, b = clip_line_segment(clip_prev, clip_corner, K)
            clipped_uv_verts = np.vstack([clipped_uv_verts, a[:2].reshape(-1, 2)])
            clipped_uv_verts = np.vstack([clipped_uv_verts, b[:2].reshape(-1, 2)])

        return clipped_uv_verts

    clipped_uv_verts = np.zeros((0, 2))
    # Draw the sides
    for i in range(4):
        corner_f = corners[i]  # front corner
        corner_b = corners[i + 4]  # back corner

        clip_c_f, clip_c_b = clip_segment_v3_plane_n(corner_f, corner_b, planes)
        if clip_c_f is None or clip_c_b is None:
            continue
        a, b = clip_line_segment(clip_c_f, clip_c_b, K)

        clipped_uv_verts = np.vstack([clipped_uv_verts, a[:2].reshape(-1, 2)])
        clipped_uv_verts = np.vstack([clipped_uv_verts, b[:2].reshape(-1, 2)])

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    front_verts = clip_rect(corners[:4], clipped_uv_verts)
    back_verts = clip_rect(corners[4:], clipped_uv_verts)

    clipped_uv_verts = np.vstack([clipped_uv_verts, front_verts.reshape(-1, 2)])
    clipped_uv_verts = np.vstack([clipped_uv_verts, back_verts.reshape(-1, 2)])

    if clipped_uv_verts.shape[0] == 0:
        return None

    bbox_2d = compute_point_cloud_bbox(clipped_uv_verts)
    return bbox_2d
