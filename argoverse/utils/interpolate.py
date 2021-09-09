# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import Optional, Tuple, cast

import numpy as np

# For a single line segment
NUM_CENTERLINE_INTERP_PTS = 10


def compute_lane_width(left_even_pts: np.ndarray, right_even_pts: np.ndarray) -> float:
    """Compute the width of a lane, given an explicit left and right boundary.
    
    Requires an equal number of waypoints on each boundary. For 3d polylines, this incorporates
    the height difference between the left and right polyline into the lane width as a hypotenuse
    of triangle formed by lane width in a flat plane, and the height difference.

    Args:
        left_even_pts: Numpy array of shape (N,2) or (N,3)
        right_even_pts: Numpy array of shape (N,2) or (N,3)

    Returns:
        lane_width: float representing average width of a lane
    """
    assert left_even_pts.shape == right_even_pts.shape
    lane_width = cast(float, np.mean(np.linalg.norm(left_even_pts - right_even_pts, axis=1)))
    return lane_width


def compute_mid_pivot_arc(single_pt: np.ndarray, arc_pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Given a line of points on one boundary, and a single point on the other side,
    produce the middle arc we get by pivoting around the single point.

    Occurs when mapping cul-de-sacs.

    Args:
        single_pt: Numpy array of shape (2,) representing a single 2d coordinate.
        arc_pts: Numpy array of shape (N,2) representing a 2d polyline.

    Returns:
        centerline_pts: Numpy array of shape (N,3)
        lane_width: average width of the lane.
    """
    num_pts = len(arc_pts)
    # form ladder with equal number of vertices on each side
    single_pt_tiled = np.tile(single_pt, (num_pts, 1))
    # compute midpoint for each rung of the ladder
    centerline_pts = (single_pt_tiled + arc_pts) / 2.0
    lane_width = compute_lane_width(single_pt_tiled, arc_pts)
    return centerline_pts, lane_width


def compute_midpoint_line(
    left_ln_bnds: np.ndarray,
    right_ln_bnds: np.ndarray,
    num_interp_pts: int = NUM_CENTERLINE_INTERP_PTS,
) -> Tuple[np.ndarray, float]:
    """
    Compute the lane segment centerline by interpolating n points along each
    boundary, and then averaging left and right waypoints.

    Note that the number of input waypoints along the left and right boundaries
    can be vastly different -- consider cul-de-sacs, for example.

    Args:
        left_ln_bnds: Numpy array of shape (M,2)
        right_ln_bnds: Numpy array of shape (N,2)
        num_interp_pts: number of midpoints to compute for this lane segment,
            except if it is a cul-de-sac, in which case the number of midpoints
            will be equal to max(M,N).

    Returns:
        centerline_pts:
    """
    if left_ln_bnds.ndim != 2 or right_ln_bnds.ndim != 2:
        raise ValueError("Left and right lane boundaries must consistent of a sequence of 2d or 3d waypoints.")

    dim = left_ln_bnds.shape[1]
    if dim not in [2,3]:
        raise ValueError("Left and right lane boundaries must be 2d or 3d.")

    if left_ln_bnds.shape[1] != right_ln_bnds.shape[1]:
        raise ValueError("Left ")

    if dim == 2:
        # first, remove duplicates (might only be left with a single pt afterwards)
        px, py = eliminate_duplicates_2d(left_ln_bnds[:, 0], left_ln_bnds[:, 1])
        left_ln_bnds = np.hstack([px[:, np.newaxis], py[:, np.newaxis]])
        px, py = eliminate_duplicates_2d(right_ln_bnds[:, 0], right_ln_bnds[:, 1])
        right_ln_bnds = np.hstack([px[:, np.newaxis], py[:, np.newaxis]])

    if len(left_ln_bnds) == 1:
        centerline_pts, lane_width = compute_mid_pivot_arc(single_pt=left_ln_bnds, arc_pts=right_ln_bnds)
        return centerline_pts[:, :2], lane_width

    if len(right_ln_bnds) == 1:
        centerline_pts, lane_width = compute_mid_pivot_arc(single_pt=right_ln_bnds, arc_pts=left_ln_bnds)
        return centerline_pts[:, :2], lane_width
    
    pz_left = left_ln_bnds[:, 2] if dim == 3 else None
    left_even_pts = interp_arc(num_interp_pts, px=left_ln_bnds[:, 0], py=left_ln_bnds[:, 1], pz=pz_left)
    pz_right = right_ln_bnds[:, 2] if dim == 3 else None
    right_even_pts = interp_arc(num_interp_pts, px=right_ln_bnds[:, 0], py=right_ln_bnds[:, 1], pz=pz_right)

    centerline_pts = (left_even_pts + right_even_pts) / 2.0

    lane_width = compute_lane_width(left_even_pts, right_even_pts)
    return centerline_pts, lane_width


def get_duplicate_indices_1d(coords_1d: np.ndarray) -> np.ndarray:
    """
    Given a 1D polyline, remove consecutive duplicate coordinates.

    Args:
        coords_1d:

    Returns:
        dup_vals:
    """
    num_pts = coords_1d.shape[0]
    _, unique_inds = np.unique(coords_1d, return_index=True)
    all_inds = np.arange(num_pts)

    # Find the set difference of two arrays.
    # Return the sorted, unique values in array1 that are not in array2.
    dup_inds = np.setdiff1d(all_inds, unique_inds)
    return dup_inds


def assert_consecutive(shared_dup_inds: int, num_pts: int, coords_1d: np.ndarray) -> None:
    """
    Args:
        shared_dup_inds
        num_pts
    """
    left_dup = False
    right_dup = False

    # not at the far right edge already
    if shared_dup_inds != (num_pts - 1):
        right_dup = coords_1d[shared_dup_inds] == coords_1d[shared_dup_inds + 1]

    # not at the far left edge already
    if shared_dup_inds != 0:
        left_dup = coords_1d[shared_dup_inds] == coords_1d[shared_dup_inds - 1]

    assert left_dup or right_dup


def eliminate_duplicates_2d(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    We compare indices to ensure that deleted values are exactly
    adjacent to each other in the polyline sequence.
    """
    num_pts = px.shape[0]
    assert px.shape[0] == py.shape[0]
    px_dup_inds = get_duplicate_indices_1d(px)
    py_dup_inds = get_duplicate_indices_1d(py)
    shared_dup_inds = np.intersect1d(px_dup_inds, py_dup_inds)

    if shared_dup_inds.size > 0:
        assert shared_dup_inds.size < 2
        shared_dup_inds = int(shared_dup_inds)
        # enforce that must be consecutive indices
        assert_consecutive(shared_dup_inds, num_pts, px)
        assert_consecutive(shared_dup_inds, num_pts, py)

        px = np.delete(px, [shared_dup_inds])
        py = np.delete(py, [shared_dup_inds])

    return px, py


def interp_arc(t: int, px: np.ndarray, py: np.ndarray, pz: Optional[np.ndarray] = None) -> np.ndarray:
    """Linearly interpolate equally-spaced points along a polyline, either in 2d or 3d.

    We use a chordal parameterization so that interpolated arc-lengths
    will approximate original polyline chord lengths.
        Ref: M. Floater and T. Surazhsky, Parameterization for curve
            interpolation. 2005.
            https://www.mathworks.com/matlabcentral/fileexchange/34874-interparc

    For the 2d case, we remove duplicate consecutive points, since these have zero
    distance and thus cause division by zero in chord length computation.

    Args:
        t: number of points that will be uniformly interpolated and returned
        px: Numpy array of shape (N,), representing x-coordinates of the arc
        py: Numpy array of shape (N,), representing y-coordinates of the arc
        pz: Numpy array of shape (N,), representing z-coordinates of the arc

    Returns:
        pt: Numpy array of shape (N,2)
    """
    dim = 3 if pz is not None else 2

    if dim == 2:
        px, py = eliminate_duplicates_2d(px, py)

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # the number of points on the curve itself
    n = px.size

    # are px and py both vectors of the same length?
    assert px.size == py.size
    if pz is not None:
        assert px.size == pz.size

    if dim == 2:
        pxy = np.array((px, py)).T  # 2d polyline
    else:
        pxy = np.array((px, py, pz)).T  # 3d polyline

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen = np.linalg.norm(np.diff(pxy, axis=0), axis=1)
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength
    cumarc = np.append(0, np.cumsum(chordlen))

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins = np.digitize(eq_spaced_points, cumarc)

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = pxy[tbins - 1, :]
    # broadcast to scale each row of pxy by a different row of s
    offsets = (pxy[tbins, :] - pxy[tbins - 1, :]) * s.reshape(-1,1)
    pt = anchors + offsets

    return pt
