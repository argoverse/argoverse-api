"""
Utility for 2d rigid body transformations with scaling.

Refs:
    http://ethaneade.com/lie_groups.pdf
    https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Similarity3.h
"""

from typing import Union

import numpy as np

from argoverse.utils.helpers import assert_np_array_shape


class Sim2:
    """ Implements the Similarity(2) class."""

    def __init__(self, R: np.ndarray, t: np.ndarray, s: Union[int, float]) -> None:
        """Initialize from rotation R, translation t, and scale s.

        Args:
            R: array of shape (2x2) representing 2d rotation matrix
            t: array of shape (2,) representing 2d translation
            s: scaling factor
        """
        assert_np_array_shape(R, (2, 2))
        assert_np_array_shape(t, (2,))

        assert isinstance(s, float) or isinstance(s, int)
        if np.isclose(s, 0.0):
            raise ZeroDivisionError("3x3 matrix formation would require division by zero")

        self.R_ = R.astype(np.float32)
        self.t_ = t.astype(np.float32)
        self.s_ = float(s)

    def __eq__(self, other: object) -> bool:
        """Check for equality with other Sim(2) object"""
        if not isinstance(other, Sim2):
            return False

        if not np.isclose(self.scale, other.scale):
            return False

        if not np.allclose(self.rotation, other.rotation):
            return False

        if not np.allclose(self.translation, other.translation):
            return False

        return True

    @property
    def rotation(self) -> np.ndarray:
        """Return the 2x2 rotation matrix"""
        return self.R_

    @property
    def translation(self) -> np.ndarray:
        """Return the (2,) translation vector"""
        return self.t_

    @property
    def scale(self) -> float:
        """Return the scale."""
        return self.s_

    @property
    def matrix(self) -> np.ndarray:
        """Calculate 3*3 matrix group equivalent"""
        T = np.zeros((3, 3))
        T[:2, :2] = self.R_
        T[:2, 2] = self.t_
        T[2, 2] = 1 / self.s_
        return T

    def compose(self, S: "Sim2") -> "Sim2":
        """Composition with another Sim2."""
        return Sim2(self.R_ * S.R_, ((1.0 / S.s_) * self.t_) + self.R_ @ S.t_, self.s_ * S.s_)

    def inverse(self) -> "Sim2":
        """Return the inverse."""
        Rt = self.R_.T
        sRt = -Rt @ (self.s_ * self.t_)
        return Sim2(Rt, sRt, 1.0 / self.s_)

    def transform_from(self, point_cloud: np.ndarray) -> np.ndarray:
        """Transform point cloud such that if they are in frame A,
        and our Sim(3) transform is defines as bSa, then we get points
        back in frame B:
            p_b = bSa * p_a
        Action on a point p is s*(R*p+t).

        Args:
            point_cloud: Nx2 array representing 2d points in frame A

        Returns:
            transformed_point_cloud: Nx2 array representing 2d points in frame B
        """
        assert_np_array_shape(point_cloud, (None, 2))
        # (2,2) x (2,N) + (2,1) = (2,N) -> transpose
        transformed_point_cloud = (self.R_ @ point_cloud.T + self.t_.reshape(2, 1)).T

        # now scale points
        return transformed_point_cloud * self.s_

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Alias for `transform_from()`, for synchrony w/ API provided by SE(2) and SE(3) classes."""
        return self.transform_from(point_cloud)
