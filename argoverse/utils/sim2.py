"""
Utility for 2d rigid body transformations with scaling.

References:
    http://ethaneade.com/lie_groups.pdf
    https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Similarity3.h
"""

import json
import os
from typing import Union

import numpy as np

from argoverse.utils.helpers import assert_np_array_shape
from argoverse.utils.json_utils import save_json_dict

_PathLike = Union[str, "os.PathLike[str]"]


class Sim2:
    """Implements the Similarity(2) class."""

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

    @property
    def theta_deg(self) -> float:
        """Recover the rotation angle `theta` (in degrees) from the 2d rotation matrix.

        Note: the first column of the rotation matrix R provides sine and cosine of theta,
            since R is encoded as [c,-s]
                                  [s, c]

        We use the following identity: tan(theta) = s/c = (opp/hyp) / (adj/hyp) = opp/adj
        """
        c, s = self.R_[0, 0], self.R_[1, 0]
        theta_rad = np.arctan2(s, c)
        return float(np.rad2deg(theta_rad))

    def __repr__(self) -> str:
        """Return a human-readable string representation of the class."""
        return f"Angle (deg.): {self.theta_deg:.1f}, Trans.: {np.round(self.t_,2)}, Scale: {self.s_:.1f}"

    def __eq__(self, other: object) -> bool:
        """Check for equality with other Sim(2) object."""
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
        """Return the 2x2 rotation matrix."""
        return self.R_

    @property
    def translation(self) -> np.ndarray:
        """Return the (2,) translation vector."""
        return self.t_

    @property
    def scale(self) -> float:
        """Return the scale."""
        return self.s_

    @property
    def matrix(self) -> np.ndarray:
        """Calculate 3*3 matrix group equivalent."""
        T = np.zeros((3, 3))
        T[:2, :2] = self.R_
        T[:2, 2] = self.t_
        T[2, 2] = 1 / self.s_
        return T

    def compose(self, S: "Sim2") -> "Sim2":
        """Composition with another Sim2.

        This can be understood via block matrix multiplication, if self is parameterized as (R1,t1,s1)
        and if `S` is parameterized as (R2,t2,s2):

        [R1  t1]   [R2  t2]   [R1 @ R2   R1@t2 + t1/s2]
        [0 1/s1] @ [0 1/s2] = [ 0          1/(s1*s2)  ]
        """
        # fmt: off
        return Sim2(
            R=self.R_ @ S.R_,
            t=self.R_ @ S.t_ + ((1.0 / S.s_) * self.t_),
            s=self.s_ * S.s_
        )
        # fmt: on

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
        if not point_cloud.ndim == 2:
            raise ValueError("Input point cloud is not 2-dimensional.")
        assert_np_array_shape(point_cloud, (None, 2))
        # (2,2) x (2,N) + (2,1) = (2,N) -> transpose
        transformed_point_cloud = (point_cloud @ self.R_.T) + self.t_

        # now scale points
        return transformed_point_cloud * self.s_

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Alias for `transform_from()`, for synchrony w/ API provided by SE(2) and SE(3) classes."""
        return self.transform_from(point_cloud)

    def save_as_json(self, save_fpath: _PathLike) -> None:
        """Save the Sim(2) object to a JSON representation on disk.

        Args:
            save_fpath: path to where json file should be saved
        """
        dict_for_serialization = {
            "R": self.rotation.flatten().tolist(),
            "t": self.translation.flatten().tolist(),
            "s": self.scale,
        }
        save_json_dict(save_fpath, dict_for_serialization)

    @classmethod
    def from_json(cls, json_fpath: _PathLike) -> "Sim2":
        """Generate class inst. from a JSON file containing Sim(2) parameters as flattened matrices (row-major)."""
        with open(json_fpath, "r") as f:
            json_data = json.load(f)

        R = np.array(json_data["R"]).reshape(2, 2)
        t = np.array(json_data["t"]).reshape(2)
        s = float(json_data["s"])
        return cls(R, t, s)

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Sim2":
        """Generate class instance from a 3x3 Numpy matrix."""
        if np.isclose(T[2, 2], 0.0):
            raise ZeroDivisionError("Sim(2) scale calculation would lead to division by zero.")

        R = T[:2, :2]
        t = T[:2, 2]
        s = 1 / T[2, 2]
        return cls(R, t, s)
