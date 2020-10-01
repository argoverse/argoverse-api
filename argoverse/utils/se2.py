# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Module for `SE2`."""

import numpy as np

from argoverse.utils.helpers import assert_np_array_shape


class SE2:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Initialize.

        Args:
            rotation: np.ndarray of shape (2,2).
            translation: np.ndarray of shape (2,1).

        Raises:
            ValueError: if rotation or translation do not have the required shapes.
        """
        assert_np_array_shape(rotation, (2, 2))
        assert_np_array_shape(translation, (2,))
        self.rotation = rotation
        self.translation = translation
        self.transform_matrix = np.eye(3)
        self.transform_matrix[:2, :2] = self.rotation
        self.transform_matrix[:2, 2] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(2) transformation to point_cloud.

        Args:
            point_cloud: np.ndarray of shape (N, 2).

        Returns:
            transformed_point_cloud: np.ndarray of shape (N, 2).

        Raises:
            ValueError: if point_cloud does not have the required shape.
        """
        assert_np_array_shape(point_cloud, (None, 2))
        num_points = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_points, 1))])
        transformed_point_cloud = homogeneous_pts.dot(self.transform_matrix.T)
        return transformed_point_cloud[:, :2]

    def inverse(self) -> "SE2":
        """Return the inverse of the current SE2 transformation.

        For example, if the current object represents target_SE2_src, we will return instead src_SE2_target.

        Returns:
            inverse of this SE2 transformation.
        """
        return SE2(rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation))

    def inverse_transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Transform the point_cloud by the inverse of this SE2.

        Args:
            point_cloud: Numpy array of shape (N,2).

        Returns:
            point_cloud transformed by the inverse of this SE2.
        """
        return self.inverse().transform_point_cloud(point_cloud)

    def right_multiply_with_se2(self, right_se2: "SE2") -> "SE2":
        """Multiply this SE2 from right by right_se2 and return the composed transformation.

        Args:
            right_se2: SE2 object to multiply this object by from right.

        Returns:
            The composed transformation.
        """
        chained_transform_matrix = self.transform_matrix.dot(right_se2.transform_matrix)
        chained_se2 = SE2(
            rotation=chained_transform_matrix[:2, :2],
            translation=chained_transform_matrix[:2, 2],
        )
        return chained_se2
