# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""SE3 class for point cloud rotation and translation."""

import numpy as np


class SE3:
    """An SE3 class allows point cloud rotation and translation operations."""

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Initialize an SE3 instance with its rotation and translation matrices.

        Args:
            rotation: Array of shape (3, 3)
            translation: Array of shape (3,)
        """
        assert rotation.shape == (3, 3)
        assert translation.shape == (3,)
        self.rotation = rotation
        self.translation = translation

        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3, :3] = self.rotation
        self.transform_matrix[:3, 3] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(3) transformation to this point cloud.

        Args:
            point_cloud: Array of shape (N, 3)

        Returns:
            transformed_point_cloud: Array of shape (N, 3)
        """
        # convert to homogeneous
        num_pts = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_pts, 1))])
        transformed_point_cloud = homogeneous_pts.dot(self.transform_matrix.T)
        return transformed_point_cloud[:, :3]

    def inverse_transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Undo the translation and then the rotation (Inverse SE(3) transformation)."""
        point_cloud = point_cloud.copy()
        point_cloud -= self.translation
        return point_cloud.dot(self.rotation)

    def inverse(self) -> "SE3":
        """Return the inverse of the current SE3 transformation.

        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.

        Returns:
            src_SE3_target: instance of SE3 class, representing
                inverse of SE3 transformation target_SE3_src
        """
        return SE3(rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation))

    def right_multiply_with_se3(self, right_se3: "SE3") -> "SE3":
        """Right multiply the transformation matrix with another SE3 instance.

        Algebraic representation: chained_se3 = T * right_se3

        Args:
            right_se3: instance of SE3 class

        Returns:
            chained_se3: instance of SE3 class
        """
        chained_transform_matrix = self.transform_matrix.dot(right_se3.transform_matrix)
        chained_se3 = SE3(
            rotation=chained_transform_matrix[:3, :3],
            translation=chained_transform_matrix[:3, 3],
        )
        return chained_se3
