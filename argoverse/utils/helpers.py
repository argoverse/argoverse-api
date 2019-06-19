# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import Optional, Sequence

import numpy as np


def assert_np_array_shape(array: np.ndarray, target_shape: Sequence[Optional[int]]) -> None:
    """Check for shape correctness.

    Args:
        array: array to check dimensions of.
        target_shape: desired shape. use None for unknown dimension sizes.

    Raises:
        ValueError: if array's shape does not match target_shape for any of the specified dimensions.
    """
    for index_dim, (array_shape_dim, target_shape_dim) in enumerate(zip(array.shape, target_shape)):
        if target_shape_dim and array_shape_dim != target_shape_dim:
            raise ValueError(f"array.shape[{index_dim}]: {array_shape_dim} != {target_shape_dim}.")
