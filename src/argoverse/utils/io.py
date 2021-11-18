# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Helper functions for deserializing AV2 data."""

from typing import Optional, Tuple

import pandas as pd
from pyarrow import feather

from argoverse.utils.typing import PathLike


def read_feather(path: PathLike, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """Read Apache Feather data from a .feather file.

    AV2 uses .feather to serialize much of its data. This function handles the deserialization
    process and returns a `pandas` DataFrame with rows corresponding to the records and the
    columns corresponding to the record attributes.

    Args:
        path (PathLike): Source data file (e.g., 'lidar.feather', 'calibration.feather', etc.)
        columns (Optional[Tuple[str, ...]], optional): Tuple of columns to load for the given record. Defaults to None.

    Returns:
        pd.DataFrame: (N,len(columns)) Apache Feather data represented as a `pandas` DataFrame.
    """
    return feather.read_feather(path, columns=columns)
