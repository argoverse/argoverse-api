"""I/O for manipulating the Argoverse 2.0 dataset."""
from pathlib import Path
from typing import Tuple

import pandas as pd
from polars import from_pandas
from polars.eager import DataFrame
from polars.io import read_ipc


def read_feather(path: Path, index_names: Tuple[str, ...]) -> DataFrame:
    if "lidar" in str(path):
        return read_lidar(path)

    data = read_ipc(path, use_pyarrow=False)
    for i, column_name in enumerate(index_names[::-1]):
        parent = path.parents[i]
        data[column_name] = [parent.stem for _ in range(len(data))]
    return data


def read_lidar(path: Path) -> DataFrame:
    return from_pandas(pd.read_feather(path).astype(float))
