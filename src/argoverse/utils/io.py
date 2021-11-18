"""I/O for manipulating AV2."""

from typing import Optional, Tuple

import pandas as pd
from pyarrow import feather

from argoverse.utils.typing import PathLike


def read_feather(path: PathLike, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    return feather.read_feather(path, columns=columns)
