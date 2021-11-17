"""I/O for manipulating AV2."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from pyarrow import feather


def read_feather(path: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    return feather.read_feather(path, columns=columns)
