"""Dataloading class for a sensor-based dataset."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple
import pandas as pd

import pyarrow.dataset as ds

from argoverse.io.loading import load_dataset


@dataclass
class SensorDataset:
    rootdir: Path

    lidar: Optional[ds.Dataset] = None
    poses: Optional[ds.Dataset] = None

    calibration: Optional[ds.Dataset] = None
    labels: Optional[ds.Dataset] = None

    def __post_init__(self, format: str = "feather") -> None:
        dtypes = ("lidar", "poses", "calibration", "labels")
        for dtype in dtypes:
            src = (self.rootdir / dtype).with_suffix(f".{format}")
            if src.exists():
                setattr(self, dtype, load_dataset(src).get_fragments())

    def __len__(self) -> int:
        if self.lidar is None:
            return 0
        return len(self.lidar)

    def __iter__(self) -> Iterator[Tuple[int, pd.DataFrame]]:
        item = next(self.lidar)
        tov = int(Path(item.path).parent.stem)
        yield tov, item.to_table().to_pandas()
