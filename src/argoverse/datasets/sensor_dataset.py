from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow.dataset as ds


@dataclass
class SensorDataset:
    rootdir: Path

    lidar: Optional[ds.Dataset]
    poses: Optional[ds.Dataset]

    calibration: Optional[ds.Dataset] = None
    labels: Optional[ds.Dataset] = None

    def __post_init__(self, format: str = "feather") -> None:
        lidar_fpath = self.rootdir / "lidar.feather"
        poses_fpath = self.rootdir / "poses.feather"
        calibration_fpath = self.rootdir / "calibration.feather"
        labels_fpath = self.rootdir / "labels.feather"

        self.lidar = ds.dataset(lidar_fpath, format=format)
        self.poses = ds.dataset(poses_fpath, format=format)
        self.calibration = ds.dataset(calibration_fpath, format=format)
        self.labels = ds.dataset(labels_fpath, format=format)

    def __len__(self) -> int:
        return len(self.lidar)
