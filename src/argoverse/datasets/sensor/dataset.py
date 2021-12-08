"""Dataloader for the Argoverse 2 (AV2) sensor dataset."""
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from argoverse.datasets.dataset import Dataset
from argoverse.datasets.sensor.constants import INDEX_KEYS
from argoverse.utils.io import read_feather

logger = logging.Logger(__name__)


class DataloaderMode(Enum):
    DETECTION = "DETECTION"
    TRACKING = "TRACKING"


@dataclass
class SensorDataset(Dataset):

    mode: DataloaderMode
    index_names: Tuple[str, ...] = INDEX_KEYS

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.metadata = self._get_metadata()
        self.num_logs = self.metadata["log_id"].unique().shape[0]
        self.num_sweeps = self.metadata[self.metadata["sensor_name"] == "lidar"].shape[0]


    def _get_metadata(self) -> pd.DataFrame:
        keys: List[Dict[str, Union[int, str]]] = []
        lidar_pattern = "*/sensors/lidar/*.feather"
        lidar_paths: List[Path] = sorted(self.root_dir.glob(lidar_pattern))
        for lp in lidar_paths:
            key = _get_key(lp, log_parent_idx=2)
            keys.append(key)

        camera_pattern = "*/sensors/cameras/*/*.jpg"
        camera_paths: List[Path] = sorted(self.root_dir.glob(camera_pattern))
        for cp in camera_paths:
            key = _get_key(cp, log_parent_idx=3)
            keys.append(key)
        return pd.DataFrame(keys)

    def __len__(self) -> int:
        if self.mode == DataloaderMode.DETECTION:
            return self.num_sweeps
        elif self.mode == DataloaderMode.TRACKING:
            return self.num_logs
        raise NotImplementedError(f"{self.mode} is not implemented!")

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """Get an item from the dataset.

        [extended_summary]

        Args:
            idx (int): Index in [0, n - 1] where n
                is the length of the entire dataset.

        Returns:
            DataFrame: Tabular representation of the item.
        """
        keys = list(map(str, self.metadata.iloc[idx].tolist()))
        keys.insert(1, "sensors")

        key_path = Path(*keys).with_suffix(".feather")
        lidar_path = self.root_dir / key_path
        return read_feather(lidar_path)


def _get_key(path: Path, log_parent_idx: int) -> Dict[str, Union[int, str]]:
    log_id = path.parents[log_parent_idx].stem
    sensor_name = path.parent.stem
    tov_ns = int(path.stem)
    return {"log_id": log_id, "sensor_name": sensor_name, "tov_ns": tov_ns}
