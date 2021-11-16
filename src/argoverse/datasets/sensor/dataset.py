"""Dataloader for the Argoverse 2 (AV2) sensor dataset."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from argoverse.datasets.dataset import Dataset
from argoverse.datasets.sensor.constants import INDEX_KEYS

logger = logging.Logger(__name__)


@dataclass
class SensorDataset(Dataset):

    index_names: Tuple[str, ...] = INDEX_KEYS

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        lidar_paths = sorted(self.root_dir.glob("*/sensors/lidar/*.feather"))
        camera_paths = sorted(self.root_dir.glob("*/*/sensors/cameras/*/*.jpg"))
        sensor_paths = lidar_paths + camera_paths

        # metadata: List[Tuple[str, ...]] = []
        # for sensor_path in sensor_paths:
        #     record_type = sensor_path.parent.name
        #     log_id = sensor_path.parent.parent.parent.stem
        #     split = sensor_path.parent.parent.parent.parent.stem
        #     metadata.append((split, log_id, record_type))
        # self.metadata = pd.DataFrame(metadata, columns=["split", "log_id", "record_type"])
        breakpoint()

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """Get an item from the dataset.

        [extended_summary]

        Args:
            idx (int): Index in [0, n - 1] where n
                is the length of the entire dataset.

        Returns:
            DataFrame: Tabular representation of the item.
        """
        datum = super().__getitem__(idx)
        return datum
