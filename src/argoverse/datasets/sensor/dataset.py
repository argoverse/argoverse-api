"""General dataloader for Argoverse 2.0 datasets."""
import logging
from dataclasses import dataclass
from typing import Tuple

from polars import col
from polars.eager import DataFrame

from argoverse.datasets.dataset import Dataset
from argoverse.datasets.sensor.constants import INDEX_KEYS

logger = logging.Logger(__name__)


@dataclass
class SensorDataset(Dataset):

    index_names: Tuple[str, ...] = INDEX_KEYS

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

    def __getitem__(self, idx: int) -> DataFrame:
        datum = super().__getitem__(idx)
        return datum

    def get_log_ids(self) -> DataFrame:
        return self.metadata["log_id"].unique().sort().to_frame()

    def get_lidar_paths(self) -> DataFrame:
        return self.get_paths(col("record_type") == "lidar")
