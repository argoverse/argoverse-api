"""Dataloader for the Argoverse 2 (AV2) sensor dataset."""
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from argoverse.datasets.dataset import Dataset
from argoverse.datasets.sensor.constants import INDEX_KEYS
from argoverse.utils.distributed import compute_chunksize, parallelize
from argoverse.utils.io import read_feather

logger = logging.Logger(__name__)


class DataloaderMode(Enum):
    DETECTION = "DETECTION"
    TRACKING = "TRACKING"


@dataclass
class SensorDataset(Dataset):

    mode: DataloaderMode
    index_names: Tuple[str, ...] = INDEX_KEYS

    with_cache: bool = True
    with_annotations: bool = True
    with_imagery: bool = True

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        self.metadata = self._get_metadata()

        self.num_logs = len(self.metadata.index.unique("log_id"))
        self.num_sweeps = self.metadata.index.get_level_values("sensor_name").value_counts()["lidar"]
        self.synchronized_metadata: Optional[pd.DataFrame] = None

        if self.with_imagery:
            sync_path = Path(self.root_dir, "_sync_db")
            if self.with_cache and sync_path.exists():
                self.synchronized_metadata = read_feather(sync_path)
            else:
                log_ids = self.metadata.index.unique(level="log_id")

                synchronized_metadata: List[pd.DataFrame] = []
                for log_id in log_ids:
                    log_data = self.metadata.loc[log_id]

                    sensors: List[str] = log_data.index.unique(level="sensor_name").values.tolist()
                    sensors.remove("lidar")

                    reference_sensor = log_data.loc["lidar"]
                    for sensor in sensors:
                        target_sensor = log_data.loc[sensor]
                        target_sensor[sensor] = target_sensor["tov_ns"]

                        reference_sensor = pd.merge_asof(
                            reference_sensor, target_sensor, on="tov_ns", direction="nearest"
                        )
                    reference_sensor.insert(0, "log_id", log_id)
                    synchronized_metadata.append(reference_sensor)
                self.synchronized_metadata = pd.concat(synchronized_metadata).reset_index(drop=True)
                self.synchronized_metadata.to_feather(str(sync_path))
            self.synchronized_metadata = self.synchronized_metadata.set_index(["log_id", "tov_ns"])

    def _get_metadata(self) -> pd.DataFrame:
        metadata_path = Path(self.root_dir, "_metadata")
        if self.with_cache and metadata_path.exists():
            return read_feather(metadata_path).set_index(["log_id", "sensor_name"])
        lidar_pattern = "*/sensors/lidar/*.feather"
        lidar_paths: List[Path] = sorted(self.root_dir.glob(lidar_pattern))

        logger.info("Loading lidar data ...")
        items = parallelize(
            _get_key, lidar_paths, chunksize=compute_chunksize(len(lidar_paths)), with_progress_bar=True
        )
        lidar_keys, lidar_data = list(zip(*items))

        camera_pattern = "*/sensors/cameras/*/*.jpg"
        camera_paths: List[Path] = sorted(self.root_dir.glob(camera_pattern))

        logger.info("Loading camera data ...")
        results = parallelize(
            _get_key, camera_paths, chunksize=compute_chunksize(len(camera_paths)), with_progress_bar=True
        )
        camera_keys, camera_data = list(zip(*results))

        keys = lidar_keys + camera_keys
        data = lidar_data + camera_data
        index: pd.MultiIndex = pd.MultiIndex.from_tuples(keys, names=["log_id", "sensor_name"], sortorder=0)
        metadata = pd.DataFrame(index=index, data=data)
        metadata.reset_index().to_feather(str(metadata_path))
        return metadata

    def __len__(self) -> int:
        if self.mode == DataloaderMode.DETECTION:
            return self.num_sweeps
        elif self.mode == DataloaderMode.TRACKING:
            return self.num_logs
        raise NotImplementedError(f"{self.mode} is not implemented!")

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """Get an item from the dataset.

        [extended_summary]

        Args:
            idx (int): Index in [0, n - 1] where n
                is the length of the entire dataset.

        Returns:
            DataFrame: Tabular representation of the item.
        """
        record = self.metadata.xs("lidar", level=1).iloc[idx]
        log_id = str(record.name)
        tov_ns = int(record[0])

        sensors_root = Path(self.root_dir, log_id, "sensors")
        lidar_path = Path(sensors_root, "lidar", str(tov_ns)).with_suffix(".feather")
        lidar = read_feather(lidar_path)

        datum: Dict[str, Union[Dict[str, str], np.ndarray, pd.DataFrame]] = {
            "lidar": lidar,
            "metadata": {"log_id": log_id},
        }

        with_annotations = False
        if with_annotations:
            annotations_path = Path(self.root_dir, log_id, "annotations.feather")
            annotations = read_feather(annotations_path)
            sweep_annotations = annotations[annotations["tov_ns"] == tov_ns]
            datum |= {"annotations": sweep_annotations}

        if self.with_imagery:
            synchronized_record = self.synchronized_metadata.loc[(log_id, tov_ns)]
            sensor_data: Dict[str, np.ndarray] = {}
            for sensor, tov_ns in synchronized_record.items():
                sensor_path = Path(sensors_root, "cameras", sensor, str(tov_ns)).with_suffix(".jpg")
                sensor_data[sensor] = cv2.imread(str(sensor_path))
            datum |= sensor_data
        return datum


def _get_key(path: Path) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    sensor_name = path.parent.stem

    idx = 3
    if sensor_name == "lidar":
        idx = 2

    log_id = path.parents[idx].stem
    sensor_name = path.parent.stem
    tov_ns = int(path.stem)
    return (log_id, sensor_name), {"tov_ns": tov_ns}
