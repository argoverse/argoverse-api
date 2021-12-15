"""Dataloader for the Argoverse 2 (AV2) sensor dataset."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from argoverse.datasets.dataset import Dataset
from argoverse.utils.distributed import compute_chunksize, parallelize
from argoverse.utils.io import read_feather, read_im

logger = logging.Logger(__name__)


@dataclass
class SensorDataset(Dataset):

    # Number of unique lidar sweeps.
    num_sweeps: int = 0

    # Number of unique logs.
    num_logs: int = 0

    # Flag to enable file directory caching.
    # Recommended to keep this on unless you need to flush the cache.
    # This will significantly speed up dataloading.
    with_cache: bool = True

    # Flag to return annotations in the __getitem__ method.
    with_annotations: bool = True

    # Flag to load and return synchronized imagery in the __getitem__ method.
    with_imagery: bool = True

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # Load log_id, sensor_type, and time of validity (tov) information.
        self.metadata = self._load_metadata()

        # Compute the total number of unique logs in the dataset.
        self.num_logs = len(self.metadata.index.unique("log_id"))

        # Compute the number of unique lidar sweeps in the dataset.
        self.num_sweeps = self.metadata.index.get_level_values("sensor_name").value_counts()["lidar"]

        # Initialize synchronized metadata variable.
        # This is only populated when self.use_imagery is set.
        self.synchronized_metadata: Optional[pd.DataFrame] = None

        # Populate synchronization database.
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
                            reference_sensor,
                            target_sensor,
                            on="tov_ns",
                            direction="nearest",
                        )
                    reference_sensor.insert(0, "log_id", log_id)
                    synchronized_metadata.append(reference_sensor)
                self.synchronized_metadata = pd.concat(synchronized_metadata).reset_index(drop=True)
                self.synchronized_metadata.to_feather(str(sync_path))
            self.synchronized_metadata = self.synchronized_metadata.set_index(["log_id", "tov_ns"])

    def _load_metadata(self) -> pd.DataFrame:
        metadata_path = Path(self.root_dir, "_metadata")
        if self.with_cache and metadata_path.exists():
            return read_feather(metadata_path).set_index(["log_id", "sensor_name"])
        lidar_pattern = "*/sensors/lidar/*.feather"
        lidar_paths: List[Path] = sorted(self.root_dir.glob(lidar_pattern))

        logger.info("Loading lidar data ...")
        items = parallelize(
            _get_key,
            lidar_paths,
            chunksize=compute_chunksize(len(lidar_paths)),
            with_progress_bar=True,
        )
        keys, data = list(zip(*items))

        if self.with_imagery:
            camera_pattern = "*/sensors/cameras/*/*.jpg"
            camera_paths: List[Path] = sorted(self.root_dir.glob(camera_pattern))

            logger.info("Loading camera data ...")
            results = parallelize(
                _get_key,
                camera_paths,
                chunksize=compute_chunksize(len(camera_paths)),
                with_progress_bar=True,
            )
            camera_keys, camera_data = list(zip(*results))

            keys += camera_keys
            data += camera_data

        index: pd.MultiIndex = pd.MultiIndex.from_tuples(keys, names=["log_id", "sensor_name"], sortorder=0)
        metadata = pd.DataFrame(index=index, data=data)
        metadata.reset_index().to_feather(str(metadata_path))
        return metadata

    def __len__(self) -> int:
        """Return the number of lidar sweeps in the dataset

        Returns:
            int: Number of lidar sweeps.
        """
        return self.num_sweeps

    def __getitem__(self, idx: int) -> Dict[str, Union[Dict[str, str], np.ndarray, pd.DataFrame]]:
        """Get a dictionary which contains sensor data and metadata for the provided index.

        [extended_summary]

        Args:
            idx (int): Index within [0, self.num_sweeps]

        Returns:
            Dict[str, Union[np.ndarray, pd.DataFrame]]: Mapping of sensor dataset field name to the data.
        """
        record = self.metadata.xs("lidar", level=1).iloc[idx]
        log_id = str(record.name)
        tov_ns = int(record.item())

        sensors_root = Path(str(self.root_dir), log_id, "sensors")
        lidar_path = Path(sensors_root, "lidar", str(tov_ns)).with_suffix(".feather")
        lidar = read_feather(lidar_path)

        datum: Dict[str, Union[Dict[str, str], np.ndarray, pd.DataFrame]] = {
            "lidar": lidar,
            "metadata": {"log_id": log_id},
        }

        if self.with_annotations:
            annotations_path = Path(str(self.root_dir), log_id, "annotations").with_suffix(".feather")
            annotations = read_feather(annotations_path)
            sweep_annotations = annotations[annotations["tov_ns"] == tov_ns]
            datum |= {"annotations": sweep_annotations}

        if self.with_imagery:
            if self.synchronized_metadata is not None:
                index = pd.Index((log_id, str(tov_ns)))
                synchronized_record = self.synchronized_metadata.loc[index]
                sensor_data: Dict[str, np.ndarray] = {}
                for sensor, tov_ns in synchronized_record.items():
                    sensor_path = Path(sensors_root, "cameras", str(sensor), str(tov_ns)).with_suffix(".jpg")
                    sensor_data[str(sensor)] = read_im(sensor_path)
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
