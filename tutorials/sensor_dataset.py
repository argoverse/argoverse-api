"""Example script for loading data from the AV2 sensor dataset."""

from pathlib import Path

from argoverse.datasets.sensor import SensorDataset
from argoverse.datasets.sensor.dataset import DataloaderMode
from argoverse.utils.constants import HOME
from argoverse.utils.typing import PathLike


def main(dataset_dir: PathLike) -> None:
    dataset = SensorDataset(dataset_dir, DataloaderMode.DETECTION)
    for datum in dataset:
        annotations = datum["annotations"]
        lidar = datum["lidar"]
        breakpoint()


if __name__ == "__main__":
    dataset_dir = Path("/data") / "datasets" / "av2" / "sensor" / "v0002"
    main(dataset_dir)
