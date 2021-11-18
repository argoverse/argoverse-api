"""Example script for loading data from the AV2 sensor dataset."""

from os import PathLike
from pathlib import Path

from argoverse.datasets.sensor import SensorDataset
from argoverse.utils.constants import HOME


def main(dataset_dir: PathLike) -> None:
    SensorDataset(dataset_dir)

    breakpoint()


if __name__ == "__main__":
    dataset_dir = HOME / "data" / "datasets" / "av2" / "processed"
    main(dataset_dir)
