"""Example script for loading data from the AV2 sensor dataset."""

from pathlib import Path

from argoverse.datasets.sensor import SensorDataset
from argoverse.utils.pathlib import HOME


def main(dataset_dir: Path) -> None:
    SensorDataset(dataset_dir)

    breakpoint()


if __name__ == "__main__":
    dataset_dir = HOME / "data" / "datasets" / "av2" / "processed"
    main(dataset_dir)
