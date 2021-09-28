"""Example script for loading data from the Argoverse 2.0 sensor dataset."""
from pathlib import Path

from argoverse.datasets.sensor.dataset import SensorDataset
from argoverse.io.loading import read_feather
from argoverse.utils.pathlib import HOME


def main(dirname: Path) -> None:
    """Entry function for exploring the Argoverse 2.0 dataset.

    Args:
        dirname (Path): Directory name.
    """
    dataset = SensorDataset(dirname)

    for path in dataset.get_lidar_paths():
        datum = read_feather(path, dataset.index_names)


if __name__ == "__main__":
    dirname = HOME / "data" / "datasets" / "argoverse-v2"
    main(dirname)
