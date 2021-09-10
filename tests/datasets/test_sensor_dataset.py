"""Unit tests for sensor-dataset dataloader."""
from pathlib import Path

from argoverse.datasets.sensor_dataset import SensorDataset


def test_sensor_dataset() -> None:
    rootdir = Path("data")
    sensor_dataset = SensorDataset(rootdir)
    pass


if __name__ == "__main__":
    pass
