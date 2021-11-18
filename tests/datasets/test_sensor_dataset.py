"""Unit tests for sensor-dataset dataloader."""
from pathlib import Path

from argoverse.datasets.sensor.dataset import SensorDataset


def test_sensor_dataset() -> None:
    rootdir = Path("data")
    SensorDataset(rootdir)


if __name__ == "__main__":
    pass
