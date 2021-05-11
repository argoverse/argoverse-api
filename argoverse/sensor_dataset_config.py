from typing import NamedTuple, Optional

from hydra.experimental import compose, initialize_config_module
from hydra.utils import instantiate


class SensorConfig(NamedTuple):
    """Image dimensions for each sensor are provided in pixels."""

    img_height: int
    img_width: int


class SensorSuiteConfig(NamedTuple):
    """Contains information about image dimensions for each sensor."""

    ring_front_center: SensorConfig
    ring_front_left: SensorConfig
    ring_front_right: SensorConfig
    ring_side_left: SensorConfig
    ring_side_right: SensorConfig
    ring_rear_left: SensorConfig
    ring_rear_right: SensorConfig
    stereo_front_right: Optional[SensorConfig]
    stereo_front_left: Optional[SensorConfig]


class SensorDatasetConfig(NamedTuple):
    """Global constants regarding frame rate and image dimensions."""

    dataset_name: str
    ring_cam_fps: int
    stereo_cam_fps: int
    sensors: SensorSuiteConfig


DATASET_NAME = "argoverse1.1"

with initialize_config_module(config_module="argoverse.config"):
    cfg = compose(config_name=f"{DATASET_NAME}.yml")
    ArgoverseConfig: SensorDatasetConfig = instantiate(cfg.SensorDatasetConfig)
