from typing import NamedTuple, Optional

import hydra
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

with hydra.initialize_config_module(config_module="argoverse.config"):
    cfg = hydra.compose(config_name=f"{DATASET_NAME}.yaml")
    ArgoverseConfig: SensorDatasetConfig = instantiate(cfg.SensorDatasetConfig)
