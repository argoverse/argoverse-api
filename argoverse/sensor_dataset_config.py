from typing import NamedTuple, Optional

from hydra.experimental import compose, initialize_config_module
from hydra.utils import instantiate


class ImageDimensions(NamedTuple):
    """Dimensions are provided in pixels."""

    height: int
    width: int


class SensorDimensions(NamedTuple):
    """Contains information about image dimensions for each sensor."""

    ring_front_center: ImageDimensions
    ring_front_left: ImageDimensions
    ring_front_right: ImageDimensions
    ring_side_left: ImageDimensions
    ring_side_right: ImageDimensions
    ring_rear_left: ImageDimensions
    ring_rear_right: ImageDimensions
    stereo_front_right: Optional[ImageDimensions]
    stereo_front_left: Optional[ImageDimensions]


class SensorDatasetConfig(NamedTuple):
    """Global constants regarding frame rate and image dimensions."""

    dataset_name: str
    ring_cam_fps: int
    stereo_cam_fps: int
    sensors: SensorDimensions


DATASET_NAME = "argoverse1.1"

with initialize_config_module(config_module="argoverse.config"):
    cfg = compose(config_name=f"{DATASET_NAME}.yml")
    ArgoverseConfig: SensorDatasetConfig = instantiate(cfg.SensorDatasetConfig)
