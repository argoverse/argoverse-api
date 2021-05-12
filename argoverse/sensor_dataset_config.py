from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import MISSING


@dataclass(frozen=True)
class SensorConfig:
    """Image dimensions for each camera sensor are provided in pixels."""

    img_height: int = MISSING
    img_width: int = MISSING
    name: str = MISSING


@dataclass(frozen=True)
class SensorSuiteConfig:
    """Contains information about image dimensions for each camera sensor."""

    ring_front_center: SensorConfig = MISSING
    ring_front_left: SensorConfig = MISSING
    ring_front_right: SensorConfig = MISSING
    ring_side_left: SensorConfig = MISSING
    ring_side_right: SensorConfig = MISSING
    ring_rear_left: SensorConfig = MISSING
    ring_rear_right: SensorConfig = MISSING
    stereo_front_right: Optional[SensorConfig] = None
    stereo_front_left: Optional[SensorConfig] = None
    stereo_front_right_rect: Optional[SensorConfig] = None
    stereo_front_left_rect: Optional[SensorConfig] = None

    def has_camera(self, camera_name: str) -> bool:
        """Check to see if the query camera """
        if camera_name not in self.__dict__.keys():
            return False
        # ensure field is not empty
        return getattr(self, camera_name) is not None


@dataclass(frozen=True)
class SensorDatasetConfig:
    """Global constants regarding frame rate and image dimensions."""

    dataset_name: str = MISSING
    ring_cam_fps: int = MISSING
    stereo_cam_fps: int = MISSING
    camera_sensors: SensorSuiteConfig = MISSING


DATASET_NAME = "argoverse-v1.1"

with hydra.initialize_config_module(config_module="argoverse.config"):
    cfg = hydra.compose(config_name=f"{DATASET_NAME}.yaml")
    ArgoverseConfig: SensorDatasetConfig = instantiate(cfg.SensorDatasetConfig)
