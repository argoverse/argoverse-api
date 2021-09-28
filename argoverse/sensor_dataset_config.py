from dataclasses import dataclass
from typing import Optional

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
        """Check to see if metadata regarding a camera of interest is present."""
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


@dataclass(frozen=True)
class ArgoverseConfig(SensorDatasetConfig):
    RING_CAMERA_HEIGHT: int = 1200
    RING_CAMERA_WIDTH: int = 1920

    STEREO_CAMERA_HEIGHT: int = 2056
    STEREO_CAMERA_WIDTH: int = 2464

    dataset_name: str = "argoverse-v1.1"
    ring_cam_fps: int = 30
    stereo_cam_fps: int = 5

    camera_sensors: SensorSuiteConfig = SensorSuiteConfig(
        ring_front_center=SensorConfig(RING_CAMERA_HEIGHT, RING_CAMERA_WIDTH, "ring_front_center"),
        ring_front_left=SensorConfig(RING_CAMERA_HEIGHT, RING_CAMERA_WIDTH, "ring_front_left"),
        ring_front_right=SensorConfig(RING_CAMERA_HEIGHT, RING_CAMERA_WIDTH, "ring_front_right"),
        ring_side_left=SensorConfig(RING_CAMERA_HEIGHT, RING_CAMERA_WIDTH, "ring_side_left"),
        ring_side_right=SensorConfig(RING_CAMERA_HEIGHT, RING_CAMERA_WIDTH, "ring_side_right"),
        ring_rear_left=SensorConfig(RING_CAMERA_HEIGHT, RING_CAMERA_WIDTH, "ring_rear_left"),
        ring_rear_right=SensorConfig(RING_CAMERA_HEIGHT, RING_CAMERA_WIDTH, "ring_rear_right"),
        stereo_front_left=SensorConfig(STEREO_CAMERA_HEIGHT, STEREO_CAMERA_WIDTH, "stereo_front_left"),
        stereo_front_right=SensorConfig(STEREO_CAMERA_HEIGHT, STEREO_CAMERA_WIDTH, "stereo_front_right"),
        stereo_front_left_rect=SensorConfig(STEREO_CAMERA_HEIGHT, STEREO_CAMERA_WIDTH, "stereo_front_left_rect"),
        stereo_front_right_rect=SensorConfig(STEREO_CAMERA_HEIGHT, STEREO_CAMERA_WIDTH, "stereo_front_right_rect"),
    )
