import hydra
from hydra.utils import instantiate

from argoverse.sensor_dataset_config import SensorDatasetConfig


def test_sensor_dataset_config():
    """Ensure that config fields are populated correctly from YAML."""
    dataset_name = "argoverse-v1.1"

    with hydra.initialize_config_module(config_module="argoverse.config"):
        cfg = hydra.compose(config_name=f"{dataset_name}.yaml")
        argoverse_config: SensorDatasetConfig = instantiate(cfg.SensorDatasetConfig)

    # check a few camera names
    assert argoverse_config.camera_sensors.has_camera("ring_rear_left")
    assert argoverse_config.camera_sensors.has_camera("stereo_front_left_rect")
    assert not argoverse_config.camera_sensors.has_camera("ring_rear_dummyname")

    # check sample camera dimensions
    assert argoverse_config.camera_sensors.ring_rear_left.img_width == 1920
    assert argoverse_config.camera_sensors.ring_rear_left.img_height == 1200

    # check other properties
    assert argoverse_config.dataset_name == "argoverse-v1.1"
    assert argoverse_config.ring_cam_fps == 30
    assert argoverse_config.stereo_cam_fps == 5
