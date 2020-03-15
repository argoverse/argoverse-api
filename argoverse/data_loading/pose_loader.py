# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import logging
from pathlib import Path
from typing import Optional

from argoverse.utils.json_utils import read_json_file
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

import numpy as np

logger = logging.getLogger(__name__)


def get_city_SE3_egovehicle_at_sensor_t(sensor_timestamp: int, dataset_dir: str, log_id: str) -> Optional[SE3]:
    """Get translation from city to ego vechile coordinates at a given timestamp.

        Args:
            sensor_timestamp: in nanoseconds
            dataset_dir:
            log_id:

        Returns:
            SE3 for translating city coordinates to ego vehicle coordinates if found, else None.
    """
    pose_fpath = f"{dataset_dir}/{log_id}/poses/city_SE3_egovehicle_{sensor_timestamp}.json"
    if not Path(pose_fpath).exists():
        logger.error(f"missing pose {sensor_timestamp}")
        return None

    pose_city_to_ego = read_json_file(pose_fpath)
    rotation = np.array(pose_city_to_ego["rotation"])
    translation = np.array(pose_city_to_ego["translation"])
    city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(rotation), translation=translation)
    return city_to_egovehicle_se3
