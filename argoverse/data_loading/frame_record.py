#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import Tuple

import numpy as np


class FrameRecord:
    """
    Store representation of a bounding box in some timeframe, in different coordinate systems.
    This bounding box comes from a track that shares the same color.
    """

    def __init__(
        self,
        bbox_city_fr: np.ndarray,
        bbox_ego_frame: np.ndarray,
        occlusion_val: int,
        color: Tuple[float, float, float],
        track_uuid: str,
        obj_class_str: str,
    ) -> None:
        """Initialize FrameRecord.
        Args:
          bbox_city_fr: bounding box for city frame.
          bbox_ego_frame: bounding box for ego frame.
          occlusion_val: occlusion value.
          color: tuple representing color. RGB values should be within [0,1] range.
          track_uuid: track uuid
          obj_class_str: object class string
        """
        self.bbox_city_fr = bbox_city_fr
        self.bbox_ego_frame = bbox_ego_frame
        self.occlusion_val = occlusion_val
        self.color = color
        self.track_uuid = track_uuid
        self.obj_class_str = obj_class_str
