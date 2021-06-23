# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
from typing import List, Optional

import numpy as np


class LaneSegment:
    def __init__(
        self,
        id: int,
        has_traffic_control: bool,
        turn_direction: str,
        is_intersection: bool,
        l_neighbor_id: Optional[int],
        r_neighbor_id: Optional[int],
        predecessors: List[int],
        successors: Optional[List[int]],
        centerline: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            has_traffic_control:
            turn_direction: 'RIGHT', 'LEFT', or 'NONE'
            is_intersection: Whether or not this lane segment is an intersection
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            predecessors: The IDs of the lane segments that come after this one
            successors: The IDs of the lane segments that come before this one.
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.has_traffic_control = has_traffic_control
        self.turn_direction = turn_direction
        self.is_intersection = is_intersection
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors
        self.centerline = centerline
