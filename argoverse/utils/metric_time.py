from enum import Enum, auto

import numpy as np


class TimeUnit(Enum):
    Second = auto()
    Millisecond = auto()
    Microsecond = auto()
    Nanosecond = auto()


def to_metric_time(ts: int, src: TimeUnit, dst: TimeUnit) -> float:
    """ """
    units_per_sec = {TimeUnit.Second: 1, TimeUnit.Millisecond: 1e3, TimeUnit.Microsecond: 1e6, TimeUnit.Nanosecond: 1e9}
    # ts is in units of `src`, which will cancel with the denominator
    return ts * (units_per_sec[dst] / units_per_sec[src])
