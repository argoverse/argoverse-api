from typing import Final, List, Tuple

import numpy as np

COMPETITION_CLASSES: Final[Tuple[str, ...]] = ("VEHICLE", "PEDESTRIAN", "BUS")

TP_ERROR_NAMES: Final[List[str]] = ["ATE", "ASE", "AOE"]
N_TP_ERRORS: Final[int] = len(TP_ERROR_NAMES)

STATISTIC_NAMES: Final[List[str]] = ["AP"] + TP_ERROR_NAMES + ["CDS"]

MAX_SCALE_ERROR: Final[float] = 1.0
MAX_YAW_ERROR: Final[float] = np.pi

# Higher is better.
MIN_AP: Final[float] = 0.0
MIN_CDS: Final[float] = 0.0

# Lower is better.
MAX_NORMALIZED_ATE: Final[float] = 1.0
MAX_NORMALIZED_ASE: Final[float] = 1.0
MAX_NORMALIZED_AOE: Final[float] = 1.0

# Max number of boxes considered per class per scene.
MAX_NUM_BOXES: Final[int] = 500

SIGNIFICANT_DIGITS: Final[int] = 3
