from typing import List

import numpy as np

COMPETITION_CLASSES: List[str] = ["VEHICLE", "PEDESTRIAN", "BUS"]

TP_ERROR_NAMES: List[str] = ["ATE", "ASE", "AOE"]
N_TP_ERRORS: int = len(TP_ERROR_NAMES)

STATISTIC_NAMES: List[str] = ["AP"] + TP_ERROR_NAMES + ["CDS"]

MAX_SCALE_ERROR: float = 1.0
MAX_YAW_ERROR: float = np.pi

# Higher is better.
MIN_AP: float = 0.0
MIN_CDS: float = 0.0

# Lower is better.
MAX_NORMALIZED_ATE: float = 1.0
MAX_NORMALIZED_ASE: float = 1.0
MAX_NORMALIZED_AOE: float = 1.0

# Max number of boxes considered per class per scene.
MAX_NUM_BOXES: int = 500

SIGNIFICANT_DIGITS: float = 3
