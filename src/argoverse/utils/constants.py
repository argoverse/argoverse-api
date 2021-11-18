# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Common mathematical and computing system constants used in AV2."""

import math
import multiprocessing as mp
from pathlib import Path
from typing import Final

# 3.14159 ...
PI: Final[float] = math.pi

# Not a number.
NAN: Final[float] = math.nan

# System home directory.
HOME: Final[Path] = Path.home()

# Max number of logical cores available on the current machine.
MAX_CPUS: Final[int] = mp.cpu_count()
