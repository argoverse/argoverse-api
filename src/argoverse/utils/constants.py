import multiprocessing as mp
from typing import Final

import numpy as np

PI: Final[float] = np.pi
NAN: Final[float] = float("nan")
MAX_CPUS: Final[int] = mp.cpu_count()
