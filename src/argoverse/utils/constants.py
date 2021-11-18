import math
import multiprocessing as mp
from typing import Final

PI: Final[float] = math.pi
NAN: Final[float] = float("nan")
MAX_CPUS: Final[int] = mp.cpu_count()
