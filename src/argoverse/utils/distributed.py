import multiprocessing as mp
from typing import Any, Callable, Final, List

from tqdm.contrib.concurrent import process_map

NCPUS: Final[int] = mp.cpu_count()


def compute_chunksize(njobs: int) -> int:
    return max(njobs // NCPUS, 1)


def parallelize(
    fn: Callable[..., Any],
    jobs: List[Any],
    chunksize: int = 1,
    with_progress_bar: bool = False,
    use_starmap: bool = False,
) -> List[Any]:
    if with_progress_bar:
        outputs: List[Any] = process_map(
            fn,
            jobs,
            max_workers=NCPUS,
            chunksize=chunksize,
        )
    elif use_starmap:
        with mp.Pool(NCPUS) as p:
            outputs: List[Any] = p.starmap(fn, jobs, chunksize=chunksize)
    else:
        with mp.Pool(NCPUS) as p:
            outputs: List[Any] = p.map(fn, jobs, chunksize=chunksize)
    return outputs
