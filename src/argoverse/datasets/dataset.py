"""General dataloader for Argoverse 2.0 datasets."""
import logging
import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

from polars.eager import DataFrame
from polars.io import read_ipc
from polars.lazy import col
from polars.lazy.expr import Expr

from argoverse.distributed.utils import compute_chunksize, parallelize
from argoverse.io.loading import read_feather

logger = logging.Logger(__name__)


@dataclass
class Dataset:

    rootdir: Path
    index_names: Tuple[str, ...]
    metadata: DataFrame = field(init=False)
    paths: List[Path] = field(init=False)

    def __post_init__(self):
        """Post initialization."""
        self.crawl()
        # self.paths = self.get_paths(self.metadata)

    def crawl(self):
        """Crawl the metadata files.

        Load the root metadata file for fasting dataloading / indexing.
        This prevents overloading the filesystem with I/O operations.
        """
        metadata_file = osp.join(self.rootdir, "_metadata")
        self.metadata = read_ipc(metadata_file, use_pyarrow=False)

    def get_records(self, col_name: str) -> DataFrame:
        return self.metadata.filter(col("record_type") == col_name)

    def get_paths(self, expr: Expr = None) -> List[Path]:
        metadata = self.metadata.select(self.index_names)
        if expr is not None:
            metadata = metadata.filter(expr)

        keys = metadata.select(self.index_names)
        keys = keys.fold(lambda x, y: x + "/" + y).apply(
            lambda x: osp.join(self.rootdir, x, "part.feather")
        )
        return keys

    def load_data(self, metadata: DataFrame) -> List[DataFrame]:
        """Load the data from the provided metadata.

        Constructs the absolute paths of the dataset from from the
        root directory and the keys in the metadata file. Parallelizes
        dataloading using all cores on the machine with a chunksize
        of `metadata.shape[0] / num_cpus`.

        Args:
            metadata (DataFrame): (N,34) DataFrame.

        Returns:
            List[DataFrame]: (N,) Dataframes of variable size.
        """
        paths = self.get_paths(metadata)
        chunksize = compute_chunksize(len(paths))
        paths = [(path, self.index_names) for path in paths]
        data_list: List[DataFrame] = parallelize(
            read_feather, paths, chunksize, use_starmap=True
        )
        return data_list

    def __getitem__(self, idx: int) -> Any:
        return read_feather(Path(self.paths[idx]), self.index_names)
