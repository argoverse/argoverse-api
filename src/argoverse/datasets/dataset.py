"""General dataloader for Argoverse 2.0 datasets."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import pandas as pd
from argoverse.io.loading import read_feather

logger = logging.Logger(__name__)


@dataclass
class Dataset:

    root_dir: Path
    index_names: Tuple[str, ...]
    metadata: pd.DataFrame = field(init=False)
    splits: Tuple[str, ...] = ("train", "val", "test")

    def __post_init__(self) -> None:
        """Post initialization."""
        # self.crawl()
        # self.paths = self.get_paths()

    # def crawl(self) -> None:
    #     """Crawl the metadata files.

    #     Load the root metadata file for fasting dataloading / indexing.
    #     This prevents overloading the filesystem with I/O operations.
    #     """
    #     metadata_file = osp.join(self.rootdir, "_metadata")
    # self.metadata = read_ipc(metadata_file, use_pyarrow=False)

    # def get_records(self, col_name: str) -> pd.DataFrame:
    #     return self.metadata.filter(col("record_type") == col_name)

    # def get_paths(self, expr: Optional[Expr] = None) -> List[Path]:
    #     metadata = self.metadata.select(self.index_names)
    #     if expr is not None:
    #         metadata = metadata.filter(expr)

    #     keys = metadata.select(self.index_names)
    #     keys = keys.fold(lambda x, y: x + "/" + y).apply(
    #         lambda x: osp.join(self.rootdir, x, "part.feather")
    #     )
    #     return keys

    # def load_data(self, metadata: pd.DataFrame) -> List[pd.DataFrame]:
    #     """Load the data from the provided metadata.

    #     Constructs the absolute paths of the dataset from from the
    #     root directory and the keys in the metadata file. Parallelizes
    #     dataloading using all cores on the machine with a chunksize
    #     of `metadata.shape[0] / num_cpus`.

    #     Args:
    #         metadata (pd.DataFrame): (N,34) pd.DataFrame.

    #     Returns:
    #         List[pd.DataFrame]: (N,) pd.DataFrames of variable size.
    #     """
    #     paths = self.get_paths(metadata)
    #     chunksize = compute_chunksize(len(paths))
    #     paths = [(path, self.index_names) for path in paths]
    #     data_list: List[pd.DataFrame] = parallelize(
    #         read_feather, paths, chunksize, use_starmap=True
    #     )
    #     return data_list

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return read_feather(Path(self.paths[idx]), self.index_names)
