import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
from polars.eager import DataFrame


@dataclass
class Dataset:

    rootdir: Path
    index_names: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Optional[DataFrame] = None

    def __post_init__(self):
        self.crawl()

    def crawl(self):
        metadata_file = osp.join(self.rootdir, "_metadata")
        self.metadata = pl.io.read_ipc(metadata_file, use_pyarrow=False)

