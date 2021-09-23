import logging
import os.path as osp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from polars.eager import DataFrame
from polars.io import read_ipc

logger = logging.Logger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class Dataset:

    rootdir: Path
    index_names: Tuple[str, ...] = field(default_factory=tuple)
    metadata: DataFrame = field(init=False)

    def __post_init__(self):
        self.crawl()

    def crawl(self):
        metadata_file = osp.join(self.rootdir, "_metadata")
        self.metadata = read_ipc(metadata_file, use_pyarrow=False)
