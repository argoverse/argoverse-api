import logging
import os.path as osp
import sys
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import List, Tuple

import polars as pl
from polars.eager import DataFrame
from polars.io import read_ipc
from polars.lazy import col

from argoverse.distributed.utils import compute_chunksize, parallelize
from argoverse.evaluation.detection.eval import DetectionCfg, evaluate

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

    def evaluate(self, dts: List, detection_classes: List, splits):
        logger.info(f"\nEvaluating on the following splits: {splits}.")

        # Construct detection config.
        cfg = DetectionCfg(
            dt_classes=detection_classes,
            eval_only_roi_instances=False,
            save_figs=True,
            splits=splits,
        )

        metadata = self.metadata
        split_predicate = reduce(
            lambda x, y: x | y, [col("split") == split for split in splits]
        )
        metadata = metadata.filter(split_predicate)

        labels_predicate = col("record_type") == "labels"
        labels_metadata = metadata.filter(labels_predicate)

        jobs = [(self.rootdir, key) for key in labels_metadata.to_numpy()]
        n = compute_chunksize(len(jobs))
        labels = pl.concat(parallelize(format_label, jobs, n, with_progress_bar=True))

        metrics = evaluate(dts, labels, None, cfg)
        metrics = pl.from_pandas(metrics.reset_index())
        return metrics


def format_label(job: Tuple[str, Tuple[str, ...]]) -> DataFrame:
    srcdir, keys = job
    path = osp.join(srcdir, *keys[:4], "part.feather")
    lab = read_ipc(path, use_pyarrow=False)
    lab["log_id"] = [keys[1]] * lab.shape[0]
    lab["tov_ns"] = [keys[3]] * lab.shape[0]
    return lab
