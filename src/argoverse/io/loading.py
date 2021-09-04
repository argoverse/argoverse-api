"""I/O for manipulating the Argoverse 2.0 dataset."""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def load_calibration(fpath: Path, columns: Optional[Tuple[str, ...]]) -> pd.DataFrame:
    calibration = pd.read_feather(fpath)
    return calibration


def load_labels(fpath: Path, columns: Optional[Tuple[str, ...]]) -> pd.DataFrame:
    labels = pd.read_feather(fpath)
    return labels


def load_lidar(fpath: Path, columns: Optional[Tuple[str, ...]]) -> pd.DataFrame:
    lidar = pd.read_feather(fpath)
    return lidar


def load_poses(fpath: Path, columns: Optional[Tuple[str, ...]]) -> pd.DataFrame:
    poses = pd.read_feather(fpath)
    return poses
