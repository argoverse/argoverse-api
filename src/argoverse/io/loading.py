"""I/O for manipulating the Argoverse 2.0 dataset."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_calibration(
    fpath: Path, columns: Optional[Tuple[str, ...]] = None
) -> pd.DataFrame:
    """Loads the calibration metadata for the Argoverse 2.0 sensor dataset.

    Schema/DataType:
        Sensor name
        Focal length (x): np.float64
        Focal length (y): np.float64
        Focal Center (x): np.float64
        Focal Center (y): np.float64
        Skew: np.float64
        Sensor width: np.uint16
        Sensor height: np.uin16
        Translation (x): np.float64
        Trans. (y): np.float64
        Trans. (z): np.float64
        Quaternion coeff. (w): np.float64
        Quaternion coeff. (x): np.float64
        Quaternion coeff. (y): np.float64
        Quaternion coeff. (z): np.float64

    Args:
        fpath (Path): [description]
        columns (Optional[Tuple[str, ...]]): [description]

    Returns:
        pd.DataFrame: (N,15) Dataframe of calibration metadata.
    """
    calibration = pd.read_feather(fpath)
    return calibration


def load_labels(fpath: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """Loads the track labels for the Argoverse 2.0 sensor dataset.

    The Argoverse 2.0 track labels consist of 3D cuboids with 6-DOF pose.

    Schema/DataType:
        Translation (x): np.float64
        Translation (y): np.float64
        Translation (z): np.float64
        Quaternion coefficient (w): np.float64
        Quaternion coefficient (x): np.float64
        Quaternion coefficient (y): np.float64
        Quaternion coefficient (z): np.float64
        Time of Validity: np.int64

    Args:
        fpath (Path): Source file path.
        columns (Optional[Tuple[str, ...]]): DataFrame columns to load.

    Returns:
        pd.DataFrame: (N,13) Dataframe of .
    """
    labels = pd.read_feather(fpath)
    return labels


def load_lidar(fpath: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """[summary]

    Schema/DataType:
        x: np.float16
        y: np.float16
        z: np.float16
        i: np.uint8
        s: np.uint8
        tov: np.int64

    Args:
        fpath (Path): [description]
        columns (Optional[Tuple[str, ...]], optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """
    lidar = pd.read_feather(fpath)
    return lidar


def load_poses(fpath: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    poses = pd.read_feather(fpath)
    return poses
