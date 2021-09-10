"""I/O for manipulating the Argoverse 2.0 dataset."""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def load_calibration(
    fpath: Path, columns: Optional[Tuple[str, ...]] = None
) -> pd.DataFrame:
    """Loads the calibration metadata for the Argoverse 2.0 sensor dataset.

    Schema/DataType:
        Sensor Name (name): [TODO]
        Focal Length (x): np.float64
        Focal Length (y): np.float64
        Focal Center (x): np.float64
        Focal Center (y): np.float64

        Skew (s): np.float64
        Sensor Width (width): np.uint16
        Sensor Height (height): np.uin16

        Translation (tx): np.float64
        Translation (ty): np.float64
        Translation (tz): np.float64

        Quaternion Coefficient (qw): np.float64
        Quaternion Coefficient (qx): np.float64
        Quaternion Coefficient (qy): np.float64
        Quaternion Coefficient (qz): np.float64

    Args:
        fpath (Path): [description]
        columns (Optional[Tuple[str, ...]]): [description]

    Returns:
        pd.DataFrame: (N,15) Dataframe of calibration metadata.
    """
    calibration = load_schema(fpath, columns=columns)
    return calibration


def load_labels(fpath: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """Loads the track labels for the Argoverse 2.0 sensor dataset.

    The Argoverse 2.0 track labels consist of 3D cuboids with 6-DOF pose.

    Schema/DataType:
        Translation (tx): np.float64
        Translation (ty): np.float64
        Translation (tz): np.float64

        Length (length): np.float64
        Width (width): np.float64
        Height (height): np.float64

        Quaternion Coefficient (qw): np.float64
        Quaternion Coefficient (qx): np.float64
        Quaternion Coefficient (qy): np.float64
        Quaternion Coefficient (qz): np.float64

        Label Class (label_class): [TODO]
        Track UUID (track_uuid): [TODO]
        Time of Validity (tov): np.int64

    Args:
        fpath (Path): Source file path.
        columns (Optional[Tuple[str, ...]]): DataFrame columns to load.

    Returns:
        pd.DataFrame: (N,13) Dataframe of .
    """
    labels = load_schema(fpath, columns=columns)
    return labels


def load_lidar(fpath: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """[summary]

    Schema/DataType:
        X-Coordinate (x): np.float16
        Y-Coordinate (y): np.float16
        Z-Coordinate (z): np.float16

        Intensity (i): np.uint8
        Sensor (s): np.uint8
        Time of Validity (tov): np.int64

    Args:
        fpath (Path): [description]
        columns (Optional[Tuple[str, ...]], optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: (N,6) DataFrame containing Cartesian coordinates, intensity,
            sensor name, and time of validity.
    """
    lidar = load_schema(fpath, columns=columns)
    return lidar


def load_poses(fpath: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """[summary]

    Schema/DataType:
        Translation (tx): np.float64
        Translation (ty): np.float64
        Translation (tz): np.float64

        Quaternion Coefficient (qw): np.float64
        Quaternion Coefficient (qx): np.float64
        Quaternion Coefficient (qy): np.float64
        Quaternion Coefficient (qz): np.float64
        Time of Validity (tov): np.int64

    Args:
        fpath (Path): [description]
        columns (Optional[Tuple[str, ...]], optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: (N,8) DataFrame containing SE(3) pose and time of validity.
    """
    poses = load_schema(fpath, columns=columns)
    return poses


def load_schema(fpath: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    table = pa.feather.read_feather(fpath, columns=columns)
    return table.as_pandas()


def load_dataset(fpath: Path) -> ds.Dataset:
    # TODO Is there a native way to remove the dot?
    format = fpath.suffix[1:]
    return ds.dataset(fpath, format=format)
