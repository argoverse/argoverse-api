from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from argoverse.io.loading import read_feather
from scipy.spatial.transform import Rotation as R


@dataclass
class PinholeModel:

    # Intrinsics
    fx: np.ndarray
    cx: np.ndarray
    fy: np.ndarray
    cy: np.ndarray

    # Extrinsics
    # Quaternion
    qw: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray

    # Translation
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def K(self) -> np.ndarray:
        K = np.zeros((self.ncams, 3, 4))
        K[:, 0, 0] = self.fx
        K[:, 0, 2] = self.cx
        K[:, 1, 1] = self.fy
        K[:, 1, 2] = self.cy
        K[:, 2, 2] = 1.0
        return K

    @property
    def R(self) -> np.ndarray:
        quat = np.stack((self.qx, self.qy, self.qz, self.qw), axis=1)
        return R.from_quat(quat).as_matrix()

    @property
    def t(self) -> np.ndarray:
        return np.stack((self.x, self.y, self.z), axis=1)

    @property
    def ncams(self) -> int:
        return self.fx.shape[0]

    @classmethod
    def load(cls, path: Path) -> PinholeModel:
        calibration = read_feather(path)
        return cls(
            fx=calibration["fx"].to_numpy(),
            cx=calibration["cx"].to_numpy(),
            fy=calibration["fy"].to_numpy(),
            cy=calibration["cy"].to_numpy(),
            qw=calibration["qw"].to_numpy(),
            qx=calibration["qx"].to_numpy(),
            qy=calibration["qy"].to_numpy(),
            qz=calibration["qz"].to_numpy(),
            x=calibration["tx"].to_numpy(),
            y=calibration["ty"].to_numpy(),
            z=calibration["tz"].to_numpy(),
        )

    def __repr__(self) -> str:
        repr = pd.DataFrame({k: v for k, v in self.__dict__.items()})
        return str(repr)
