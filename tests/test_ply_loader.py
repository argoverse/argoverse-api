# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import pathlib

import numpy as np

from argoverse.utils.ply_loader import load_ply

_TEST_DIR = pathlib.Path(__file__).parent


def test_load_ply():
    ply_fpath = _TEST_DIR / "test_data/tracking/1/lidar/PC_0.ply"
    pc = load_ply(ply_fpath)
    pc_gt = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
            [4.0, 0.0, 5.0],
            [5.0, 0.0, 5.0],
            [6.0, 0.0, 5.0],
            [7.0, 0.0, 5.0],
            [8.0, 0.0, 5.0],
            [9.0, 0.0, 5.0],
        ]
    )

    assert (pc == pc_gt).all()
