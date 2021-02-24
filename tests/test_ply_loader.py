# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import os
import pathlib

import numpy as np

from argoverse.utils.ply_loader import load_ply, load_ply_by_attrib

_TEST_DIR = pathlib.Path(__file__).parent


def test_load_ply() -> None:
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


def test_load_ply_by_attrib():
    """ Ensure intensity (i.e. reflectance) and ring index can be loaded from toy sweep """
    ply_fpath = _TEST_DIR / "test_data/tracking/1/lidar/PC_0.ply"
    pc = load_ply_by_attrib(ply_fpath, attrib_spec="xyzil")
    assert pc.shape == (10, 5)
    # intensities should be bounded between [0,255]
    assert np.all(np.logical_and(pc[:, 3] >= 0, pc[:, 3] <= 255))


def test_load_ply_by_attrib_invalid_attributes():
    """Try to load a toy sweep, using invalid attribute specification """
    ply_fpath = _TEST_DIR / "test_data/tracking/1/lidar/PC_0.ply"
    # "a" is an invalid point attribute
    pc = load_ply_by_attrib(ply_fpath, "xya")
    assert pc is None


def test_load_ply_by_attrib_real_sweep():
    """ Load an actual LiDAR sweep, with valid attribute specification """
    ply_fpath = os.path.join(_TEST_DIR, "test_data", "d60558d2-d1aa-34ee-a902-e061e346e02a__PC_315971347819783000.ply")
    pc = load_ply_by_attrib(ply_fpath, attrib_str="xyzil")
    assert pc.shape == (91083, 5)
    # intensities should be bounded between [0,255]
    assert np.all(np.logical_and(pc[:, 3] >= 0, pc[:, 3] <= 255))

    # laser numbers should be bounded between [0,31]
    assert np.all(np.logical_and(pc[:, 4] >= 0, pc[:, 4] <= 31))

    # make sure it matches the "xyz"-only API
    assert np.allclose(load_ply_by_attrib(ply_fpath, attrib_spec="xyz"), load_ply(ply_fpath))
