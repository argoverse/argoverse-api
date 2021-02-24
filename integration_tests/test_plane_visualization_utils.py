import os
import pathlib
import sys

import matplotlib.pyplot as plt
import pytest

from argoverse.utils.calibration import load_calib

try:
    import mayavi.mlab

    from argoverse.utils.frustum_clipping import generate_frustum_planes
    from argoverse.utils.plane_visualization_utils import (
        get_perpendicular,
        plot_frustum_planes_and_normals,
        populate_frustum_voxels,
    )
except ImportError:
    MAYAVI_MISSING = True
else:
    MAYAVI_MISSING = False


_TEST_DIR = pathlib.Path(__file__).parent.parent / "tests"


if not MAYAVI_MISSING:
    calib = load_calib(os.fspath(_TEST_DIR / "test_data/tracking/1/vehicle_calibration_info.json"))["ring_front_center"]
    planes = generate_frustum_planes(calib.K, calib.camera)
    assert planes is not None


skip_if_mayavi_missing = pytest.mark.skipif(
    MAYAVI_MISSING,
    reason="Could not test functionality that depends on mayavi because mayavi is missing.",
)


@skip_if_mayavi_missing  # type: ignore
def test_plot_frustum_planes_and_normals() -> None:
    """"""

    assert planes is not None
    plot_frustum_planes_and_normals(planes, cuboid_verts=None, near_clip_dist=0.5)


@skip_if_mayavi_missing  # type: ignore
def test_populate_frustum_voxels() -> None:
    """"""
    fig, axis_pair = plt.subplots(1, 1, figsize=(20, 15))

    assert planes is not None
    populate_frustum_voxels(planes, fig, axis_pair)


@skip_if_mayavi_missing  # type: ignore
def test_get_perpendicular() -> None:
    """"""
    n = [0, 1, 1]
    result = get_perpendicular(n)
