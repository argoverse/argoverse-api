# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Smokescreen unit tests to make sure our bird's eye view plotting util function can execute successfully."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from argoverse.visualization.mpl_point_cloud_vis import draw_point_cloud_bev

_TEST_DIR = pathlib.Path(__file__).parent


def test_draw_point_cloud_bev_smokescreen():
    """Test :ref:`draw_point_cloud_bev`"""
    fig3d = plt.figure(figsize=(15, 8))
    ax_bev = fig3d.add_subplot(111)

    point_cloud = np.loadtxt(_TEST_DIR / "test_data/sample_argoverse_sweep.txt")
    draw_point_cloud_bev(ax_bev, point_cloud, color="w", x_lim_3d=None, y_lim_3d=None, z_lim_3d=None)
    plt.close("all")
