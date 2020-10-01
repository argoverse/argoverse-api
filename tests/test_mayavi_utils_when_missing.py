# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Smokescreen unit tests to make sure our Mayavi utility functions work."""

try:
    import mayavi.mlab

    MISSING_MAYAVI = False

except ImportError:
    MISSING_MAYAVI = True

import pathlib

import numpy as np
import pytest

from argoverse.utils import mayavi_wrapper

_TEST_DIR = pathlib.Path(__file__).parent.parent / "tests"

skip_if_mayavi = pytest.mark.skipif(not MISSING_MAYAVI, reason="mayavi installed")


@skip_if_mayavi  # type: ignore
def test_raises_when_we_try_to_use_a_missing_mayavi() -> None:
    n_mer, n_long = 6, 11
    pi = np.pi
    dphi = pi / 1000.0
    phi = np.arange(0.0, 2 * pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    z = np.sin(n_long * mu / n_mer) * 0.5

    with pytest.raises(mayavi_wrapper.NoMayaviAvailableError):
        mayavi_wrapper.mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap="Spectral")
