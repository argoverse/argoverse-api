# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np

from argoverse.utils import polyline_density


def test_polyline_length():
    line = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1]])

    length = polyline_density.get_polyline_length(line)
    assert abs(length - 5.0) < 1e-10
