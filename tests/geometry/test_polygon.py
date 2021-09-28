import numpy as np

from argoverse.geometry.polygon import cuboid2poly


def test_cuboid2poly():
    cuboid = np.array([[0, 1, 1, 2, 2, 2, 0]])
    cuboid2poly(cuboid)
