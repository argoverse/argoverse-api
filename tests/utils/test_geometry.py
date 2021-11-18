"""Unit tests for the geometry utilities."""

import numpy as np

from argoverse.utils.geometry import cart2hom, hom2cart


def test_cart2hom() -> None:
    """Test converting Cartesian to Homogeneous coordinates."""
    np.random.seed(0)

    expected_hom = np.array(
        [
            [0.5488135, 0.71518937, 0.60276338, 1.0],
            [0.54488318, 0.4236548, 0.64589411, 1.0],
            [0.43758721, 0.891773, 0.96366276, 1.0],
            [0.38344152, 0.79172504, 0.52889492, 1.0],
            [0.56804456, 0.92559664, 0.07103606, 1.0],
            [0.0871293, 0.0202184, 0.83261985, 1.0],
            [0.77815675, 0.87001215, 0.97861834, 1.0],
            [0.79915856, 0.46147936, 0.78052918, 1.0],
            [0.11827443, 0.63992102, 0.14335329, 1.0],
            [0.94466892, 0.52184832, 0.41466194, 1.0],
        ]
    )

    M, N = 10, 3
    cart = np.random.rand(M, N)
    hom = cart2hom(cart)
    assert np.allclose(hom, expected_hom)


def test_hom2cart() -> None:
    """Test converting Homogeneous to Cartesian coordinates."""
    np.random.seed(0)

    expected_cart = np.array(
        [
            [1.00721314, 1.3125554, 1.10622496],
            [0.47507022, 0.72428086, 0.4906935],
            [1.82203066, 0.7249862, 1.49694204],
            [6.51955844, 10.6232535, 0.81529472],
            [0.02323921, 0.95702094, 0.89442056],
            [1.25378829, 1.02386764, 0.59123909],
            [0.12520199, 0.67740243, 0.15174977],
            [0.67401914, 0.53557724, 0.34169995],
            [0.73854293, 0.92033886, 0.03042215],
            [0.89773761, 0.90483372, 1.38415955],
        ]
    )

    M, N = 10, 4
    hom = np.random.rand(M, N)
    cart = hom2cart(hom)
    assert np.allclose(cart, expected_cart)


def test_wrap_angles() -> None:
    pass
