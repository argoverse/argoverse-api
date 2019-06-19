# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Unit tests for datetime_utils."""

from argoverse.utils.datetime_utils import generate_datetime_string


def test_datetime_smokescreen() -> None:
    """Basic test to ensure generate_datetime_string() returns a string."""
    datetime_str = generate_datetime_string()
    assert isinstance(datetime_str, str)
