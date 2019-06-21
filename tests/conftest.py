# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Configure pytest."""

import logging

import pytest


@pytest.fixture(autouse=True)
def set_log_level(caplog):
    """Set the log level.

    Set the log level to DEBUG for our testing to make sure that any bad log
    statements throw errors.
    """
    caplog.set_level(logging.DEBUG)
