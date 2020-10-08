# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Configure pytest."""

import logging

import pytest
from _pytest.logging import LogCaptureFixture


@pytest.fixture(autouse=True)  # type: ignore
def set_log_level(caplog: LogCaptureFixture) -> None:
    """Set the log level.

    Set the log level to DEBUG for our testing to make sure that any bad log
    statements throw errors.
    """
    caplog.set_level(logging.DEBUG)
