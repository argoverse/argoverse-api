# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for argoverse/utils/subprocess_utils.py"""

from argoverse.utils.subprocess_utils import run_command


def test_run_command_smokescreen() -> None:
    """
    Do not check output, just verify import works
    and does not crash.
    """
    cmd = "echo 5678"
    run_command(cmd)


def test_run_command_cat() -> None:
    """
    Execute a command to dump a string to standard output. Returned
    output will be in byte format with a carriage return.
    """
    cmd = "echo 5678"
    stdout_data, stderr_data = run_command(cmd, return_output=True)
    assert stderr_data is None
    assert stdout_data == "5678\n".encode()
