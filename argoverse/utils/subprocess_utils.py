# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import subprocess
from typing import Optional, Tuple


def run_command(cmd: str, return_output: bool = False) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    Block until system call completes

    Args:
        cmd: string, representing shell command

    Returns:
        Tuple of (stdout, stderr) output if return_output is True, else None
    """
    (stdout_data, stderr_data) = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()

    if return_output:
        return stdout_data, stderr_data
    return None, None
