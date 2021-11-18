from __future__ import annotations

import os
import typing
from typing import TYPE_CHECKING

PathLike = typing.Union[str, bytes, os.PathLike[str]]

if TYPE_CHECKING:
    reveal_locals()
