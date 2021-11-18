"""General dataloader for AV2 datasets."""
import logging
from dataclasses import dataclass

from argoverse.utils.typing import PathLike

logger = logging.Logger(__name__)


@dataclass
class Dataset:
    root_dir: PathLike

    def __post_init__(self) -> None:
        """Post initialization."""
        pass
