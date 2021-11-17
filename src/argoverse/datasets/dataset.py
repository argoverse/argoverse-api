"""General dataloader for AV2 datasets."""
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.Logger(__name__)


@dataclass
class Dataset:
    root_dir: Path

    def __post_init__(self) -> None:
        """Post initialization."""
        pass
