"""Example script for loading data from the AV2 sensor dataset."""

from pathlib import Path
from typing import Dict, Final

import cv2
import numpy as np
import torch
from torchvision.io import write_video
from tqdm import tqdm

from argoverse.datasets.sensor import SensorDataset
from argoverse.rendering.rasterize import AV2_CATEGORY_CMAP, overlay_annotations, pc2im
from argoverse.rendering.video import FFMPEG_OPTIONS
from argoverse.utils.typing import PathLike

VOXEL_RESOLUTION: Final[np.ndarray] = np.array([1e-1, 1e-1, 2e-1])
GRID_SIZE: Final[np.ndarray] = np.array([100.0, 100.0, 5.0])


def main(dataset_dir: PathLike) -> None:
    dataset = SensorDataset(dataset_dir, with_annotations=True, with_imagery=False)

    prev_log = None
    ims = []
    for _, datum in enumerate(tqdm(dataset)):
        annotations = datum["annotations"]
        lidar = datum["lidar"]
        metadata = datum["metadata"]
        log_id = metadata["log_id"]

        offset_ns = lidar["offset_ns"]
        cmap = np.full((offset_ns.shape[0], 3), 128.0)
        bev = pc2im(
            lidar[["x", "y", "z", "intensity"]].to_numpy().astype(float),
            cmap=cmap,
            voxel_resolution=VOXEL_RESOLUTION,
            grid_size=GRID_SIZE,
        )

        bev = overlay_annotations(
            bev,
            annotations,
            voxel_resolution=VOXEL_RESOLUTION,
            category_cmap=AV2_CATEGORY_CMAP,
        )
        cv2.imwrite("bev.jpg", bev)

        curr_log = datum["metadata"]["log_id"]
        if prev_log is not None and curr_log != prev_log:
            break

        ims.append(torch.as_tensor(bev))
        prev_log = curr_log

    video = torch.stack(ims)
    write_video("bev.mp4", video, fps=10, options=FFMPEG_OPTIONS)


if __name__ == "__main__":
    dataset_dir = Path("/data") / "datasets" / "av2" / "sensor" / "v0002"
    main(dataset_dir)
