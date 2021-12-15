"""Example script for loading data from the AV2 sensor dataset."""

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from tqdm import tqdm

from argoverse.datasets.sensor import SensorDataset
from argoverse.rendering.rasterize import AV2_CATEGORY_CMAP, overlay_annotations, pc2im
from argoverse.rendering.video import FFMPEG_OPTIONS, write_video
from argoverse.utils.typing import PathLike

VOXEL_RESOLUTION: Final[np.ndarray] = np.array([1e-1, 1e-1, 2e-1])
GRID_SIZE: Final[np.ndarray] = np.array([100.0, 100.0, 5.0])


def main(dataset_dir: PathLike, with_annotations: bool = False, with_imagery: bool = False) -> None:
    dataset = SensorDataset(dataset_dir, with_annotations=with_imagery, with_imagery=with_imagery)

    prev_log = None
    video_list = []
    for i, datum in enumerate(tqdm(dataset)):
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

        if with_annotations:
            annotations = datum["annotations"]
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

        video_list.append(bev)
        prev_log = curr_log

        # if i > 20:
        #     break
    video = np.stack(video_list).astype(np.uint8)
    write_video(video, Path(f"../videos/{log_id}.mp4"))
    breakpoint()


if __name__ == "__main__":
    # dataset_dir = Path("/data") / "datasets" / "av2" / "sensor" / "v0002"
    dataset_dir = Path("/data") / "datasets" / "v0002"
    main(dataset_dir)
