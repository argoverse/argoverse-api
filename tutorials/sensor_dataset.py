"""Example script for loading data from the AV2 sensor dataset."""

from math import remainder
from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.color import bgr_to_rgb
from torchvision.io import write_video

from argoverse.datasets.sensor import SensorDataset
from argoverse.datasets.sensor.dataset import DataloaderMode
from argoverse.utils.constants import HOME
from argoverse.utils.typing import PathLike


def main(dataset_dir: PathLike) -> None:
    dataset = SensorDataset(dataset_dir, DataloaderMode.DETECTION)

    prev_log = None

    ims = []
    for i, datum in enumerate(dataset):
        annotations = datum["annotations"]
        lidar = datum["lidar"]

        curr_log = datum["metadata"]["log_id"]
        if prev_log is not None and curr_log != prev_log:
            break

        tiled_im = tile_cameras(datum)
        ims.append(torch.as_tensor(tiled_im))
        prev_log = curr_log

        # if i > 10:
        #     break

    video = torch.stack(ims)
    write_video("tiled_video.mp4", video, fps=10, options={"crf": "27"})


def tile_cameras(datum) -> np.ndarray:
    h = 1550 + 1550 + 1550
    w = 2048 + 1550 + 2048
    tiled_im = np.zeros((h, w, 3), dtype=np.uint8)

    ring_rear_left = datum["ring_rear_left"]
    ring_side_left = datum["ring_side_left"]
    ring_front_center = datum["ring_front_center"]
    ring_front_left = datum["ring_front_left"]
    ring_front_right = datum["ring_front_right"]
    ring_side_right = datum["ring_side_right"]
    ring_rear_right = datum["ring_rear_right"]

    tiled_im[:1550, :2048] = ring_front_left
    tiled_im[:2048, 2048 : 2048 + 1550] = ring_front_center
    tiled_im[:1550, 2048 + 1550 :] = ring_front_right

    tiled_im[1550:3100, :2048] = ring_side_left
    tiled_im[1550:3100, 2048 + 1550 :] = ring_side_right

    start = (w - 4096) // 2
    tiled_im[3100:4650, start : start + 2048] = np.fliplr(ring_rear_left)
    tiled_im[3100:4650, start + 2048 : start + 4096] = np.fliplr(ring_rear_right)
    return cv2.cvtColor(tiled_im, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    dataset_dir = Path("/data") / "datasets" / "av2" / "sensor" / "v0002"
    main(dataset_dir)
