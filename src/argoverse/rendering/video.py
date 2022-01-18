"""Rendering tools for video visualizations."""
from pathlib import Path
from typing import Dict, Final, Union

import av
import cv2
import numpy as np
import pandas as pd

FFMPEG_OPTIONS: Final[Dict[str, str]] = {"crf": "27"}


def tile_cameras(named_sensors: Dict[str, Union[np.ndarray, pd.DataFrame]]) -> np.ndarray:
    """Combine ring cameras into a tiled image.

    Layout:

    ##########################################################
    # ring_front_left # ring_front_center # ring_front_right #
    ##########################################################
    # ring_side_left  #                   #  ring_side_right #
    ##########################################################
    ############ ring_rear_left # ring_rear_right ############
    ##########################################################

    Args:
        named_sensors (Dict[str, Union[np.ndarray, pd.DataFrame]]): Dictionary of camera names
            to the (width, height, 3) images.

    Returns:
        np.ndarray: Tiled image.
    """
    landscape_width = 2048
    landscape_height = 1550

    height = landscape_height + landscape_height + landscape_height
    width = landscape_width + landscape_height + landscape_width
    tiled_im = np.zeros((height, width, 3), dtype=np.uint8)

    ring_rear_left = named_sensors["ring_rear_left"]
    ring_side_left = named_sensors["ring_side_left"]
    ring_front_center = named_sensors["ring_front_center"]
    ring_front_left = named_sensors["ring_front_left"]
    ring_front_right = named_sensors["ring_front_right"]
    ring_side_right = named_sensors["ring_side_right"]
    ring_rear_right = named_sensors["ring_rear_right"]

    tiled_im[:landscape_height, :landscape_width] = ring_front_left
    tiled_im[:landscape_width, landscape_width : landscape_width + landscape_height] = ring_front_center
    tiled_im[:landscape_height, landscape_width + landscape_height :] = ring_front_right

    tiled_im[landscape_height:3100, :landscape_width] = ring_side_left
    tiled_im[landscape_height:3100, landscape_width + landscape_height :] = ring_side_right

    start = (width - 4096) // 2
    tiled_im[3100:4650, start : start + landscape_width] = np.fliplr(ring_rear_left)
    tiled_im[3100:4650, start + landscape_width : start + 4096] = np.fliplr(ring_rear_right)
    return cv2.cvtColor(tiled_im, cv2.COLOR_BGR2RGB)


def write_video(
    video: np.ndarray,
    dst: Path,
    codec: str = "libx264",
    fps: int = 10,
    crf: int = 27,
    preset: str = "veryfast",
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(dst), "w") as output:
        stream = output.add_stream(codec, fps)
        stream.width = video.shape[1]
        stream.height = video.shape[2]
        stream.options = {
            "crf": str(crf),
            "hwaccel": "auto",
            "movflags": "+faststart",
            "preset": preset,
            "profile:v": "main",
            "tag": "hvc1",
        }
        for _, im in enumerate(video):
            frame = av.VideoFrame.from_ndarray(im)
            output.mux(stream.encode(frame))
        output.mux(stream.encode(None))