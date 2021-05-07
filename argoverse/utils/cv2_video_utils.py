#!/usr/bin/python3

from typing import Optional

import cv2
import numpy as np

"""
Python-based utilities to avoid blowing up the disk with images, as FFMPEG requires.

Inspired by Detectron2 and MSeg:
    https://github.com/facebookresearch/detectron2/blob/bab413cdb822af6214f9b7f70a9b7a9505eb86c5/demo/demo.py
    https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/utils/cv2_video_utils.py
See OpenCV documentation for more details:
    https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videowriter-videowriter
"""


class VideoWriter:
    """
    Lazy init, so that the user doesn't have to know width/height a priori.
    Our default codec is "mp4v", though you may prefer "x264", if available
    on your system
    """

    def __init__(self, output_fpath: str, fps: int = 30) -> None:
        """Initialize VideoWriter options."""
        self.output_fpath = output_fpath
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.codec = "mp4v"

    def init_outf(self, height: int, width: int) -> None:
        """Initialize the output video file."""
        self.writer = cv2.VideoWriter(
            filename=self.output_fpath,
            # some installations of OpenCV may not support x264 (due to its license),
            # you can try another format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*self.codec),
            fps=float(self.fps),
            frameSize=(width, height),
            isColor=True,
        )

    def add_frame(self, rgb_frame: np.ndarray) -> None:
        """Append a frame of shape (h,w,3) to the end of the video file."""
        h, w, _ = rgb_frame.shape
        if self.writer is None:
            self.init_outf(height=h, width=w)
        bgr_frame = rgb_frame[:, :, ::-1]
        if self.writer is not None:
            self.writer.write(bgr_frame)

    def complete(self) -> None:
        """ """
        if self.writer is not None:
            self.writer.release()
