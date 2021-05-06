#!/usr/bin/python3

import cv2
import numpy as np

"""
Python-based utilities to avoid blowing up the disk with images, as FFMPEG requires.
Inspired by Detectron2 and MSeg:
https://github.com/facebookresearch/detectron2/blob/bab413cdb822af6214f9b7f70a9b7a9505eb86c5/demo/demo.py
https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/utils/cv2_video_utils.py
See OpenCV documentation for more details:
https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
"""


class VideoWriter:
    """
    Lazy init, so that the user doesn't have to know width/height a priori.
    Our default codec is "mp4v", though you may prefer "x264", if available
    on your system
    """

    def __init__(self, output_fpath: str, fps: int = 30) -> None:
        """ """
        self.output_fpath = output_fpath
        self.fps = fps
        self.writer: cv2.VideoWriter = None
        self.codec: str = "mp4v"

    def init_outf(self, height: int, width: int) -> None:
        """ """
        self.writer = cv2.VideoWriter(
            filename=self.output_fpath,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*self.codec),
            fps=float(self.fps),
            frameSize=(width, height),
            isColor=True,
        )

    def add_frame(self, rgb_frame: np.ndarray) -> None:
        """"""
        h, w, _ = rgb_frame.shape
        if self.writer is None:
            self.init_outf(height=h, width=w)
        bgr_frame = rgb_frame[:, :, ::-1]
        self.writer.write(bgr_frame)

    def complete(self) -> None:
        """ """
        self.writer.release()
