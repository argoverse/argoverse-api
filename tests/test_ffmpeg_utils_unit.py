# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Unit tests for ffmpeg_utils.py."""

from argoverse.utils.ffmpeg_utils import write_nonsequential_idx_video, write_video


def test_ffmpeg_seq_frame_vid_smokescreen():
    """
        """
    image_prefix = "imgs_%d.jpg"
    output_prefix = "out"
    write_video(image_prefix, output_prefix)


def test_ffmpeg_nonseq_frame_vid_smokescreen():
    """
        """
    img_wildcard = "imgs_%*.jpg"
    output_fpath = "out.mp4"
    fps = 10
    write_nonsequential_idx_video(img_wildcard, output_fpath, fps)
