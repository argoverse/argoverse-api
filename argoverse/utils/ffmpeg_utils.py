# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import os
from pathlib import Path

from argoverse.utils.subprocess_utils import run_command

"""
Create a high-quality video using the encoder x264.
For x264, the valid Constant Rate Factor (crf) range is 0-51.

The range of the quantizer scale is 0-51: where 0 is lossless,
23 is default, and 51 is worst possible. A lower value is a
higher quality and a subjectively sane range is 18-28. Consider 18 to be
visually lossless or nearly so: it should look the same or nearly the
same as the input but it isn't technically lossless.
"""


def write_video(image_prefix: str, output_prefix: str, fps: int = 10) -> None:
    """
    Use FFMPEG to write a video to disk, from a sequence of images.

    Args:
        image_prefix: string, with %d embedded inside and ending with
            a prefix, e.g. .png/.jpg. Absolute path
        output_prefix: absolute path for output video, without .mp4 prefix
        fps: integer, frames per second
    """
    codec_params_string = get_ffmpeg_codec_params_string()
    cmd = f"ffmpeg -r {fps} -i {image_prefix} {codec_params_string} {output_prefix}_{fps}fps.mp4"
    print(cmd)
    run_command(cmd)


def write_nonsequential_idx_video(img_wildcard: str, output_fpath: str, fps: int) -> None:
    """
    Args:
        img_wildcard: string
        output_fpath: string
        fps: integer, frames per second
    """
    codec_params_string = get_ffmpeg_codec_params_string()
    cmd = f"ffmpeg -r {fps} -f image2 -i {img_wildcard} {codec_params_string} {output_fpath}"
    print(cmd)
    run_command(cmd)


def ffmpeg_compress_video(uncompressed_mp4_path: str, fps: int) -> None:
    """Generate compressed version of video, and delete uncompressed version.
    Args:
        img_wildcard: path to video to compress
    """
    codec_params_string = get_ffmpeg_codec_params_string()
    fname_stem = Path(uncompressed_mp4_path).stem
    compressed_mp4_path = f"{Path(uncompressed_mp4_path).parent}/{fname_stem}_compressed.mp4"
    cmd = f"ffmpeg -r {fps} -i {uncompressed_mp4_path} {codec_params_string} {compressed_mp4_path}"
    print(cmd)
    run_command(cmd)
    os.remove(uncompressed_mp4_path)


def get_ffmpeg_codec_params_string() -> str:
    """Generate command line params for FFMPEG for a widely compatible codec with good compression"""
    codec_params = [
        "-vcodec libx264",
        "-profile:v main",
        "-level 3.1",
        "-preset medium",
        "-crf 23",
        "-x264-params ref=4",
        "-acodec copy",
        "-movflags +faststart",
        "-pix_fmt yuv420p",
        "-vf scale=920:-2",
    ]
    return " ".join(codec_params)
