#!/usr/bin/env python

"""A simple python script to generate sequence videos."""

import argparse
import os
import shutil
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", help="sequence location ")
parser.add_argument(
    "--max_videos",
    type=int,
    help="maximum number of sequence to process. -1 to get all sequence",
    default=10,
)
parser.add_argument("--output_dir", help="output directory", default="vis_video")


def main(arguments: List[str]) -> int:

    args = parser.parse_args(arguments)

    sequence_list = os.listdir(args.root)

    os.makedirs(args.output_dir, exist_ok=True)

    max_videos = args.max_videos
    for seq_name in sequence_list[:max_videos]:
        seq_path = os.path.join(args.root, seq_name)
        df = pd.read_csv(seq_path)
        count = 0
        time_list = np.sort(np.unique(df["TIMESTAMP"].values))

        # Get API for Argo Dataset map
        avm = ArgoverseMap()
        city_name = df["CITY_NAME"].values[0]
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

        x_min = min(df["X"])
        x_max = max(df["X"])
        y_min = min(df["Y"])
        y_max = max(df["Y"])

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline

            if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)

        seq_out_dir = os.path.join(args.output_dir, seq_name.split(".")[0])

        for time in time_list:

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            df_cur = df.loc[df["TIMESTAMP"] <= time]

            viz_sequence(
                df_cur,
                lane_centerlines=lane_centerlines,
                show=False,
                smoothen=False,
            )

            os.makedirs(seq_out_dir, exist_ok=True)

            plt.savefig(
                os.path.join(seq_out_dir, f"{count}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
            count += 1

        from moviepy.editor import ImageSequenceClip

        img_idx = sorted([int(x.split(".")[0]) for x in os.listdir(seq_out_dir)])
        list_video = [f"{seq_out_dir}/{x}.png" for x in img_idx]
        clip = ImageSequenceClip(list_video, fps=10)
        video_path = os.path.join(args.output_dir, f"{seq_name.split('.')[0]}.mp4")
        clip.write_videofile(video_path)
        shutil.rmtree(seq_out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
