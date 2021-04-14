from typing import List

import numpy as np

# We consider the disparity of a pixel to be correctly estimated if the absolute disparity error is less than a
# threshold and its relative error is less than 10% of its true value.
# We define three disparity error thresholds: 3, 5, and 10 pixels.
# Similar to the KITTI Stereo 2015 evaluation, we empirically found that the combination of absolute and relative
# disparity errors ensures an evaluation which is faithful with respect to the annotation errors in the ground truth.
# Absolute error thresholds in pixels, also used for KITTI stereo eval
DEFAULT_ABS_ERROR_THRESHOLDS: List[int] = [10, 5, 3]

# Relative error thresholds in pixels, also used for KITTI stereo eval
DEFAULT_REL_ERROR_THRESHOLDS: List[float] = [0.1, 0.1, 0.1]

# The disparity error image uses a custom log-color scale depicting correct estimates in blue and wrong estimates in
# red color tones, as in the KITTI Stereo 2015 Benchmark [1].
# LOG_COLORMAP = [disparity error range, RGB color], where the disparity error range is defined as:
# y = 2^x for x = [-inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, inf].
LOG_COLORMAP = [
    [np.array([2 ** -np.inf, 2 ** -4]), np.array([49, 54, 149], dtype=np.uint8)],
    [np.array([2 ** -4, 2 ** -3]), np.array([69, 117, 180], dtype=np.uint8)],
    [np.array([2 ** -3, 2 ** -2]), np.array([116, 173, 209], dtype=np.uint8)],
    [np.array([2 ** -2, 2 ** -1]), np.array([171, 217, 233], dtype=np.uint8)],
    [np.array([2 ** -1, 2 ** 0]), np.array([224, 243, 248], dtype=np.uint8)],
    [np.array([2 ** 0, 2 ** 1]), np.array([254, 224, 144], dtype=np.uint8)],
    [np.array([2 ** 1, 2 ** 2]), np.array([253, 174, 97], dtype=np.uint8)],
    [np.array([2 ** 2, 2 ** 3]), np.array([244, 109, 67], dtype=np.uint8)],
    [np.array([2 ** 3, 2 ** 4]), np.array([215, 48, 39], dtype=np.uint8)],
    [np.array([2 ** 4, 2 ** np.inf]), np.array([165, 0, 38], dtype=np.uint8)],
]
