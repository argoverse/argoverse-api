# distutils: language = c++
# distutils: sources = nn_interpolation.cpp
# cython: language_level=3

import cython

import numpy as np
cimport numpy as np


cdef extern from "opencv2/opencv.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        Mat(float, float, float) except +
        void create(float, float, float)
        void* data

cdef extern from "opencv2/opencv.hpp":
    cdef float CV_32F

cdef extern from "nn_interpolation.h":
    cdef Mat nn_interpolation(Mat disparity)

def disparity_interpolator(np.ndarray[np.float32_t, ndim=2, mode="c"] disparity):

    cdef int m = disparity.shape[0], n = disparity.shape[1]

    # copy data on
    cdef Mat x
    x.create(m, n, CV_32F)
    (<np.float32_t[:m, :n]> x.data)[:, :] = disparity

    # run interpolation
    cdef Mat z = nn_interpolation(x)

    # copy data off
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] c = np.zeros_like(disparity)
    c[:, :] = <np.float32_t[:m, :n]> z.data

    return c
