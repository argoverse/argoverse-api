from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext
from setuptools import setup

ext_modules = [
    Extension(
        name="disparity_interpolation",
        sources=[
            "disparity_interpolation/source/disparity_interpolation.pyx",
            "disparity_interpolation/source/nn_interpolation.cpp",
        ],
        include_dirs=["/usr/local/include/opencv4/", numpy.get_include()],
        libraries=["opencv_core", "opencv_imgproc"],
        language="c++",
        extra_compile_args=["-w", "-O3"],
    )
]

setup(
    name="disparity_interpolation",
    version="1.0.0",
    author="Jhony Kaesemodel Pontes",
    description="Nearest neighbor interpolation for disparity images with missing parts.",
    packages=["disparity_interpolation"],
    package_data={"disparity_interpolation": ["disparity_interpolation/source/*"]},
    scripts=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
