from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

# Helper to locate pybind11
try:
    import pybind11
    from pybind11 import get_include
except ImportError:
    raise RuntimeError("pybind11 is required. Install with `pip install pybind11`.")

class get_pybind_include(object):
    def __str__(self):
        return get_include()

ext_modules = [
    Extension(
        'octomap_py',  # import name
        ['src/bt_buffer_to_voxel.cpp', 'src/bt_buffer_to_local_voxel.cpp'],  # source file
        include_dirs=[
            get_pybind_include(),
            '/usr/include',        # system include path
            '/usr/include/eigen3', # Eigen used by OctoMap
        ],
        libraries=['octomap'],     # Link to -loctomap
        language='c++',
        extra_compile_args=['-O3', '-std=c++14'],
    ),
]

setup(
    name='octomap_py',
    version='0.1.0',
    author='Your Name',
    author_email='your@email.com',
    description='Local voxel tensor extractor from Octomap binary using pybind11',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
