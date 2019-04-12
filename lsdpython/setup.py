#!/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("lsd", ["lsd.pyx", 'lsd_1.6/lsd.c'])]
)
