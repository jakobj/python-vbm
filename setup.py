from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("vbm",
                  ["vbm.pyx"],
                  libraries=["blas"],
                  depends=["vbm.h"],
                  extra_compile_args=["-funroll-loops", "-ffast-math", "-O3"]),
    ])
