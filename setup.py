from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "gemm_pyx",
        sources=["gemm_pyx.pyx"],
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
