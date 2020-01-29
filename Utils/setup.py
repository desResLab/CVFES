from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "optimizedSolidStressCalculate",
        ["optimizedSolidStressCalculate.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules = cythonize(ext_modules), # generalizedAlphaSolver.pyx optimizedAssemble.pyx
    # ["optimizedAssemble.pyx", "assemble.pyx"]
)
