from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "optimizedAssemble",
        ["optimizedAssemble.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "optimizedFluidAssemble",
        ["optimizedFluidAssemble.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "optimizedSolidAssemble",
        ["optimizedSolidAssemble.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules = cythonize(ext_modules), # generalizedAlphaSolver.pyx optimizedAssemble.pyx
    # ["optimizedAssemble.pyx", "assemble.pyx"]
)

# directives = {'linetrace': False, 'language_level': 3}
