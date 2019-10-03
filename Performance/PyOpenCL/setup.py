from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    # Extension(
    #     "cy_spmv",
    #     ["cy_spmv.pyx"],
    #     extra_compile_args=['-fopenmp'],
    #     extra_link_args=['-fopenmp'],
    # ),
    Extension(
        "util",
        ["util.pyx"],
    )
]

setup(
    ext_modules = cythonize(ext_modules), # generalizedAlphaSolver.pyx optimizedAssemble.pyx
    # ["optimizedAssemble.pyx", "assemble.pyx"]
)

# directives = {'linetrace': False, 'language_level': 3}
