# CVFES
An Ensemble Finite Element Cardiovascular Solver

To build with cython use
~~~
python3 setup.py build_ext --inplace
~~~

## Scripts to run the solver through the Notre Dame CRC
To run the fluid solver on multiple CPUs use
~~~
2. qsub CPUCVFES_Fluid.script
~~~
To run the structural solver on multiple CPUs use
~~~
qsub CPUCVFES_Solid.script
~~~
To run the structural solver on multiple GPUs use
~~~
qsub GPUCVFES_Solid.script
~~~

## Required modules
To run CVFES the following modules are required:
- parmetismodule. An in-house Python wrapper for Parmetis can be found at https://github.com/melindalx/parmetismodule (need gcc, mpich, mpi4py, parmetis&metis libs).
- VTK libraries. 
- Python modules: configobj, mpi4py (use "pip install --user xxx" to install these modules).
