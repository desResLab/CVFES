# CVFES
A Cardiovascular solver using Finite Element method.

Running command:
1. python3 setup.py build_ext --inplace
2. qsub CPUCVFES_Fluid.script # Run fluid solver on CPUs
or qsub CPUCVFES_Solid.script # Run structure solver on CPUs
or qsub GPUCVFES_Solid.script # Run structure solver on GPUs

Libraries required:
1. parmetismodule # a self-written python interface of parmetis lib https://github.com/melindalx/parmetismodule (need gcc, mpich, mpi4py, parmetis&metis libs.)
2. VTK
3. python modules: configobj, mpi4py # use command "pip install --user xxx"