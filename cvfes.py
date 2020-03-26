#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    CVFES class is the main solver of the CVFES project.
"""

from configobj import ConfigObj
from cvconfig import CVConfig
# from cvcomm import CVCOMM
from mesh import *
from solver import *
from meshPartition import MeshPartition
from meshColoring import MeshColoring

from mpi4py import MPI


__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"



class CVFES:

    def __init__(self, comm):

        self.comm = comm
        # For using convenient.
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        print('size {} rank {}\n'.format(self.size, self.rank))

    def ReadInputFile(self, filename, nSmp=None):
        # Read configuration file.
        self.cvConfig = CVConfig(ConfigObj(filename))

        # For performance statistics.
        # Correct the 'nSmp' in configs with what passed into program.
        if nSmp is not None:
            self.cvConfig.nSmp = nSmp
            self.cvConfig.meshes['wall'].nSmp = nSmp
            self.cvConfig.solver.nSmp = nSmp

        # Loading meshes.
        config = self.cvConfig
        meshes = self.cvConfig.meshes
        equations = self.cvConfig.equations
        self.meshes = {'lumen': FluidMesh(self.comm, config, meshes['lumen'], equations['fluid']),
                       'wall': SolidMesh(self.comm, config, meshes['wall'], equations['solid'])}

    def Distribute(self):
        # TODO:: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        wallPartName = 'MeshPartition/Wall_{}_{}'.format(self.size, self.rank)
        if MeshPartition(wallPartName, self.comm, self.meshes['wall']) < 0:
            print('Distribute mesh failed!')
            return -1

        return 0

    def Coloring(self):
        wallColoringName = 'MeshColoring/Wall_{}_{}'.format(self.size, self.rank)
        if MeshColoring(wallColoringName, self.meshes['wall']) < 0:
            print('Coloring mesh failed!')
            return -1

        return 0


    def Solve(self):

        # TODO:: Try to read the existing calculated results from local files, if exists start from there,
        #        if not start from initial conditions.


        # TODO:: Write the solution has been calculated into files when program has been cutoff accidentally.

        solverSwitcher = {
            'transient generalized-a': TransientGeneralizedASolver,
            'transient': TransientSolver,
            'explicit VMS': TransientExplicitVMSSolver
        }

        SolverClass = solverSwitcher.get(self.cvConfig.solver.method, None)

        if SolverClass is None:
            print('Unknown method: {}'.format(self.cvConfig.solver.method))
            return

        self.solver = SolverClass(self.comm, self.meshes, self.cvConfig.solver)
        self.solver.Solve()

        # TODO:: Write back the calculation result.

    # ? Finalize
