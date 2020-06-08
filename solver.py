#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Solver class provides the solver of the CVFES project.
    One Solver instance corresponds to one mesh and one method
    which can be decided by solver configuration.

    du: velocity
    p: pressure
    ddu: acceleration
    u: displacement
"""

from cvconfig import CVConfig
from mpi4py import MPI
from mesh import *
from physicsSolver import *
from physicsSolverGPUs import *
from generalizedAlphaSolver import *
# from generalizedAlphaSolverRe import *
from explicitVMSSolver import *

# from math import floor
# from math import cos, pi
import math
import sys

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


TAG_COMM_DOF = 211
TAG_COMM_DOF_VALUE = 212
# TAG_ELM_ID = 221
TAG_STRESSES = 222
TAG_DISPLACEMENT = 223
# TAG_UNION = 224
TAG_CHECKING_STIFFNESS = 311


""" This is the big solver we are going to use here.
"""
class Solver:

    def __init__(self, comm, meshes, config): # the config is actually solver config
        self.comm = comm
        self.meshes = meshes

        self.saveResNum = config.saveResNum
        self.saveStressFilename = config.saveStressFilename

    def Solve(self):
        pass


class TransientSolver(Solver):
    """ Solver employing time looping style, where inertial is not trivial."""

    def __init__(self, comm, meshes, config):
        Solver.__init__(self, comm, meshes, config)

        # Set the current time which also the time to start,
        # it might not be 0 in which case solving starts from
        # results calculated last time and written into a file.
        self.time = config.time
        self.dt = config.dt
        self.endtime = config.endtime

        self.restart = config.restart
        self.restartFilename = config.restartFilename
        self.restartTimestep = config.restartTimestep

        self.t = np.append(np.arange(self.endtime, step=self.dt), self.endtime)

        # Calculate the pressure applied for solid part.
        # a = b = config.constant_pressure/2.0
        # n = math.pi/config.constant_T
        # self.appPressures = a - b*np.cos(n*self.t)
        # self.appPressures[self.t >= config.constant_T] = config.constant_pressure

        # self.appPressures = config.constant_pressure
        # self.appPressures = 0.0

        # Total time steps.
        self.nTimeSteps = len(self.t) - 1

        # Init the solver which is inside of the time loop.
        self.__initPhysicSolver__(comm, meshes, config)


    def __initPhysicSolver__(self, comm, meshes, config):
        """ Initialize the fluid and solid solver. """

        self.fluidSolver = FluidSolver(comm, meshes['lumen'], config)
        # self.solidSolver = SolidSolver(comm, meshes['wall'], config)
        # self.solidSolver = SolidSolver(comm, meshes['wall'], config,
        #                                self.appPressures[self.restartTimestep])

        self.solidSolver = GPUSolidSolver(comm, meshes['wall'], config)

    def Solve(self):

        # Calculate when to save the result into file.
        saveSteps = np.linspace(0, self.nTimeSteps, self.saveResNum+1, dtype=int)

        for timeStep in range(self.restartTimestep, self.nTimeSteps):
            t = round(self.t[timeStep], 4)
            dt = self.t[timeStep+1] - self.t[timeStep]

            # Solve for the fluid part.
            self.fluidSolver.Solve(t, dt)

            # Solve for the solid part based on calculation result of fluid part.
            self.solidSolver.RefreshContext(self.fluidSolver)
            self.solidSolver.Solve(t, dt)
            # Refresh the fluid solver's context
            # before next loop start.
            self.fluidSolver.RefreshContext(self.solidSolver)

            if timeStep+1 in saveSteps:
                self.fluidSolver.Save(self.saveStressFilename, timeStep+1)
                # self.solidSolver.Save(self.saveStressFilename, timeStep+1)
                self.solidSolver.SaveDisplacement(self.saveStressFilename, timeStep+1)

            # Tell if flow becomes steady with constant inflow.
            if self.fluidSolver.Steady():
                print('Fluid reaches steady at time {}s'.format(t))
                break


""" For generalized-a method:
"""
class TransientGeneralizedASolver(TransientSolver):
    """ Time looping style solver which employs the
        generalized-a time integration algorithm.
    """

    def __init__(self, comm, meshes, config):
        TransientSolver.__init__(self, comm, meshes, config)

    def __initPhysicSolver__(self, comm, meshes, config):
        self.fluidSolver = GeneralizedAlphaFluidSolver(comm, meshes['lumen'], config)
        self.solidSolver = GeneralizedAlphaSolidSolver(comm, meshes['wall'], config)


""" For explicit VMS method:
"""
class TransientExplicitVMSSolver(TransientSolver):
    """ Time looping style solver which employs the
        explicit VMS method.
    """

    def __init__(self, comm, meshes, config):
        TransientSolver.__init__(self, comm, meshes, config)

    def __initPhysicSolver__(self, comm, meshes, config):
        self.fluidSolver = ExplicitVMSSolver(comm, meshes['lumen'], config)
        self.solidSolver = GeneralizedAlphaSolidSolver(comm, meshes['wall'], config)

