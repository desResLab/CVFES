#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Solver class provides the solver of the CVFES project.
    One Solver instance corresponds to one mesh and one method
    which can be decided by solver configuration.

    u: velocity
    p: pressure
    du: acceleration
    d: displacement
"""

from cvconfig import CVConfig
from mpi4py import MPI
from mesh import *
from math import floor

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


class PhysicSolver:
    """ One time step solver inside the time loop. """

    def __init__(self, mesh, config):

        self.mesh = mesh

        # Initialize the context.
        self.du = mesh.iniDu
        self.u = mesh.iniU
        self.p = mesh.iniP
        self.d = mesh.iniD

        # Calculate the prameters gonna used
        self.rho_infinity = config.rho_infinity
        self.alpha_m = 1.0 / (1.0+self.rho_infinity)
        self.alpha_f = (3.0-self.rho_infinity) / (2.0+2.0*self.rho_infinity)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f

    def RefreshContext(self, physicSolver):
        pass

    def Solve():

        # Predict the start value of current time step.
        self.Predict()

        # Newton-Raphson loop to approximate.
        while True:

            # Initialize the value of current loop of current time step.
            self.Initialize()

            # Assemble the RHS and LHS of the linear system.
            self.Assemble()

            # Solve the linear system assembled.
            self.SolveLinearSystem()

            # Do the correction.
            self.Correct()

            # Decide if it's been converged.
            if residual < self.tolerance:
                break

    def Predict(self, du, u, p):
        pass

    def Initialize(self):
        pass

    def Assemble(self):
        pass

    def SolveLinearSystem(self):
        pass

    def Correct(self):
        pass


class FluidSolver(PhysicSolver):

    def __init__(self):
        PhysicSolver.__init__(self)

    def RefreshContext(self, physicSolver):
        self.d = physicSolver.d

    def Predict(self):
        # Reset the previous context to current one to start a new loop.
        # The xP represent previous value.
        self.duP = self.du
        self.uP = self.u
        self.pP = self.p

        # predict du using parameter gamma.
        self.du = (self.gamma-1)/self.gamma * self.duP
        # u dose not change.
        self.u = self.uP
        # p does not change.
        self.p = self.pP

    def Initialize(self):

        self.interDu = (1-self.alpha_m)*self.duP + self.alpha_m*self.du
        self.interU = (1-self.alpha_f)*self.uP + self.alpha_f*self.u

    def Assemble(self):

        # Loop through the local mesh to do the assembling.
        for iElm, elm in enumerate(self.mesh.elements):
            pass


class SolidSolver(PhysicSolver):

    def __init__(self):
        PhysicSolver.__init__(self)

        self.beta = 0.25 * (1 + self.alpha_f - self.alpha_m)**2

    def RefreshContext(self, physicSolver):
        self.du = physicSolver.du
        self.u = physicSolver.u
        self.p = physicSolver.p

    def Predict(self):
        # Reset the previous context to current one to start a new loop.
        # The xP represent previous value.
        self.dP = self.d

        # predict d.
        self.d = self.dP + self.u * self.dt + (self.gamma*0.5 - self.beta)/(self.gamma - 1) * self.du * (self.dt ** 2)

    def Initialize(self):

        self.interD = (1-self.alpha_f)*self.dP + self.alpha_f*self.d


class Solver:

    def __init__(self, mesh, config): # the config is actually solver config
        self.mesh = mesh

    def Solve(self):
        pass


class TransientSolver(Solver):
    """ Solver employing time looping style. """

    def __init__(self, mesh, config):
        Solver.__init__(self, mesh, config)

        # Set the current time which also the time to start,
        # it might not be 0 in which case solving starts from
        # results calculated last time and written into a file.
        self.time = config.time
        self.dt = config.dt
        self.endtime = config.endtime

        # Set the tolerance used to decide where stop calculating.
        self.tolerance = config.tolerance

        # Init the solver which is inside of the time loop.
        self.__initPhysicSolver__()

    def __initPhysicSolver__(self):
        """ Initialize the fluid and solid solver. """

        self.fluidSolver = PhysicSolver()
        self.solidSolver = PhysicSolver()

    def Solve(self):

        while self.time < self.endtime:
            # Solve for the fluid part.
            self.fluidSolver.Solve()
            # Solve for the solid part based on
            # calculation result of fluid part.
            self.solidSolver.RefreshContext(self.fluidSolver)
            self.solidSolver.Solve()
            # Refresh the fluid solver's context
            # before next loop start.
            self.fluidSolver.RefreshContext(self.solidSolver)

class TransientGeneralizedASolver(TransientSolver):
    """ Time looping style solver which employs the
        generalized-a time integration algorithm.
    """

    def __init__(self, mesh, config):
        TransientSolver.__init__(self, mesh, config)

    def __initPhysicSolver__(self):
        self.fluidSolver = FluidSolver()
        self.solidSolver = SolidSolver()

