#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Solver class provides the solver of the CVFES project.
"""

from cvconfig import CVConfig
from mpi4py import MPI
from mesh import *
from math import floor

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


class Solver:

    def __init__(self, config):
        pass

    def Solve(self):
        pass


class TransientSolver(Solver):
    """ Solver employing time looping style. """

    def __init__(self, config):
        Solver.__init__(self, config)

        # Set the current time which also the time to start,
        # it might not be 0 in which case solving starts from
        # results calculated last time and written into a file.
        self.time = config.solver.time
        self.dt = config.solver.dt
        self.endtime = config.solver.endtime

        # Set the tolerance used to decide where stop calculating.
        self.tolerance = config.solver.tolerance

    def Solve(self):

        while self.time < self.endtime:

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

    def Predict(self):
        pass

    def Initialize(self):
        pass

    def Assemble(self):
        pass

    def SolveLinearSystem(self):
        pass

    def Correct(self):
        pass

class TransientGeneralizedASolver(TransientSolver):
    """ Time looping style solver which employs the
        generalized-a time integration algorithm.
    """

    def __init__(self, config):
        TransientSolver.__init__(self, config)
