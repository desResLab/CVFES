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
from math import floor
from math import cos, pi
import numpy as np

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"

""" Gaussian quadrature.
    TODO:: Add more details and subclasses according to different shape functions
        and accuracy order needed. Right now is only for quadratic triangle.
"""
class GaussianQuadrature:

    # Xi's and corresponding weights.
    # XW = np.array([[0.5, 0.5, 0, 1.0/3.0],
    #               [0.5, 0, 0.5, 1.0/3.0],
    #               [0, 0.5, 0.5, 1.0/3.0]])
    XW = [[0.5, 0.5, 0, 1.0/3.0],
          [0.5, 0, 0.5, 1.0/3.0],
          [0, 0.5, 0.5, 1.0/3.0]]
    # XW = [[1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/2.0]]

    def __init__(self):
        pass

    @classmethod
    def Integrate(cls, f, area):
        integral = 0.0
        for i in range(len(cls.XW)):
            integral += f(cls.XW[i][0:3]) * cls.XW[i][3]
        return integral * (area)

class GQTetrahedron(GaussianQuadrature):

    # Xi's and corresponding weights.
    # For 1st order accuracy.
    XW = np.array([[(5.0+3.0*5.0**0.5)/20.0, (5.0-5.0**0.5)/20.0, (5.0-5.0**0.5)/20.0, 1.0/24.0],
                   [(5.0-5.0**0.5)/20.0, (5.0+3.0*5.0**0.5)/20.0, (5.0-5.0**0.5)/20.0, 1.0/24.0],
                   [(5.0-5.0**0.5)/20.0, (5.0-5.0**0.5)/20.0, (5.0+3.0*5.0**0.5)/20.0, 1.0/24.0],
                   [(5.0-5.0**0.5)/20.0, (5.0-5.0**0.5)/20.0, (5.0-5.0**0.5)/20.0, 1.0/24.0]])

    def __init__(self, detJ):
        GaussianQuadrature.__init__(self)
        # self.w = (self.XW[:,-1] * detJ).reshape(1,4)
        self.w = self.XW[:,-1] * detJ

    def IntegrateByMatrix(self, m):
        w = self.w
        return w[0]*m[0] + w[1]*m[1] + w[2]*m[2] + w[3]*m[3]

    @classmethod
    def Integrate(cls, f, detJ):
        integral = 0.0
        for i in range(len(cls.XW)):
            integral += f(cls.XW[i][0], cls.XW[i][1], cls.XW[i][2]) * cls.XW[i][3]
        return integral * (detJ)

    @classmethod
    def IntegrateFunc(cls, detJ):
        def IntegrateByMatrix(m):
            integral = 0.0
            for i,mi in enumerate(m):
                integral += mi * cls.XW[i][3]
            return integral * detJ
        return IntegrateByMatrix
