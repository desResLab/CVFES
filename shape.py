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

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


""" Shape functions
"""
class Shape:

    def __init__(self, nodes):
        self.nodes = nodes
        self.area = 0.0

class TriangularForSolid(Shape):
    """ Constant-strain triangular element for solid.
        Make sure the nodes used are on the local plane.
    """

    k = 5.0/6.0 # parameter for CMM method (refer to CMM paper)

    def __init__(self, nodes):
        Shape.__init__(self, nodes)

        self.area = np.linalg.det([[1, nodes[0,0], nodes[0,1]],
                                   [1, nodes[1,0], nodes[1,1]],
                                   [1, nodes[2,0], nodes[2,1]]]) * 0.5

    def N(self, xi):
        return np.array([[xi[0], 0, 0, xi[1], 0, 0, xi[2], 0, 0],
                         [0, xi[0], 0, 0, xi[1], 0, 0, xi[2], 0],
                         [0, 0, xi[0], 0, 0, xi[1], 0, 0, xi[2]]])

    def B(self):
        # Calculate the temporary params.
        y23 = self.nodes[1,1] - self.nodes[2,1]
        y31 = self.nodes[2,1] - self.nodes[0,1]
        y12 = self.nodes[0,1] - self.nodes[1,1]
        x32 = self.nodes[2,0] - self.nodes[1,0]
        x13 = self.nodes[0,0] - self.nodes[2,0]
        x21 = self.nodes[1,0] - self.nodes[0,0]

        return np.array([[y23, 0, 0, y31, 0, 0, y12, 0, 0],
                         [0, x32, 0, 0, x13, 0, 0, x21, 0],
                         [x32,y23,0, x13,y31,0, x21,y12,0],
                         [0, 0, y23, 0, 0, y31, 0, 0, y12],
                         [0, 0, x32, 0, 0, x13, 0, 0, x21]
                        ]) / (2.0*self.area)

    @classmethod
    def D(cls, E, v):
        """ Calculate the static D matrix. """
        return np.array([[1.0, v, 0, 0, 0],
                         [v, 1.0, 0, 0, 0],
                         [0, 0, 0.5*(1-v), 0, 0],
                         [0, 0, 0, 0.5*cls.k*(1-v), 0],
                         [0, 0, 0, 0, 0.5*cls.k*(1-v)]])*E/(1-v**2)

# Linear Tetrahedron.
class Tetrahedron(Shape):

    # Define shape functions.
    # shapes = [lambda r,s,t: 1 - r - s - t,
    #           lambda r,s,t: r,
    #           lambda r,s,t: s,
    #           lambda r,s,t: t]

    shapes = [lambda r,s,t: r,
              lambda r,s,t: s,
              lambda r,s,t: t,
              lambda r,s,t: 1 - r - s - t]

    # Define the derivative of shape functions.
    # One shape function's derivatives w.r.t. r,s,t stored in one row.
    ds = [[ 1.0,  0.0,  0.0],
          [ 0.0,  1.0,  0.0],
          [ 0.0,  0.0,  1.0],
          [-1.0, -1.0, -1.0]]

    def __init__(self, nodes):
        Shape.__init__(self, nodes)

        # areaM = np.ones((4, 4))
        # areaM[:,1:] = nodes
        # self.volumn = np.linalg.det(areaM) / 6.0

        # Define the Jacobian matrix and its determinate.
        x = nodes[:,0]
        y = nodes[:,1]
        z = nodes[:,2]
        self.jacobian = [[x[1]-x[0], x[2]-x[0], x[3]-x[0]],
                         [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
                         [z[1]-z[0], z[2]-z[0], z[3]-z[0]]]
        self.detJ = np.linalg.det(self.jacobian)

        # Define the global derivatives.
        invJ = np.linalg.inv(self.jacobian)
        self.ds = np.dot(Tetrahedron.ds, invJ)
        # Element metric tensor.
        self.G = np.transpose(invJ).dot(invJ)
