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

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


""" Shape functions
"""
class Shape:

    def __init__(self, nodes, area):
        self.nodes = nodes
        self.area = area

class TriangularForSolid(Shape):
    """ Constant-strain triangular element for solid. """

    k = 5.0/6.0 # parameter for CMM method (refer to CMM paper)

    def __init__(self, nodes, area):
        Shape.__init__(self, nodes, area)

    def N(self, xi):
        return np.array([[xi[0], 0, 0, xi[1], 0, 0, xi[2], 0, 0],
                         [0, xi[0], 0, 0, xi[1], 0, 0, xi[2], 0],
                         [0, 0, xi[0], 0, 0, xi[1], 0, 0, xi[2]]])

    def B(self):
        # Calculate the temporary params.
        y23 = nodes[1,1] - nodes[2,1]
        y31 = nodes[2,1] - nodes[0,1]
        y12 = nodes[0,1] - nodes[1,1]
        x32 = nodes[2,0] - nodes[1,0]
        x13 = nodes[0,0] - nodes[2,0]
        x21 = nodes[1,0] - nodes[0,0]

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


""" Gaussian quadrature.
    TODO:: Add more details and subclasses according to different shape functions
        and accuracy order needed. Right now is only for quadratic triangle.
"""
class GaussianQuadrature:

    # Xi's and corresponding weights.
    XW = [[0.5, 0.5, 0, 1.0/3.0],
          [0.5, 0, 0.5, 1.0/3.0],
          [0, 0.5, 0.5, 1.0/3.0]]

    def __init__(self):
        pass

    @classmethod
    def Integrate(cls, f, area):
        integral = 0.0
        for i in xrange(len(cls.XW)):
            integral += f(XW[i,0:3]) * XW[i,3]
        return integral * (2.0*area)


""" Solid and Fluid Solvers.
"""
class PhysicSolver:
    """ One time step solver inside the time loop. """

    def __init__(self, mesh, config):

        self.mesh = mesh

        # Initialize the context.
        self.ddu = mesh.iniDDu # acceleration
        self.du = mesh.iniDu # velocity
        self.p = mesh.iniP # pressure
        self.u = mesh.iniU # displacement

    def RefreshContext(self, physicSolver):
        pass

    def Solve(self):
        pass


class FluidSolver(PhysicSolver):

    def __init__(self, mesh, config):
        PhysicSolver.__init__(self, mesh, config)


class SolidSolver(PhysicSolver):

    def __init__(self, mesh, config):
        PhysicSolver.__init__(self, mesh, config)

    def Solve(self):
        pass

    def Assemble(self):
        """ Now assume that:
            element type: triangular
        """
        # The elemental D matrix is static.
        tD = TriangularForSolid.D(self.mesh.E, self.mesh.v)

        # Start to loop through the elements.
        for iElm, elm in enumerate(self.mesh.elements):
            # Get element nodes with coordinates.
            nodes = self.mesh.nodes[elm.nodes]
            # Transform the global coordinates into local plain one.
            T = SolidSolver.CoordinateTransformation(nodes)
            # Transform.
            nodes = np.dot(nodes, np.transpose(T))
            # Calculate local mass and stiffness matrix.
            triangular = TriangularForSolid(nodes, elm.area)
            localM = GaussianQuadrature.Integrate(
                            lambda xi: np.dot(np.transpose(triangular.N(xi)),triangular.N(xi)),
                            elm.area) * self.mesh.density
            localK = np.dot(np.dot(np.transpose(triangular.B()), tD), triangular.B()) * elm.area
            # Calculate the RHS f.
            # TODO:: Add the body force and initial stress and strain conditions.
            # TODO:: Figure out what's the form of body force, traction and initial strain, eg. what's the right hand side.
            localf = np.dot(triangular.N([1,1,1]/3.0), np.array([0,0,1,0,0,1,0,0,1])*self.mesh.traction) * elm.area
            # Transform back to the glocal coordinates.
            bT = SolidSolver.BigTransformation(T)
            bTp = np.transpose(bT)
            # Transform.
            M = np.dot(np.dot(bTp, localM), bT)
            K = np.dot(np.dot(bTp, localK), bT)
            f = np.dot(bTp, localf)

            # Assemble!!!

    @staticmethod
    def CoordinateTransformation(nodes):
        # Calculate two edges.
        edge0 = nodes[2]-nodes[1]
        edge1 = nodes[0]-nodes[2]

        # Calculate the transform matrix.
        T = np.zeros([3, 3])
        T[0] = normalize(edge0)
        T[1] = normalize(edge1 - np.dot(edge1, T[0]) * T[0])
        T[2] = np.cross(T[0], T[1])
        return T

    @staticmethod
    def BigTransformation(T):
        bT = np.zeros([9, 9])
        bT[0:3,0:3] = bT[3:6,3:6] = bT[6:9,6:9] = T
        return bT


""" Generalized-a method
"""
class GeneralizedAlphaSolver(PhysicSolver):

    def __init__(self, mesh, config):
        PhysicSolver.__init__(self, mesh, config)

        # Calculate the prameters gonna used
        self.rho_infinity = config.rho_infinity
        self.alpha_m = 1.0 / (1.0+self.rho_infinity)
        self.alpha_f = (3.0-self.rho_infinity) / (2.0+2.0*self.rho_infinity)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f

    def Solve(self):

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


class GeneralizedAlphaFluidSolver(GeneralizedAlphaSolver):

    def __init__(self, mesh, config):
        GeneralizedAlphaSolver.__init__(self, mesh, config)

    def RefreshContext(self, physicSolver):
        self.du = physicSolver.du

    def Predict(self):
        # Reset the previous context to current one to start a new loop.
        # The xP represent previous value.
        self.dduP = self.ddu
        self.duP = self.du
        self.pP = self.p

        # predict ddu using parameter gamma.
        self.ddu = (self.gamma-1)/self.gamma * self.dduP
        # u dose not change.
        self.du = self.duP
        # p does not change.
        self.p = self.pP

    def Initialize(self):

        self.interDDu = (1-self.alpha_m)*self.dduP + self.alpha_m*self.ddu
        self.interDu = (1-self.alpha_f)*self.duP + self.alpha_f*self.du

    def Assemble(self): # TODO:: distinguish btw solid part and fluid part in meshes.

        # Loop through the local mesh to do the assembling.
        for iElm, elm in enumerate(self.mesh.elements):
            pass


class GeneralizedAlphaSolidSolver(GeneralizedAlphaSolver):

    def __init__(self, mesh, config):
        GeneralizedAlphaSolver.__init__(self, mesh, config)

        self.beta = 0.25 * (1 + self.alpha_f - self.alpha_m)**2

    def RefreshContext(self, physicSolver):
        self.ddu = physicSolver.ddu
        self.du = physicSolver.du
        self.p = physicSolver.p

    def Predict(self):
        # Reset the previous context to current one to start a new loop.
        # The xP represent previous value.
        self.uP = self.u

        # predict d.
        self.u = self.uP + self.du * self.dt + (self.gamma*0.5 - self.beta)/(self.gamma - 1) * self.ddu * (self.dt ** 2)

    def Initialize(self):

        self.interU = (1-self.alpha_f)*self.uP + self.alpha_f*self.u


""" This is the big solver we are going to use here.
"""
class Solver:

    def __init__(self, mesh, config): # the config is actually solver config
        self.mesh = mesh

    def Solve(self):
        pass


class TransientSolver(Solver):
    """ Solver employing time looping style, where inertial is not trivial."""

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
        self.__initPhysicSolver__(mesh, config)

    def __initPhysicSolver__(self, mesh, config):
        """ Initialize the fluid and solid solver. """

        self.fluidSolver = FluidSolver(mesh, config)
        self.solidSolver = SolidSolver(mesh, config)

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


""" For generalized-a method:
"""
class TransientGeneralizedASolver(TransientSolver):
    """ Time looping style solver which employs the
        generalized-a time integration algorithm.
    """

    def __init__(self, mesh, config):
        TransientSolver.__init__(self, mesh, config)

    def __initPhysicSolver__(self, mesh, config):
        self.fluidSolver = GeneralizedAlphaFluidSolver(mesh, config)
        self.solidSolver = GeneralizedAlphaSolidSolver(mesh, config)

