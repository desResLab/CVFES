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


TAG_COMM_DOF = 211
TAG_COMM_DOF_VALUE = 212

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
            integral += f(cls.XW[i][0:3]) * cls.XW[i][3]
        return integral * (2.0*area)


""" Sparse matrix utilities.
"""
class SparseInfo:

    def __init__(self, mesh, dof):
        # Collecting sparse information.
        sparseInfo = [[] for _ in xrange(mesh.nNodes)]
        for iElm, elm in enumerate(mesh.elements):
            for iNode in elm.nodes:
                sparseInfo[iNode].extend(elm.nodes)
        sparseInfo = np.array(sparseInfo)
        for iNode in xrange(mesh.nNodes):
            sparseInfo[iNode] = np.unique(sparseInfo[iNode])

        # Generate sparse matrix.
        indptr = [0] #1
        indices = [] #2
        for iNode in xrange(mesh.nNodes):
            exNodes = SparseInfo.GenerateDof(sparseInfo[iNode], dof)
            for i in xrange(dof): # for 3 by 3 block matrix (3 rows)
                indices.extend(exNodes)
                indptr.extend([len(indices)])

        # Set self attributes.
        self.indptr = np.array(indptr, dtype=int)
        self.indices = np.array(indices, dtype=int)
        self.length = len(self.indices) # Or self.indptr[-1]

        # Remember the degree of freedom.
        self.dof = dof

    def Assemble(self, glbM, elmM, nodeIds, dofs=None):
        if dofs is None:
            dofs = sparseInfo.GenerateDof(nodeIds, self.dof)
        # TODO:: Optimize here because dofs of same node share same indices
        #        so don't need to find it everytime.
        for i, row in enumerate(dofs):
            dofIndices = self.Locate(row, dofs)
            glbM[dofIndices] += elmM[i]

    def Locate(self, row, col):
        """ Return the location of elements in row, col
            where col is array-like.
        """
        return self.indptr[row] + np.where(np.isin(self.indices[self.indptr[row]:self.indptr[row+1]], col))[0]

    def Lump(self, M):
        LM = np.zeros(len(self.indptr) - 1)
        for i in xrange(len(self.indptr) - 1):
            LM[i] = np.sum(M[self.indptr[i]:self.indptr[i+1]])
        return LM

    def MultiplyByVector(self, sparseM, vec):
        """ M dot vec """
        res = np.zeros(len(self.indptr) - 1)
        for i in xrange(len(self.indptr) - 1):
            region = np.arange(self.indptr[i], self.indptr[i+1])
            res[i] = np.dot(sparseM[region], vec[self.indices[region]])

        return res

    @staticmethod
    def GenerateDof(nodeIds, dof):
        tDofs = dof * len(nodeIds)
        indices = np.arange(0, tDofs, dof)

        dofIndices = np.zeros(tDofs, dtype=np.int64)
        for i in xrange(dof):
            dofIndices[indices+i] = dof * nodeIds + i

        return dofIndices


""" Solid and Fluid Solvers.
"""
class PhysicSolver:
    """ One time step solver inside the time loop. """

    def __init__(self, comm, mesh, config):

        self.comm = comm
        # For using convenient.
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        self.mesh = mesh

        # Initialize the context.
        self.ddu = mesh.iniDDu # acceleration
        self.du = mesh.iniDu # velocity ---used by solid
        self.p = mesh.iniP # pressure
        self.u = mesh.iniU # displacement ---used by solid

    def RefreshContext(self, physicSolver):
        pass

    def Solve(self):
        pass


class FluidSolver(PhysicSolver):

    def __init__(self, comm, mesh, config):
        PhysicSolver.__init__(self, comm, mesh, config)


class SolidSolver(PhysicSolver):

    Dof = 3 # Degree of freedom per node, need to optimize afterwards.

    def __init__(self, comm, mesh, config):
        PhysicSolver.__init__(self, comm, mesh, config)

        # Prepare the sparse info structure.
        self.sparseInfo = SparseInfo(mesh, SolidSolver.Dof)

    def Initialize(self, dt):
        # Calculate acceleration in initial using the equilibrium.
        RHS = self.f - self.LC*self.du - self.sparseInfo.MultiplyByVector(self.K, self.u)
        LHS = self.LM
        ddu = np.divide(RHS, LHS, out=np.zeros_like(RHS), where=LHS!=0)
        # Calculate u_-1 = u_0 - dt*du_0 + 0.5*dt**2*ddu_0
        self.up = self.u - dt*self.du + 0.5*(dt**2)*ddu

    def Solve(self, t, dt):
        """ One time step solver. """

        # Assemble the mass, (damping,) stiffness matrix and force vector.
        self.Assemble()
        # Synchronize the common nodes values.
        self.SyncCommNodes(self.LM)
        # self.SyncCommNodes(self.LC, SolidSolver.Dof)

        # Prepare u and up to start time integration.
        if t == 0.0:
            self.Initialize(dt)

        # Calculate the displacement for next time step (n+1).
        LHS = self.LM + 0.5*dt*self.LC
        A1 = (dt**2) * self.f
        A2 = 2.0*self.LM * self.u - self.sparseInfo.MultiplyByVector((dt**2)*self.K, self.u)
        A3 = (0.5*dt*self.LC - self.LM) * self.up
        RHS = A1 + A2 + A3
        u = np.divide(RHS, LHS, out=np.zeros_like(RHS), where=LHS!=0)
        # Synchronize u.
        self.SyncCommNodes(u)
        self.ApplyBoundaryCondition(u)

        # Update the displacement solved.
        self.up = self.u
        self.u = u

        # Post processing: calculate stress and save.
        self.PostProcess()

    def Assemble(self):
        """ Now assume that:
            element type: triangular
        """
        self.M = np.zeros(self.sparseInfo.length)
        self.K = np.zeros(self.sparseInfo.length)
        self.f = np.zeros(len(self.sparseInfo.indptr) - 1)

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
            # TODO:: Form a official way to calculate the force item.
            localf = np.array([0,0,1,0,0,1,0,0,1])*self.mesh.traction * elm.area / 3.0
            # Transform back to the glocal coordinates.
            bT = SolidSolver.BigTransformation(T)
            bTp = np.transpose(bT)
            # Transform.
            M = np.dot(np.dot(bTp, localM), bT)
            K = np.dot(np.dot(bTp, localK), bT)
            f = np.dot(bTp, localf)

            # Assemble!!!
            dofs = SparseInfo.GenerateDof(elm.nodes, SolidSolver.Dof)
            self.sparseInfo.Assemble(self.M, M, elm.nodes, dofs)
            self.sparseInfo.Assemble(self.K, K, elm.nodes, dofs)
            self.f[dofs] += f

        # Lump matrix M and K.
        self.LM = self.sparseInfo.Lump(self.M)
        self.LC = np.zeros(len(self.sparseInfo.indptr) - 1) # TODO:: Add the C matrix.

    def SyncCommNodes(self, quant, dof=None, func=None):
        """ Synchronize the quantity fo common nodes,
            here, quantity is vector-like.
        """
        if dof is None:
            dof = SolidSolver.Dof

        totalCommDofs = SparseInfo.GenerateDof(self.mesh.totalCommNodeIds, dof)
        commDofs = SparseInfo.GenerateDof(self.mesh.commNodeIds, dof)
        commQuant = quant[commDofs]

        totalQuant = np.zeros(len(totalCommDofs))
        if self.rank == 0:

            # totalQuant = np.zeros(len(totalCommDofs))
            # Add on self's (root processor's) quantity.
            indices = np.where(np.isin(totalCommDofs, commDofs))[0]
            totalQuant[indices] += commQuant

            quantIdBuf = np.zeros(len(totalCommDofs), dtype=np.int64)
            quantBuf = np.zeros(len(totalCommDofs))
            recvInfo = MPI.Status()
            for i in xrange(1, self.size):
                self.comm.Recv(quantIdBuf, MPI.ANY_SOURCE, TAG_COMM_DOF, recvInfo)
                recvLen = recvInfo.Get_count(MPI.INT64_T)
                recvSource = recvInfo.Get_source()
                # Receive the quantity.
                self.comm.Recv(quantBuf, recvSource, TAG_COMM_DOF_VALUE, recvInfo)
                # TODO:: make sure the quant received length is consistent with quantIds'.

                # Add the quantity received to the totalQuant.
                indices = np.where(np.isin(totalCommDofs, quantIdBuf[:recvLen]))[0]
                totalQuant[indices] += quantBuf[:recvLen]

            if func is not None:
                func(totalQuant)

        else:

            self.comm.Send(commDofs, 0, TAG_COMM_DOF)
            self.comm.Send(commQuant, 0, TAG_COMM_DOF_VALUE)

            # totalQuant = np.empty(len(totalCommDofs))

        # Get the collected total quantities by broadcast.
        self.comm.Bcast(totalQuant, root=0)
        # Update the original quantity.
        indices = np.where(np.isin(totalCommDofs, commDofs))[0]
        quant[commDofs] = totalQuant[indices]

    def ApplyBoundaryCondition(self, quant): # TODO:: Change to according to configuration.
        bdyDofs = SparseInfo.GenerateDof(self.mesh.boundary, SolidSolver.Dof)
        quant[bdyDofs] = 0

    def PostProcess(self):
        # Update the coordinate.
        # Calculate the stress with updated coordinates.
        pass

    @staticmethod
    def CoordinateTransformation(nodes):
        # Calculate two edges.
        edge0 = nodes[2]-nodes[1]
        edge1 = nodes[0]-nodes[2]

        # Calculate the transform matrix.
        T = np.zeros([3, 3])
        T[0] = SolidSolver.Normalize(edge0)
        T[1] = SolidSolver.Normalize(edge1 - np.dot(edge1, T[0]) * T[0])
        T[2] = np.cross(T[0], T[1])
        return T

    @staticmethod
    def BigTransformation(T):
        bT = np.zeros([9, 9])
        bT[0:3,0:3] = bT[3:6,3:6] = bT[6:9,6:9] = T
        return bT

    @staticmethod
    def Normalize(vec):
        return vec / np.linalg.norm(vec)


""" Generalized-a method
"""
class GeneralizedAlphaSolver(PhysicSolver):

    def __init__(self, comm, mesh, config):
        PhysicSolver.__init__(self, comm, mesh, config)

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

    def __init__(self, comm, mesh, config):
        GeneralizedAlphaSolver.__init__(self, comm, mesh, config)

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

    def __init__(self, comm, mesh, config):
        GeneralizedAlphaSolver.__init__(self, comm, mesh, config)

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

    def __init__(self, comm, mesh, config): # the config is actually solver config
        self.comm = comm
        self.mesh = mesh

    def Solve(self):
        pass


class TransientSolver(Solver):
    """ Solver employing time looping style, where inertial is not trivial."""

    def __init__(self, comm, mesh, config):
        Solver.__init__(self, comm, mesh, config)

        # Set the current time which also the time to start,
        # it might not be 0 in which case solving starts from
        # results calculated last time and written into a file.
        self.time = config.time
        self.dt = config.dt
        self.endtime = config.endtime

        # Set the tolerance used to decide where stop calculating.
        self.tolerance = config.tolerance

        # Init the solver which is inside of the time loop.
        self.__initPhysicSolver__(comm, mesh, config)

    def __initPhysicSolver__(self, comm, mesh, config):
        """ Initialize the fluid and solid solver. """

        self.fluidSolver = FluidSolver(comm, mesh, config)
        self.solidSolver = SolidSolver(comm, mesh, config)

    def Solve(self):

        while self.time < self.endtime:
            # Solve for the fluid part.
            self.fluidSolver.Solve()
            # Solve for the solid part based on
            # calculation result of fluid part.
            self.solidSolver.RefreshContext(self.fluidSolver)
            self.solidSolver.Solve(self.time, self.dt)
            # Refresh the fluid solver's context
            # before next loop start.
            self.fluidSolver.RefreshContext(self.solidSolver)

            self.time += self.dt


""" For generalized-a method:
"""
class TransientGeneralizedASolver(TransientSolver):
    """ Time looping style solver which employs the
        generalized-a time integration algorithm.
    """

    def __init__(self, comm, mesh, config):
        TransientSolver.__init__(self, comm, mesh, config)

    def __initPhysicSolver__(self, comm, mesh, config):
        self.fluidSolver = GeneralizedAlphaFluidSolver(comm, mesh, config)
        self.solidSolver = GeneralizedAlphaSolidSolver(comm, mesh, config)

