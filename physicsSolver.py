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
from mpi4py import MPI
from math import floor
from math import cos, pi

from cvconfig import CVConfig
from mesh import *
# from shape import *
from sparse import *

# from optimizedSolidAssemble import D
from optimizedSolidAssemble import OptimizedSolidAssemble
# # from optimizedSolidAssemble import OptimizedSolidAssembleWithDamping
from optimizedSolidAssemble import OptimizedSolidAssembleUpdate
# from optimizedSolidAssemble import OptimizedSolidAssembleUpdatef
# # from optimizedSolidAssemble import OptimizedSolidAssembleUpdateWithDampingForce
from optimizedSolidAssemble import OptimizedCalculateStress
from optimizedSolidAssemble import MultiplyByVector, MultiplyBy1DVector

# from MKL import MKL
from timeit import default_timer as timer


__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


TAG_COMM_DOF = 211
TAG_COMM_DOF_VALUE = 212
# TAG_ELM_ID = 221
TAG_STRESSES = 222
TAG_DISPLACEMENT = 223
# TAG_UNION = 224
TAG_CHECKING_STIFFNESS = 311


""" Solid and Fluid Solvers.
"""
class PhysicsSolver:
    """ One time step solver inside the time loop. """

    Dof = 3 # Degree of freedom per node, need to optimize afterwards.

    def __init__(self, comm, mesh, config):

        self.comm = comm
        # For using convenient.
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        self.mesh = mesh
        self.Dof = mesh.dof

        # Remember the calculation constants.
        self.dt = config.dt

    def RefreshContext(self, physicSolver):
        pass

    def Solve(self, t, dt):
        pass

    def Save(self, filename, counter):
        pass

    def SaveDisplacement(self, filename, counter):
        pass

    def Steady(self):
        return False


class FluidSolver(PhysicsSolver):

    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)

        # Initialize the context.
        self.ddu = mesh.iniDDu # acceleration
        self.du = mesh.iniDu # velocity
        self.p = mesh.iniP # pressure


class SolidSolver(PhysicsSolver):

    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)

        # Comparison btw MKL and my Cython implementation of sparse matrix vector multiplication.
        # self.mkl = MKL()
        # self.sparseMultiplierTimer = 0.0

        # Initialize the context.
        self.du = mesh.iniDu # velocity
        self.u = mesh.iniU # displacement
        self.appTraction = 0.0 # pressure applied


        # For test, random initial displacement.
        # um = np.random.random(self.mesh.nNodes)*1.0e-6
        # self.u = np.repeat(um, 3)
        # print self.u[:6]

        # Prepare the sparse info structure.
        self.sparseInfo = SparseInfo(mesh, self.Dof)

        # For debug info, counting how many operations may include.
        # print('Number of operations {}'.format(self.sparseInfo.OperationCounting()))

        # Get the number of samples of GMRF.
        self.nSmp = config.nSmp
        self.M = np.zeros((self.sparseInfo.length, self.nSmp, self.Dof, self.Dof))
        self.C = np.zeros((self.sparseInfo.length, self.nSmp, self.Dof, self.Dof))
        self.K = np.zeros((self.sparseInfo.length, self.nSmp, self.Dof, self.Dof))
        self.f = np.zeros(self.sparseInfo.ndof)

        self.LM = np.zeros((self.sparseInfo.ndof, self.nSmp))
        self.LC = np.zeros((self.sparseInfo.ndof, self.nSmp))

        # Constant coefficients or matrices used.
        self.InitializeProperties()
        self.LocalInitialize()

        # Counter for the update interval for stiffness matrix.
        self.updateCounter = 0
        self.updateInterval = config.update_interval

    def InitializeProperties(self):
        """ Prepare the properties on each Gaussian point of each element. """

        self.xw = np.array([[0.5, 0.5, 0, 1.0/3.0],
                            [0.5, 0, 0.5, 1.0/3.0],
                            [0, 0.5, 0.5, 1.0/3.0]])
        self.wt = self.xw[:,3].copy(order='C')
        xw = self.xw
        gps = xw[:,:3]

        # Young's Modulus
        elmVerE = self.mesh.vE[self.mesh.elementNodeIds,:]
        elmVerE = elmVerE.swapaxes(1,2)
        self.elmGE = np.matmul(elmVerE, gps.T)

        # thickness
        elmVerThick = self.mesh.vthickness[self.mesh.elementNodeIds,:]
        elmVerThick = elmVerThick.swapaxes(1,2)
        self.elmAveThick = np.mean(elmVerThick, axis=2)
        self.elmGThick = np.matmul(elmVerThick, gps.T).copy(order='C')

        # Dofs used for synchronization.
        if self.size > 1:
            self.totalCommDofs = np.array([[i*3, i*3+1, i*3+2] for i in self.mesh.totalCommNodeIds]).astype(int).ravel()
            self.commDofs = np.array([[i*3, i*3+1, i*3+2] for i in self.mesh.commNodeIds]).astype(int).ravel()
        # Boundary dofs.
        self.bdyDofs = np.array([[3*node, 3*node+1, 3*node+2] for node in self.mesh.boundary]).astype(int).ravel()


    def LocalInitialize(self):
        """ Prepare u and up to start time integration. """
        self.k = 5.0/6.0
        self.D = None
        self.localM = np.zeros((3, 9, 9))

        # Preparing items used in calculations afterwards.
        v = self.mesh.v
        k = self.k
        self.D = np.array([[1.0,   v,       0.0,         0.0,         0.0],
                           [  v, 1.0,       0.0,         0.0,         0.0],
                           [0.0, 0.0, 0.5*(1-v),         0.0,         0.0],
                           [0.0, 0.0,       0.0, 0.5*k*(1-v),         0.0],
                           [0.0, 0.0,       0.0,         0.0, 0.5*k*(1-v)]])/(1-v*v)

        # Prepared the local mass matrix.
        xw = self.xw
        for i in range(len(xw)):
            N = np.array([[xw[i][0], 0, 0, xw[i][1], 0, 0, xw[i][2], 0, 0],
                         [0, xw[i][0], 0, 0, xw[i][1], 0, 0, xw[i][2], 0],
                         [0, 0, xw[i][0], 0, 0, xw[i][1], 0, 0, xw[i][2]]])

            self.localM[i,:,:] = np.dot(N.T, N)*xw[i][3]*self.mesh.density

        # Coefficients of K matrix, (nElm, nSmp)
        gTE = self.elmGThick * self.elmGE
        self.coefK = gTE[:,:,0]*xw[0][3] + gTE[:,:,1]*xw[1][3] + gTE[:,:,2]*xw[2][3]
        self.coefK = self.coefK.copy(order='C')

        # ------------------------ Prepare for first loop ------------------------------
        # First assemble the matrices.
        self.OptimizedAssemble()
        # self.OptimizedAssembleWithDamping()

        # Calculate acceleration in initial using the equilibrium.
        Ku = np.zeros((self.sparseInfo.ndof, self.nSmp))
        MultiplyBy1DVector(self.sparseInfo.indptr, self.sparseInfo.indices, self.K, self.u, Ku)

        # Reshape du for calculating.
        self.du = np.tile(self.du[:,np.newaxis], (1,self.nSmp))
        self.u = np.tile(self.u[:,np.newaxis], (1,self.nSmp))

        # Calculate ddu_0 according to M*ddu_0 + C*du_0 + K*u_0 = f_0
        RHS = self.f - self.LC*self.du - Ku
        LHS = np.copy(self.LM)
        self.SyncCommNodes(LHS) # Only left-hand-side needs to be synchronized.
        ddu = np.divide(RHS, LHS, out=np.zeros_like(RHS), where=LHS!=0)

        # Calculate u_-1 = u_0 - dt*du_0 + 0.5*dt**2*ddu_0
        dt = self.dt
        self.up = self.u - dt*self.du + 0.5*(dt**2)*ddu
        self.SyncCommNodes(self.up)

        # Prepare the left hand side.
        self.LHS = self.LM + 0.5*dt*self.LC
        self.SyncCommNodes(self.LHS) # Only left-hand-side needs to be synchronized.

    def OptimizedAssemble(self):

        coefs = np.array([self.appTraction])

        OptimizedSolidAssemble(self.mesh.nodes, self.mesh.elementNodeIds, coefs, self.wt,
                               self.localM, self.D, self.coefK, self.elmGThick,
                               self.sparseInfo.indptr, self.sparseInfo.indices,
                               self.M, self.K, self.f, self.LM)

        self.f = self.f[:,np.newaxis]

    # def OptimizedAssembleWithDamping(self):
    #     """ With damping version. """

    #     # coefs = np.array([self.appTraction])
    #     coefs = np.array([1.0, self.mesh.damp])
    #     wt = np.array([self.xw[i,3] for i in range(len(self.xw))])

    #     OptimizedSolidAssembleWithDamping(self.mesh.nodes, self.mesh.elementNodeIds, coefs, wt,
    #                                       self.localM, self.D, self.coefK, self.elmGThick,
    #                                       self.sparseInfo.indptr, self.sparseInfo.indices,
    #                                       self.M, self.C, self.K, self.f, self.LM, self.LC)

    #     self.f = self.f[:,np.newaxis]
    #     # self.f = self.f.reshape((self.mesh.ndof, 1))


    def ApplyPressure(self, appTraction):
        self.appTraction = appTraction


    def Solve(self, t, dt):
        """ One time step solver. """
        if self.rank == 0:
            print('Current time is {}.\n'.format(t))

        # Assemble the stiffness matrix and force vector.
        self.OptimizedAssembleUpdate()

        # Calculate the displacement for next time step (n+1).
        A1 = (dt**2) * self.f

        Ku = np.zeros((self.sparseInfo.ndof, self.nSmp))
        MultiplyByVector(self.sparseInfo.indptr, self.sparseInfo.indices,
                         (dt**2)*self.K, self.u, Ku)

        A2 = 2.0*self.LM * self.u - Ku
        A3 = (0.5*dt*self.LC - self.LM) * self.up
        RHS = A1 + A2 + A3
        u = np.divide(RHS, self.LHS, out=np.zeros_like(RHS), where=self.LHS!=0)

        # Synchronize u.
        self.SyncCommNodes(u)
        self.ApplyBoundaryCondition(u)

        # Update the displacement solved.
        self.up = self.u
        self.u = u

        # Barrier everyone!
        self.comm.Barrier()

    def OptimizedAssembleUpdate(self):
        """ Optimized assemble update. """

        coefs = np.array([self.appTraction])

        if self.updateCounter >= self.updateInterval:
            # Update stiffness matrix K and forces f if it's a new update interval.

            self.mesh.UpdateCoordinates(self.u)

            # Reset the K and f to zeros, this is a more efficient way.
            self.K = np.zeros((self.sparseInfo.length, self.nSmp, self.Dof, self.Dof))
            self.f = np.zeros((self.sparseInfo.ndof, self.nSmp))
            # self.df = np.zeros((self.sparseInfo.ndof, self.nSmp))

            OptimizedSolidAssembleUpdate(self.mesh.updateNodes, self.mesh.elementNodeIds, coefs, self.D, self.coefK,
                                         self.sparseInfo.indptr, self.sparseInfo.indices, self.K, self.f)

            # OptimizedSolidAssembleUpdateWithDampingForce(self.mesh.updateNodes, self.mesh.elementNodeIds,
            #                                              coefs, self.D, self.coefK,
            #                                              self.u/self.dt/self.dt, self.mass,
            #                                              self.sparseInfo.indptr, self.sparseInfo.indices, self.K, self.f, self.df)
            # self.f = self.cf*self.appTraction + self.df

            self.updateCounter = 1
        else:
            # # Only update the forces f if it's in the update interval.
            # self.f = np.zeros((self.sparseInfo.ndof, self.nSmp))
            # OptimizedSolidAssembleUpdatef(self.mesh.updateNodes, self.mesh.elementNodeIds, coefs, self.f)

            self.updateCounter += 1


    def ApplyBoundaryCondition(self, quant): # TODO:: Change to according to configuration.
        quant[self.bdyDofs,:] = self.mesh.bdyU


    def SyncCommNodes(self, quant, dof=None, func=None):
        """ Synchronize the quantity fo common nodes,
            here, quantity is vector-like.
        """

        if self.size == 1:
            return

        if dof is None:
            dof = self.Dof

        totalCommDofs = self.totalCommDofs
        commDofs = self.commDofs
        commQuant = quant[commDofs]

        totalQuant = np.zeros((len(totalCommDofs), self.nSmp))
        if self.rank == 0:

            # Add on self's (root processor's) quantity.
            indices = np.where(np.isin(totalCommDofs, commDofs))[0]
            totalQuant[indices] += commQuant

            quantIdBuf = np.zeros(len(totalCommDofs), dtype=np.int64)
            quantBuf = np.zeros(len(totalCommDofs)*self.nSmp)
            recvInfo = MPI.Status()
            for i in range(1, self.size):
                self.comm.Recv(quantIdBuf, MPI.ANY_SOURCE, TAG_COMM_DOF, recvInfo)
                recvLen = recvInfo.Get_count(MPI.INT64_T)
                recvSource = recvInfo.Get_source()
                # Receive the quantity.
                self.comm.Recv(quantBuf, recvSource, TAG_COMM_DOF_VALUE, recvInfo)
                # TODO:: make sure the quant received length is consistent with quantIds'.

                # Add the quantity received to the totalQuant.
                indices = np.where(np.isin(totalCommDofs, quantIdBuf[:recvLen]))[0]
                totalQuant[indices] += quantBuf[:recvLen*self.nSmp].reshape(recvLen, self.nSmp)

            if func is not None:
                func(totalQuant)

        else:

            self.comm.Send(commDofs, 0, TAG_COMM_DOF)
            self.comm.Send(commQuant.reshape(len(commDofs)*self.nSmp), 0, TAG_COMM_DOF_VALUE)

            # totalQuant = np.empty(len(totalCommDofs))

        # Get the collected total quantities by broadcast.
        self.comm.Bcast(totalQuant, root=0)
        # Update the original quantity.
        indices = np.where(np.isin(totalCommDofs, commDofs))[0]
        quant[commDofs] = totalQuant[indices]


    def Save(self, filename, counter):
        # Prepare/Union the displacement.
        self.UnionDisplacement()

        # Prepare stress.
        stress = self.OptimizedCalculateStress()
        self.UnionStress(stress)

        if self.rank == 0:
            self.mesh.Save(filename, counter,
                           self.u.transpose().reshape(self.nSmp, self.mesh.nNodes, self.Dof),
                           self.glbStresses)
        # Barrier everyone!
        self.comm.Barrier()

    def UnionDisplacement(self):

        if self.size == 1:
            return

        if self.rank == 0:

            uBuf = np.zeros((self.sparseInfo.ndof, self.nSmp))
            for i in range(1, self.size):
                self.comm.Recv(uBuf, MPI.ANY_SOURCE, TAG_DISPLACEMENT)
                # Flag the nodes uBuf acctually contains.
                self.u[uBuf!=0] = uBuf[uBuf!=0]

        else:
            self.comm.Send(self.u, 0, TAG_DISPLACEMENT)

    def OptimizedCalculateStress(self):

        stress = np.empty((self.mesh.nElements, self.nSmp, 5))
        OptimizedCalculateStress(self.mesh.updateNodes,
                                 self.mesh.elementNodeIds,
                                 self.D, np.mean(self.elmGE, axis=2),
                                 self.u, stress)

        return stress

    def UnionStress(self, stress):

        if self.size == 1:
            self.glbStresses = stress
            return

        if self.rank == 0:

            self.glbStresses = np.zeros((self.mesh.gnElements, self.nSmp, 5))
            self.glbStresses[self.mesh.partition==0, :, :] = stress

            bufSize = int(self.mesh.gnElements / self.size * 1.2)
            stressesBuf = np.empty((bufSize, self.nSmp, 5))

            recvInfo = MPI.Status()
            for i in range(1, self.size):
                # Receive the stresses from each processor.
                self.comm.Recv(stressesBuf, i, TAG_STRESSES, recvInfo) # MPI.ANY_SOURCE
                recvLen = recvInfo.Get_count(MPI.DOUBLE)
                # p = recvInfo.Get_source()
                # Assign.
                self.glbStresses[self.mesh.partition==i, :, :] = stressesBuf[:int(recvLen/self.nSmp/5), :, :]
        else:

            self.comm.Send(stress, 0, TAG_STRESSES)


