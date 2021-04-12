#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Explicit VMS solver for fluid part.
"""

import math
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres, LinearOperator, spilu
from mpi4py import MPI

from physicsSolver import *
from optimizedExplicitVMSAssemble import OptimizedExplicitVMSAssemble
from optimizedExplicitVMSAssemble import OptimizedExplicitVMSInitialAssemble


# Parameters for the explicit solver.
c1 = 4.0
c2 = 2.0

vDof = 3
pDof = 1
dof = 4


class ExplicitVMSSolver(PhysicsSolver):
    """Explicit VMS method."""
    
    def __init__(self, comm, mesh, config):

        PhysicsSolver.__init__(self, comm, mesh, config)

        self.Dof = 4 # 3 fo velocity (du) and 1 of pressure
        self.constant_T = config.constant_T # for ramp

        # Initialize the context.
        self.du = mesh.iniDu.reshape((self.mesh.nNodes, 3)) # velocity
        self.p = mesh.iniP # pressure

        # self.odu = np.zeros_like(self.du)
        # self.op = np.zeros_like(self.p)
        self.odu = np.copy(self.du)
        self.op = np.copy(self.p)

        self.sdu = np.zeros((self.mesh.nElements, 4, 3)) # sub-scale velocity
        self.nsdu = np.zeros_like(self.sdu) # sdu at next time step
        self.sp = np.zeros((self.mesh.nElements, 4))
        self.nsp = np.zeros_like(self.sp)

        # Prepare the parameters gonna used.
        # Diameters of inscribed sphere of tetrohedron
        self.mesh.calcInscribeDiameters()
        # Initialize the external_force
        # TODO:: Update when f is a real function of time and space !!!!!!!
        self.f = self.mesh.f * np.ones((self.mesh.nNodes, 3))
        # Initialize shape functions ...
        self.InitializeParameters()
        self.InitializeSolver()
        # Initialize the boundary conditions
        # self.ApplyDirichletBCs(0.0)
        self.ApplyDirichletBCsWithRamp(0.0)

        # # --- Attach the initial velocity and pressure together
        # self.res = np.empty((self.mesh.nNodes, dof), dtype=float)
        # self.res[:,:3] = self.du
        # self.res[:,-1] = self.p


    def InitializeParameters(self):
        # Parameters for Tetrahedron
        alpha = 0.58541020
        beta = 0.13819660
        self.w = np.array([0.25, 0.25, 0.25, 0.25])
        self.lN = np.array([[alpha, beta, beta, beta],
                            [beta, alpha, beta, beta],
                            [beta, beta, alpha, beta],
                            [beta, beta, beta, alpha]])
        self.lDN = np.array([[-1.0, 1.0, 0.0, 0.0],
                             [-1.0, 0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0, 1.0]])

        self.coefs = np.array([c1, c2, self.mesh.dviscosity, self.dt, 1.0])

    def InitializeSolver(self):
        
        nodes = self.mesh.nodes
        elements = self.mesh.elementNodeIds

        nNodes = nodes.shape[0]
        nElms = elements.shape[0]
        nElmNodes = elements.shape[1]

        w = self.w
        lN = self.lN
        lDN = self.lDN

        # --- Initial assemble: DNs and LHS
        self.DNs = np.empty((nElms, vDof, nElmNodes), dtype=float)
        self.volumes = np.zeros(nElms, dtype=float) # For debugging
        self.LHS = np.zeros((dof*nNodes, dof*nNodes), dtype=float)
        lMs = np.zeros((nElms, vDof*nElmNodes, vDof*nElmNodes), dtype=float)
        self.invLMs = np.zeros_like(lMs)

        OptimizedExplicitVMSInitialAssemble(nodes, elements, w, lN, lDN,
                                            self.DNs, self.volumes, self.LHS, lMs)

        # Sparse matrix
        self.spLHS = csc_matrix(self.LHS)
        myPreconditioner = spilu(self.spLHS)
        M_x = lambda x: myPreconditioner.solve(x)
        self.M = LinearOperator((dof*nNodes, dof*nNodes), M_x)

        # Lumped mass
        self.lumpLHS = np.sum(self.LHS, axis=1)

        # Calculate inverse of local mass matrix for subscales calculation
        for iElm in range(nElms):
            self.invLMs[iElm,:,:] = np.linalg.inv(lMs[iElm])


    def Solve(self, t, dt):

        nodes = self.mesh.nodes
        elements = self.mesh.elementNodeIds

        nNodes = nodes.shape[0]

        self.RHS = np.zeros(nNodes*dof)
        self.R = np.zeros(nNodes*dof) # residual

        # only for debugging
        self.mRT1 = np.zeros(nNodes*vDof)
        self.mRT2 = np.zeros(nNodes*vDof)
        self.mRT3 = np.zeros(nNodes*vDof)
        self.mRT4 = np.zeros(nNodes*vDof)
        self.mRT5 = np.zeros(nNodes*vDof)

        self.pRT1 = np.zeros(nNodes)
        self.pRT2 = np.zeros(nNodes)


        # Evaluate velocity prediction hdu and pressure prediction hp
        # at time step n+1 using the second order approximation.
        hdu = 1.5*self.du - 0.5*self.odu
        hp = 1.5*self.p - 0.5*self.op

        # Calculate the invEpsilon for artificial incompressible coef.
        ASS = 5.0
        self.coefs[4] = (5.0*11.7)**2.0
        # self.coefs[4] = (5.0*11.7*11.7)**2.0
        # print('The invEpsilon = {}'.format(self.coefs[4]))

        # # Assemble the LHS and RHS.
        # OptimizedExplicitVMSAssemble(nodes, elements,
        #                              self.du, self.p, hdu, hp, self.sdu, self.nsdu,
        #                              self.f, self.mesh.inscribeDiameters,
        #                              self.w, self.lN, self.lDN, self.coefs,
        #                              self.RHS, self.R)

        # only for debugging
        OptimizedExplicitVMSAssemble(nodes, elements, self.du, self.p, hdu, hp, self.sdu, self.nsdu,
                                     self.f, self.mesh.inscribeDiameters,
                                     self.w, self.lN, self.DNs, self.volumes, self.invLMs, self.coefs,
                                     self.RHS, self.R,
                                     self.mRT1, self.mRT2, self.mRT3, self.mRT4, self.mRT5,
                                     self.pRT1, self.pRT2)

        # Solve
        self.odu[:,:] = self.du
        self.op[:] = self.p

        self.sdu = self.nsdu
        self.nsdu = np.zeros_like(self.sdu)

        # # Solve the linear system to get velocity and pressure
        # self.res, exitCode = gmres(self.spLHS, self.RHS-dt*self.R, x0=self.res.ravel(), M=self.M)
        # # print('Linear system solver at time step {}, converge {}'.format(t, exitCode))
        # # print(np.allclose(self.spLHS.dot(self.res), self.RHS-dt*self.R))

        # Use lumped mass
        self.res = - dt*self.R/self.lumpLHS

        self.res = self.res.reshape((nNodes, dof))
        self.du[:,:] = self.du + self.res[:,:3]
        self.p[:] = self.p + self.res[:,-1]

        # Apply the Dirichlet boundary conditions.
        # self.ApplyDirichletBCs(t+dt)
        self.ApplyDirichletBCsWithRamp(t+dt)
        # print('Executing here!')


    def ApplyDirichletBCs(self, t):
        # Update the inlet velocity first.
        self.mesh.updateInletVelocity(t)
        # Combine the boundary condition at the start of each time step.
        for inlet in self.mesh.faces['inlet']:
            self.du[inlet.appNodes] = inlet.inletVelocity.reshape((len(inlet.appNodes), 3))

        # dofs = self.GenerateDofs(self.mesh.wall, 3)
        self.du[self.mesh.wall] = 0.0

        # Only for debugging
        self.p[self.mesh.outlet] = 0.0


    def ApplyDirichletBCsWithRamp(self, t):
        # Update the inlet velocity first.
        self.mesh.updateInletVelocity(t)
        # Combine the boundary condition at the start of each time step.
        if t > self.constant_T:
            for inlet in self.mesh.faces['inlet']:
                self.du[inlet.appNodes] = inlet.inletVelocity.reshape((len(inlet.appNodes), 3))
        else:
            for inlet in self.mesh.faces['inlet']:
                a = b = 0.5 * inlet.inletVelocity.reshape((len(inlet.appNodes), 3))
                n = math.pi/self.constant_T
                self.du[inlet.appNodes] = a - b*math.cos(n*t)

        # dofs = self.GenerateDofs(self.mesh.wall, 3)
        self.du[self.mesh.wall] = 0.0

        # Only for debugging
        self.p[self.mesh.outlet] = 0.0


    def Save(self, filename, counter):
        # self.mesh.Save(filename, counter, self.du.reshape(self.mesh.nNodes, 3), self.p, 'velocity')

        res = self.R.reshape((self.mesh.nNodes, self.Dof))
        resDu = res[:,:3].ravel()
        resP = res[:,-1].ravel()

        vals = [self.du.reshape(self.mesh.nNodes, 3), self.p, resDu.reshape(self.mesh.nNodes, 3), resP,
                self.mRT1.reshape(self.mesh.nNodes, 3), self.mRT2.reshape(self.mesh.nNodes, 3),
                self.mRT3.reshape(self.mesh.nNodes, 3), self.mRT4.reshape(self.mesh.nNodes, 3),
                self.mRT5.reshape(self.mesh.nNodes, 3), self.pRT1, self.pRT2]
        names = ['velocity', 'pressure', 'momentum_res', 'pressure_res', 'momentum_res_term1',
                 'momentum_res_term2', 'momentum_res_term3', 'momentum_res_term4', 'momentum_res_term5',
                 'pressure_res_term1', 'pressure_res_term2']
        ptData = np.ones(11, dtype=bool)
        
        self.mesh.DebugSave(filename, counter, vals, names, ptData)


class ExplicitVMSSolidSolver(PhysicsSolver):
    
    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)
        