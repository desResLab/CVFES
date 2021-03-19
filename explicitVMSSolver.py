#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Explicit VMS solver for fluid part.
"""

import numpy as np
from mpi4py import MPI
from physicsSolver import *
from optimizedExplicitVMSAssemble import OptimizedExplicitVMSAssemble
from optimizedExplicitVMSAssemble import OptimizedExplicitVMSInitialAssemble


# Parameters for the explicit solver.
c1 = 4.0
c2 = 2.0


class ExplicitVMSSolver(PhysicsSolver):
    """Explicit VMS method."""
    
    def __init__(self, comm, mesh, config):

        PhysicsSolver.__init__(self, comm, mesh, config)

        self.Dof = 4 # 3 fo velocity (du) and 1 of pressure

        # Initialize the context.
        self.du = mesh.iniDu.reshape((self.mesh.nNodes, 3)) # velocity
        self.p = mesh.iniP # pressure

        # self.odu = np.zeros_like(self.du)
        # self.op = np.zeros_like(self.p)
        
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
        # Initialize the boundary conditions
        self.ApplyDirichletBCs(0.0)
        # Initialize solver
        # self.InitializeSolver()
        self.odu = np.zeros_like(self.du)
        self.op = np.zeros_like(self.p)

    def InitializeParameters(self):
        # Parameters for Tetrahedron
        alpha = 0.58541020
        beta = 0.13819660
        self.w = np.array([0.25, 0.25, 0.25, 0.25])
        self.lN = np.array([[alpha, beta, beta, beta],
                            [beta, alpha, beta, beta],
                            [beta, beta, alpha, beta],
                            [beta, beta, alpha, beta]])
        self.lDN = np.array([[-1.0, 1.0, 0.0, 0.0],
                             [-1.0, 0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0, 1.0]])

        self.coefs = np.array([c1, c2, self.mesh.dviscosity, self.dt, 1.0])

    def InitializeSolver(self):
        # Calculate the dt, dp
        self.LHS = np.zeros(self.mesh.nNodes*self.Dof)
        self.Res = np.zeros(self.mesh.nNodes*self.Dof)

        OptimizedExplicitVMSInitialAssemble(self.mesh.nodes, self.mesh.elementNodeIds,
                                            self.du, self.p, self.f,
                                            self.lN, self.lDN, self.w, self.coefs,
                                            self.LHS, self.Res)
        acc = np.divide(-self.Res, self.LHS, out=np.zeros_like(self.Res), where=self.LHS!=0)
        acc = acc.reshape((self.mesh.nNodes, self.Dof))
        self.odu = self.du - self.dt*acc[:,:3]
        self.op = self.p - self.dt*acc[:,-1]


    def Solve(self, t, dt):

        self.LHS = np.zeros(self.mesh.nNodes*self.Dof)
        self.RHS = np.zeros(self.mesh.nNodes*self.Dof)
        self.Res = np.zeros(self.mesh.nNodes*self.Dof)

        # only for debugging
        self.mRT1 = np.zeros(self.mesh.nNodes*3)
        self.mRT2 = np.zeros(self.mesh.nNodes*3)
        self.mRT3 = np.zeros(self.mesh.nNodes*3)
        self.mRT4 = np.zeros(self.mesh.nNodes*3)
        self.mRT5 = np.zeros(self.mesh.nNodes*3)
        self.mRT6 = np.zeros(self.mesh.nNodes*3)
        self.mRT7 = np.zeros(self.mesh.nNodes*3)

        self.pRT1 = np.zeros(self.mesh.nNodes)
        self.pRT2 = np.zeros(self.mesh.nNodes)


        # Evaluate velocity prediction hdu and pressure prediction hp
        # at time step n+1 using the second order approximation.
        hdu = 1.5*self.du - 0.5*self.odu
        hp = 1.5*self.p - 0.5*self.op

        # Calculate the invEpsilon for artificial incompressible coef.
        ASS = 5.0
        # self.coefs[4] = (ASS*np.amax(np.linalg.norm(self.du, axis=1)))**2
        # self.coefs[4] = 34521.64
        # self.coefs[4] = 3025.0
        self.coefs[4] = (5.0*11.7)**2.0
        print('The invEpsilon = {}'.format(self.coefs[4]))

        # # Assemble the LHS and RHS.
        # OptimizedExplicitVMSAssemble(self.mesh.nodes, self.mesh.elementNodeIds,
        #                              self.du, self.p, hdu, hp, self.sdu, self.nsdu,
        #                              self.f, self.mesh.inscribeDiameters,
        #                              self.lN, self.lDN, self.w, self.coefs,
        #                              self.LHS, self.RHS, self.Res)

        # only for debugging
        OptimizedExplicitVMSAssemble(self.mesh.nodes, self.mesh.elementNodeIds,
                                     self.du, self.p, hdu, hp, self.sdu, self.nsdu, self.sp, self.nsp, 
                                     self.f, self.mesh.inscribeDiameters,
                                     self.lN, self.lDN, self.w, self.coefs,
                                     self.LHS, self.RHS, self.Res, self.mRT1, self.mRT2,
                                     self.mRT3, self.mRT4, self.mRT5, self.mRT6, self.mRT7,
                                     self.pRT1, self.pRT2)


        # Solve
        self.odu = self.du
        self.op = self.p

        self.sdu = self.nsdu
        self.nsdu = np.zeros_like(self.sdu)
        self.sp = self.nsp
        self.nsp = np.zeros_like(self.sp)

        # res = self.RHS / self.LHS
        res = np.divide(self.RHS-dt*self.Res, self.LHS, out=np.zeros_like(self.RHS), where=self.LHS!=0)
        res = res.reshape((self.mesh.nNodes, self.Dof))
        self.du = res[:,:3].ravel().reshape((self.mesh.nNodes, 3))
        self.p = res[:,-1].ravel()

        # Apply the Dirichlet boundary conditions.
        self.ApplyDirichletBCs(t)

    def ApplyDirichletBCs(self, t):
        # Update the inlet velocity first.
        self.mesh.updateInletVelocity(t)
        # Combine the boundary condition at the start of each time step.
        for inlet in self.mesh.faces['inlet']:
            # dofs = self.GenerateDofs(inlet.appNodes, 3)
            self.du[inlet.appNodes] = inlet.inletVelocity.reshape((len(inlet.appNodes), 3))

        # dofs = self.GenerateDofs(self.mesh.wall, 3)
        self.du[self.mesh.wall] = 0.0

        # Only for debugging
        self.p[self.mesh.outlet] = 0.0

    # def GenerateDofs(self, nodes, dof):
    #     baseArray = np.arange(dof)
    #     return np.array([node*dof+baseArray for node in nodes])

    def Save(self, filename, counter):
        # self.mesh.Save(filename, counter, self.du.reshape(self.mesh.nNodes, 3), self.p, 'velocity')

        res = self.Res.reshape((self.mesh.nNodes, self.Dof))
        resDu = res[:,:3].ravel()
        resP = res[:,-1].ravel()

        vals = [self.du.reshape(self.mesh.nNodes, 3), self.p, resDu.reshape(self.mesh.nNodes, 3), resP,
                self.mRT1.reshape(self.mesh.nNodes, 3), self.mRT2.reshape(self.mesh.nNodes, 3),
                self.mRT3.reshape(self.mesh.nNodes, 3), self.mRT4.reshape(self.mesh.nNodes, 3),
                self.mRT5.reshape(self.mesh.nNodes, 3), self.mRT6.reshape(self.mesh.nNodes, 3),
                self.mRT7.reshape(self.mesh.nNodes, 3), self.pRT1, self.pRT2]
        names = ['velocity', 'pressure', 'momentum_res', 'pressure_res', 'momentum_res_term1',
                 'momentum_res_term2', 'momentum_res_term3', 'momentum_res_term4', 'momentum_res_term5',
                 'momentum_res_term6', 'momentum_res_term7', 'pressure_res_term1', 'pressure_res_term2']
        ptData = np.ones(13, dtype=bool)
        
        self.mesh.DebugSave(filename, counter, vals, names, ptData)


class ExplicitVMSSolidSolver(PhysicsSolver):
    
    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)
        