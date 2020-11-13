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

        self.odu = np.zeros_like(self.du)
        self.op = np.zeros_like(self.p)
        
        self.sdu = np.zeros((self.mesh.nElements, 4, 3)) # sub-scale velocity
        self.nsdu = np.zeros_like(self.sdu) # sdu at next time step

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

    def Solve(self, t, dt):

        self.LHS = np.zeros(self.mesh.nNodes*self.Dof)
        self.RHS = np.zeros(self.mesh.nNodes*self.Dof)
        self.Res = np.zeros(self.mesh.nNodes*self.Dof)

        # Evaluate velocity prediction hdu and pressure prediction hp
        # at time step n+1 using the second order approximation.
        hdu = 1.5*self.du - 0.5*self.odu
        hp = 1.5*self.p - 0.5*self.op

        # Calculate the invEpsilon for artificial incompressible coef.
        # ASS = 5.0
        # self.coefs[4] = (ASS*np.amax(np.linalg.norm(du, axis=1)))**2
        self.coefs[4] = 34521.64

        # Assemble the LHS and RHS.
        OptimizedExplicitVMSAssemble(self.mesh.nodes, self.mesh.elementNodeIds,
                                     self.du, self.p, hdu, hp, self.sdu, self.nsdu,
                                     self.f, self.mesh.inscribeDiameters,
                                     self.lN, self.lDN, self.w, self.coefs,
                                     self.LHS, self.RHS, self.Res)

        # Solve
        self.odu = self.du
        self.op = self.p

        self.sdu = self.nsdu
        self.nsdu = np.zeros_like(self.sdu)

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

    # def GenerateDofs(self, nodes, dof):
    #     baseArray = np.arange(dof)
    #     return np.array([node*dof+baseArray for node in nodes])

    def Save(self, filename, counter):
        self.mesh.Save(filename, counter, self.du.reshape(self.mesh.nNodes, 3), self.p, 'velocity')

        # res = self.Res.reshape((self.mesh.nNodes, self.Dof))
        # resDu = res[:,:3].ravel()
        # resP = res[:,-1].ravel()
        # self.mesh.Save(filename, counter, resDu.reshape(self.mesh.nNodes, 3), resP, 'residual')

