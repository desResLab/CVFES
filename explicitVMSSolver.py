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
from sklearn.linear_model import LinearRegression
from mpi4py import MPI

from physicsSolver import *
from optimizedExplicitVMSAssemble import OptimizedExplicitVMSAssemble
from optimizedExplicitVMSAssemble import OptimizedExplicitVMSInitialAssemble


# Parameters for the explicit solver.
c1 = 4.0
c2 = 2.0
# c = 5.0*(11.7**2.0)
c = 5.0*11.7

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

        self.sdu = np.zeros((self.mesh.nElements, 4, 3)) # sub-scale velocity
        self.nsdu = np.zeros_like(self.sdu) # sdu at next time step

        # Prepare the parameters gonna used.
        # Diameters of inscribed sphere of tetrohedron
        self.mesh.calcInscribeDiameters()
        # self.mesh.calcOutletNeighbors()
        self.mesh.calcOutletNeighbors(c*self.dt)
        # Initialize the external_force
        # TODO:: Update when f is a real function of time and space !!!!!!!
        self.f = self.mesh.f * np.ones((self.mesh.nNodes, 3))
        # Initialize shape functions ...
        self.InitializeParameters()
        self.InitializeSolver()
        # Initialize the boundary conditions
        # self.ApplyDirichletBCs(0.0)
        self.ApplyDirichletBCsWithRamp(0.0)

        # self.mesh.DebugReadsInletVelocity()
        # self.DebugApplyDirichletBCs(0.0)

        self.odu = np.copy(self.du)
        self.op = np.copy(self.p)
        # self.om2du = self.om1du = self.odu
        # self.om2p = self.om1p = self.op

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
        self.volumes = np.zeros(nElms, dtype=float)
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
        self.coefs[4] = c**2.0
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
        # self.om2du = self.om1du
        # self.om1du = self.odu
        self.odu = self.du
        # self.om2p = self.om1p
        # self.om1p = self.op
        self.op = self.p

        self.sdu = self.nsdu
        self.nsdu = np.zeros_like(self.sdu)

        # # Solve the linear system to get velocity and pressure
        # self.res, exitCode = gmres(self.spLHS, self.RHS-dt*self.R, x0=self.res.ravel(), M=self.M)
        # # print('Linear system solver at time step {}, converge {}'.format(t, exitCode))
        # # print(np.allclose(self.spLHS.dot(self.res), self.RHS-dt*self.R))

        # Use lumped mass
        self.res = - dt*self.R/self.lumpLHS

        self.res = self.res.reshape((nNodes, dof))
        self.du = self.du + self.res[:,:3]
        self.p = self.p + self.res[:,-1]

        # Apply the Dirichlet boundary conditions.
        # self.ApplyDirichletBCs(t+dt)
        self.ApplyDirichletBCsWithRamp(t+dt)
        # self.DebugApplyDirichletBCs(t+dt)
        # print('Executing here!')

        self.CorrectOutletBCs()


    def ApplyDirichletBCs(self, t):
        # Update the inlet velocity first.
        self.mesh.updateInletVelocity(t)
        # Combine the boundary condition at the start of each time step.
        for inlet in self.mesh.faces['inlet']:
            self.du[inlet.appNodes] = inlet.inletVelocity.reshape((len(inlet.appNodes), 3))

        # dofs = self.GenerateDofs(self.mesh.wall, 3)
        self.du[self.mesh.wall] = 0.0

        # Only for debugging
        # self.p[self.mesh.outlet] = 0.0
        outlet = np.array([ol.glbNodeIds for ol in self.mesh.faces['outlet']]).ravel()
        # self.p[outlet] = 0.0
 
        # self.CorrectOutletBCs()


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
        # self.p[self.mesh.outlet] = 0.0
        # outlet = np.array([ol.glbNodeIds for ol in self.mesh.faces['outlet']]).ravel()
        # self.p[outlet] = 0.0

        # self.CorrectOutletBCs()


    def DebugApplyDirichletBCs(self, t):

        for inlet in self.mesh.faces['inlet']:
            self.du[inlet.appNodes] = self.mesh.dbgInletVelocity[inlet.appNodes]

        # if t > self.constant_T:
        #     for outlet in self.mesh.faces['outlet']:
        #         r = np.sqrt(self.mesh.nodes[outlet.appNodes,0]**2 + self.mesh.nodes[outlet.appNodes,1]**2)
        #         self.du[outlet.appNodes,2] = 11.0 - 11.0/4.0*(r**2)
        # else:
        #     for outlet in self.mesh.faces['outlet']:
        #         r = np.sqrt(self.mesh.nodes[outlet.appNodes,0]**2 + self.mesh.nodes[outlet.appNodes,1]**2)
        #         a = b = 0.5 * (11.0 - 11.0/4.0*(r**2))
        #         n = math.pi/self.constant_T
        #         self.du[outlet.appNodes,2] = a - b*math.cos(n*t)

        self.du[self.mesh.wall] = 0.0
        # # Only for debugging
        # self.p[self.mesh.outlet] = 0.0


    # def CorrectOutletBCs(self):

    #     nodes = self.mesh.nodes
    #     neighbors = self.mesh.outletNeighbors
        
    #     for i,iOutlet in enumerate(self.mesh.outlet):
    #         # res = np.concatenate((self.du[neighbors[i]], self.p[neighbors[i],np.newaxis]), axis=1)
    #         # reg = LinearRegression().fit(nodes[neighbors[i]], res)
    #         reg = LinearRegression().fit(nodes[neighbors[i]], self.du[neighbors[i],-1])
    #         # print('Regression score: {}'.format(reg.score(nodes[neighbors[i]], res)))
    #         if math.isnan(reg.score(nodes[neighbors[i]], self.du[neighbors[i],-1])):
    #             print('Regression failed')
    #             sys.exit(-1)

    #         correctV = reg.predict(np.array([nodes[iOutlet]]))[0]

    #         # if np.isnan(correctV).any():
    #         #     print('Result is nan for {}'.format(iOutlet))
    #         #     continue
            
    #         # self.du[iOutlet,:] = correctV[:3]
    #         # self.p[iOutlet] = correctV[-1]
    #         self.du[iOutlet,-1] = correctV

    # def CorrectOutletBCs(self):

    #     nodes = self.mesh.nodes
    #     neighbors = self.mesh.outletNeighbors
        
    #     for i,iOutlet in enumerate(self.mesh.outlet):

    #         self.du[iOutlet,-1] = np.mean(self.du[neighbors[i],-1])
    #         # self.p[iOutlet] = np.mean(self.p[neighbors[i]])

    # def CorrectOutletBCs(self):
    #     nodes = self.mesh.nodes
    #     neighbors = self.mesh.outletNeighbors

    #     dt = self.dt
    #     du = self.du
    #     odu = self.odu
    #     oodu = self.oodu

    #     for i,iOutlet in enumerate(self.mesh.outlet):
    #         dx = np.amin(self.mesh.inscribeDiameters)
    #         ldx = nodes[iOutlet,-1] - np.mean(nodes[neighbors[i],-1])

    #         phiT = (du[iOutlet,-1] - odu[iOutlet,-1]) / dt
    #         phiX = (du[iOutlet,-1] - np.mean(du[neighbors[i],-1])) / ldx

    #         phi = -phiT/phiX
    #         dtddx = dt/dx
    #         if phi > dx/dt:
    #             # Cphi = dx/dt
    #             du[iOutlet,-1] = np.mean(odu[neighbors[i],-1])
    #         elif phi > 0:
    #             # Cphi = phi
    #             du[iOutlet,-1] = (1.0-dtddx*phi)/(1.0+dtddx*phi)*oodu[iOutlet,-1]+2.0*dtddx*phi/(1+dtddx*phi)*np.mean(odu[neighbors[i],-1])
    #         else:
    #             du[iOutlet,-1] = oodu[iOutlet,-1]

    def CorrectOutletBCs(self):
        elmNIds = self.mesh.elementNodeIds
        du = self.du
        odu = self.odu
        # p = self.p
        # oop = self.oop
        
        for outletFace in self.mesh.faces['outlet']:
            for i,iOutlet in enumerate(outletFace.appNodes):
                du[iOutlet,-1] = np.dot(outletFace.neighborsNs[i], odu[elmNIds[outletFace.neighbors[i]]])[-1]
                # du[iOutlet,0:2] = 0.0
                # p[iOutlet] = np.dot(outletFace.neighborsNs[i], op[elmNIds[outletFace.neighbors[i]]])
                # p[iOutlet] = oop[iOutlet]

    # def CorrectOutletBCs(self):
    #     """ tau+1: du; tau : odu; tau-1 : om1du; tau-2 : om2du """
    #     elmNIds = self.mesh.elementNodeIds
    #     du = self.du
    #     odu = self.odu
    #     om1du = self.om1du
    #     om2du = self.om2du
        
    #     # p = self.p
    #     # op = self.op
    #     # om1p = self.om1p
    #     # om2p = self.om2p

    #     dt = self.dt
    #     dx = c*dt
    #     dtddx = dt/dx
        
    #     for outletFace in self.mesh.faces['outlet']:
    #         for i,iOutlet in enumerate(outletFace.appNodes):
    #             # Decide the C_phi first
    #             neiTau = np.dot(outletFace.neighborsNs[i], odu[elmNIds[outletFace.neighbors[i]]])[-1]
    #             neiTauM2 = np.dot(outletFace.neighborsNs[i], om2du[elmNIds[outletFace.neighbors[i]]])[-1]
    #             neineiTauM1 = np.dot(outletFace.neineighborsNs[i], om1du[elmNIds[outletFace.neineighbors[i]]])[-1]
                
    #             Cphi = -(neiTau-neiTauM2)*dx/((neiTau+neiTauM2)/2.0-neineiTauM1)/(2.0*dt)
    #             if Cphi > c:
    #                 # Cphi = C
    #                 du[iOutlet,-1] = neiTau
    #             elif Cphi > 0.0:
    #                 du[iOutlet,-1] = ((1.0-dtddx*Cphi)*om1du[iOutlet,-1]+2*dtddx*Cphi*neiTau)/(1.0+dtddx*Cphi)
    #             else:
    #                 du[iOutlet,-1] = om1du[iOutlet,-1]
    #             du[iOutlet,0:2] = 0.0

    #             # # Decide the C_phi first
    #             # neiTau = np.dot(outletFace.neighborsNs[i], op[elmNIds[outletFace.neighbors[i]]])
    #             # neiTauM2 = np.dot(outletFace.neighborsNs[i], om2p[elmNIds[outletFace.neighbors[i]]])
    #             # neineiTauM1 = np.dot(outletFace.neineighborsNs[i], om1p[elmNIds[outletFace.neineighbors[i]]])
                
    #             # Cphi = -(neiTau-neiTauM2)*dx/((neiTau+neiTauM2)/2.0-neineiTauM1)/(2.0*dt)
    #             # if Cphi > c:
    #             #     # Cphi = C
    #             #     p[iOutlet] = neiTau
    #             # elif Cphi > 0.0:
    #             #     p[iOutlet] = ((1.0-dtddx*Cphi)*om1p[iOutlet]+2*dtddx*Cphi*neiTau)/(1.0+dtddx*Cphi)
    #             # else:
    #             #     p[iOutlet] = om1p[iOutlet]


    def Save(self, filename, counter):
        # self.mesh.Save(filename, counter, self.du.reshape(self.mesh.nNodes, 3), self.p, 'velocity')

        # res = self.R.reshape((self.mesh.nNodes, self.Dof))
        # resDu = res[:,:3].ravel()
        # resP = res[:,-1].ravel()

        # vals = [self.du, self.p, resDu.reshape(self.mesh.nNodes, 3), resP,
        #         self.mRT1.reshape(self.mesh.nNodes, 3), self.mRT2.reshape(self.mesh.nNodes, 3),
        #         self.mRT3.reshape(self.mesh.nNodes, 3), self.mRT4.reshape(self.mesh.nNodes, 3),
        #         self.mRT5.reshape(self.mesh.nNodes, 3), self.pRT1, self.pRT2]
        # names = ['velocity', 'pressure', 'momentum_res', 'pressure_res', 'momentum_res_term1',
        #          'momentum_res_term2', 'momentum_res_term3', 'momentum_res_term4', 'momentum_res_term5',
        #          'pressure_res_term1', 'pressure_res_term2']
        # ptData = np.ones(11, dtype=bool)

        vals = [self.du, self.p]
        names = ['velocity', 'pressure']
        ptData = np.ones(2, dtype=bool)
        
        self.mesh.DebugSave(filename, counter, vals, names, ptData)


class ExplicitVMSSolidSolver(PhysicsSolver):
    
    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)
        