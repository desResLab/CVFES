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
# from shape import *
from sparse import *
from physicsSolver import *
# from gaussianQuadrature import *
from math import floor
from math import cos, pi, sqrt
from timeit import default_timer as timer
# TODO:: Debugging!!!!!!!!!!!!!!!!!!!!!

# from assemble import Assemble, TestAssemble
from optimizedFluidAssemble import OptimizedFluidAssemble, OptimizedFluidBoundaryAssemble


__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


""" Generalized-a method
"""
class GeneralizedAlphaSolver(PhysicsSolver):

    def __init__(self, comm, mesh, config):

        PhysicsSolver.__init__(self, comm, mesh, config)

        # Set the tolerance used to decide where stop calculating.
        self.tolerance = config.tolerance

        # Solver configuration setup.
        self.imax = config.imax
        self.ci = config.ci

        # Calculate the prameters gonna used
        self.rho_infinity = config.rho_infinity
        self.alpha_f = 1.0 / (1.0+self.rho_infinity)
        self.alpha_m = (3.0-self.rho_infinity) / (2.0+2.0*self.rho_infinity)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f


    def Solve(self, t, dt):

        # Initialize boundary condition.
        self.InitializeBC()

        # Predict the start value of current time step.
        self.Predict()

        # Newton-Raphson loop to approximate.
        check = False
        for i in range(self.imax):

            # Initialize the value of current loop of current time step.
            self.Initialize()

            # Assemble the RHS and LHS of the linear system.
            # self.Assemble()
            self.OptimizedAssemble()
            # self.ComparedAssemble()

            # # Decide if it's been converged.
            # # Put it here to reuse the residual assembling, need to optimize actually.
            # if self.Check():
            #     break

            # Solve the linear system assembled.
            self.SolveLinearSystem()

            # Do the correction.
            self.Correct()

            check = self.Check()
            if check:
                break

        if not check:
            print('Solution does not converge at time step {}'.format(t))


    def InitializeBC(self):
        pass

    def Predict(self):
        pass

    def Initialize(self):
        pass

    # def Assemble(self):
    #     pass

    def OptimizedAssemble(self):
        pass

    # def ComparedAssemble(self):
    #     pass

    def SolveLinearSystem(self):
        pass

    def Correct(self):
        pass

    def Check(self):
        return True


class GeneralizedAlphaFluidSolver(GeneralizedAlphaSolver):

    Dof = 4 # 3 fo velocity (du) and 1 of pressure

    def __init__(self, comm, mesh, config):
        GeneralizedAlphaSolver.__init__(self, comm, mesh, config)

        # Initialize the context.
        self.ddu = mesh.iniDDu # acceleration
        self.du = mesh.iniDu # velocity
        self.p = mesh.iniP # pressure

        # Construct the sparse matrix structure.
        self.sparseInfo = SparseInfo(mesh, self.Dof)
        self.LHS = self.sparseInfo.New()
        self.RHS = np.zeros((self.mesh.nNodes, self.Dof))

        # Initialize the parameters during solving.
        self.InitializeParameters()

        # Debugging ...
        print('Debug: {} elements {} nodes\n'.format(self.mesh.nElements, self.mesh.nNodes))
        print('Debug: size of pressure {}\n'.format(self.p.shape))

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

        # Parameters for Triangle boundary
        self.triW = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        self.triLN = np.array([[0.5, 0.5, 0.0],
                               [0.5, 0.0, 0.5],
                               [0.0, 0.5, 0.5]])

        self.coefs = np.array([self.alpha_m, self.alpha_f, self.gamma,
                               self.dt, self.mesh.density, self.mesh.dviscosity, self.ci])

        # Initialize the external_force
        # TODO:: Update when f is a real function of time and space !!!!!!!
        self.f = self.mesh.f * np.ones((self.mesh.nNodes, 3))

    # def RefreshContext(self, physicsSolver):
    #     # TODO:: only update the boundary!
    #     self.du[:] = physicsSolver.du

    def InitializeBC(self):
        # Combine the boundary condition at the start of each time step.
        for inlet in self.mesh.faces['inlet']:
            dofs = self.sparseInfo.GenerateDofs(inlet.appNodes, 3)
            self.du[dofs] = inlet.inletVelocity

        dofs = self.sparseInfo.GenerateDofs(self.mesh.wall, 3)
        self.du[dofs] = 0.0


    def Predict(self):
        # Reset the previous context to current one to start a new loop.
        # The xP represent previous value.
        self.dduP = self.ddu
        self.duP = self.du
        self.pP = self.p

        # predict ddu using parameter gamma.
        self.ddu = (self.gamma-1.0)/self.gamma * self.dduP
        # u dose not change.
        self.du = self.duP
        # p does not change.
        self.p = self.pP


    def Initialize(self):

        self.interDDu = (1-self.alpha_m)*self.dduP + self.alpha_m*self.ddu
        self.interDu = (1-self.alpha_f)*self.duP + self.alpha_f*self.du


    def OptimizedAssemble(self):

        elements = self.mesh.elementNodeIds
        nNodes = self.mesh.nNodes
        nodes = self.mesh.nodes

        interDDu = self.interDDu.reshape(nNodes, 3)
        interDu = self.interDu.reshape(nNodes, 3)
        interP = self.p

        # External forces.
        f = self.f

        # print "OptimizedAssemble", interDu[1858]
        # print "OptimizedAssemble", interP[1858]

        self.LHS[:,:,:] = 0.0
        self.RHS[:,:] = 0.0

        OptimizedFluidAssemble(nodes, elements, interDDu, interDu, interP, f,
                               self.coefs, self.lN, self.lDN, self.w,
                               self.sparseInfo.indptr, self.sparseInfo.indices,
                               self.LHS, self.RHS)

        # nodeA = 2323
        # indptr = self.sparseInfo.indptr
        # indices = self.sparseInfo.indices
        # ind = indptr[nodeA] + np.where(indices[indptr[nodeA]:indptr[nodeA+1]] == nodeA)[0][0]
        # print ind
        # print self.LHS[ind]
        # print self.RHS[nodeA]

        for outlet in self.mesh.faces['outlet']:
            OptimizedFluidBoundaryAssemble(nodes, outlet.glbNodeIds, outlet.elementNodeIds,
                                           outlet.elementAreas, outlet.ouletH,
                                           self.triLN, self.triW, self.RHS)


        # Apply boundary condition, e.g. set the increment at boundary to be zero.
        self.sparseInfo.ApplyCondition(self.LHS, self.RHS, self.mesh.inlet, 0.0, dof=[0,1,2])
        self.sparseInfo.ApplyCondition(self.LHS, self.RHS, self.mesh.wall, 0.0, dof=[0,1,2])

        # nodeA = 1858
        # indptr = self.sparseInfo.indptr
        # indices = self.sparseInfo.indices
        # ind = indptr[nodeA] + np.where(indices[indptr[nodeA]:indptr[nodeA+1]] == nodeA)[0][0]
        # print ind
        # print self.LHS[ind]
        # print self.RHS[nodeA]

        # print self.LHS[2323]
        # print self.RHS[2323]


    def SolveLinearSystem(self):
        sstart = timer()
        # Only deals with one processor here!
        # TODO:: linear system solver on multiple processors and GPUs!!!!!!!!!!!!!
        up = self.sparseInfo.Solve(self.LHS, -self.RHS)
        udof = np.arange(self.sparseInfo.ndof).reshape(self.mesh.nNodes, self.Dof)[:,:3].reshape(self.mesh.ndof)
        self.deltaDDu = up[udof]
        # print "SolveLinearSystem", self.deltaDDu[1858*3:1858*3+3]
        pdof = np.arange(self.sparseInfo.ndof).reshape(self.mesh.nNodes, self.Dof)[:,-1].reshape(self.mesh.nNodes)
        self.deltaP = up[pdof]
        # print "SolveLinearSystem", self.deltaP[1858]

        send = timer()
        print('Solve linear system, time: %.1f ms' % ((send - sstart) * 1000.0))


    def Correct(self):
        self.ddu = self.ddu + self.deltaDDu
        self.du = self.du + self.gamma * self.dt * self.deltaDDu
        self.p = self.p + self.deltaP

        # print "Correct", self.du[1858*3:1858*3+3]

    def Check(self):
        residual = np.linalg.norm(self.RHS)
        # print residual
        return residual < self.tolerance

    def Save(self, filename, counter):
        self.mesh.Save(filename, counter, self.du.reshape(self.mesh.nNodes, 3), self.p, 'velocity')


class GeneralizedAlphaSolidSolver(GeneralizedAlphaSolver):

    def __init__(self, comm, mesh, config):
        GeneralizedAlphaSolver.__init__(self, comm, mesh, config)

        # Initialize the context.
        self.du = mesh.iniDu # velocity
        self.u = mesh.iniU # displacement

        self.beta = 0.25 * (1 + self.alpha_f - self.alpha_m)**2

    def ApplyPressure(self, appTraction):
        pass

    # def RefreshContext(self, physicSolver):
    #     # TODO:: Solid is the boundary of the fluid part, cannot do this assignment directly!
    #     self.ddu = physicSolver.ddu
    #     self.du = physicSolver.du
    #     self.p = physicSolver.p

    # def Predict(self):
    #     # Reset the previous context to current one to start a new loop.
    #     # The xP represent previous value.
    #     self.uP = self.u

    #     # predict d.
    #     self.u = self.uP + self.du * self.dt + (self.gamma*0.5 - self.beta)/(self.gamma - 1) * self.ddu * (self.dt ** 2)

    # def Initialize(self):

    #     self.interU = (1-self.alpha_f)*self.uP + self.alpha_f*self.u
