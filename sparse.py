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
from shape import *
from math import floor
from math import cos, pi
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, spilu
from scipy.sparse import csr_matrix, csc_matrix, diags
# from sksparse.cholmod import cholesky
from scipy import io

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


""" Sparse matrix utilities.
"""
class SparseInfo:

    def __init__(self, mesh, dof):
        # Collecting sparse information.
        sparseInfo = [[] for _ in range(mesh.nNodes)]
        for iElm, elm in enumerate(mesh.elements):
            for iNode in elm.nodes:
                sparseInfo[iNode].extend(elm.nodes)
        sparseInfo = np.array(sparseInfo)
        for iNode in range(mesh.nNodes):
            sparseInfo[iNode] = np.unique(sparseInfo[iNode])

        # Generate sparse matrix.
        indptr = [0] #1
        indices = [] #2
        for iNode in range(mesh.nNodes):
            indices.extend(sparseInfo[iNode])
            indptr.extend([len(indices)])

        # Set self attributes.
        self.indptr = np.array(indptr, dtype=int)
        self.indices = np.array(indices, dtype=int)
        self.length = len(self.indices) # Or self.indptr[-1]

        # Remember the degree of freedom.
        self.nNodes = mesh.nNodes
        self.dof = dof
        self.ndof = mesh.nNodes * dof
        self.dofArray = np.arange(dof)

    def New(self):
        # return [np.zeros((self.dof,self.dof)) for _ in range(self.length)]
        return np.zeros((self.length,self.dof,self.dof))

    def Assemble(self, glbM, elmM, rowNodeIds, colNodeIds=None):
        if colNodeIds is None:
            colNodeIds = rowNodeIds

        for i,nodeA in enumerate(rowNodeIds):
            for j,nodeB in enumerate(colNodeIds):
                ind = self.indptr[nodeA] + np.where(self.indices[self.indptr[nodeA]:self.indptr[nodeA+1]] == nodeB)[0][0]
                glbM[ind] += elmM[self.dof*i:self.dof*(i+1), self.dof*j:self.dof*(j+1)]

    def ApplyCondition(self, lhs, rhs, nodeIds, value, dof=None):
        for i, nodeA in enumerate(nodeIds):
            for j in range(self.indptr[nodeA], self.indptr[nodeA+1]):
                lhs[j][dof,:] = 0.0

            ind = self.indptr[nodeA] + np.where(self.indices[self.indptr[nodeA]:self.indptr[nodeA+1]] == nodeA)[0][0]
            lhs[ind][dof,dof] = 1.0

            rhs[nodeA,dof] = value

    def Solve(self, lhs, rhs):
        indptr = [0]
        indices = []
        flatLHS = []
        # Extend the sparse structure info, indptr and indices.
        for iNode in range(self.nNodes):
            extDofs = self.ExpandDofs(self.indices[self.indptr[iNode]:self.indptr[iNode+1]])
            for jDof in range(self.dof):
                indices.extend(extDofs)
                indptr.extend([len(indices)])
                # Extend each submatrix.
                flatLHS.extend(self.Flatten(lhs[self.indptr[iNode]:self.indptr[iNode+1]], jDof))

        A = csr_matrix((flatLHS, indices, indptr), shape=(self.ndof, self.ndof), dtype=np.float64)

        # TODO:: change this to decomposition solver!!!!!
        # P = diags(1.0/A.diagonal(), 0, format="csr")
        # M_x = lambda x: spsolve(P, x)
        P = spilu(A)
        M_x = lambda x: P.solve(x)
        M = LinearOperator((self.ndof, self.ndof), M_x)
        y, info = gmres(A, rhs.reshape(self.ndof), M=M)
        # print('Solving result: {}'.format(info))

        # y = np.zeros(self.ndof)

        # print "Sparse Solve", y[1858*4:1858*4+4]
        return y

    def MultiplyByVector(self, sparseM, vec):
        """ M dot vec """
        res = np.zeros(self.ndof)
        for i in range(self.nNodes):
            region = np.arange(self.indptr[i], self.indptr[i+1])
            extDofs = self.ExpandDofs(self.indices[region])
            for j in range(self.dof):
                res[self.dof*i+j] = np.dot(self.Flatten(sparseM[region], j), vec[extDofs])
        return res

    def Flatten(self, M, jDof):
        flatM = np.array([subM[jDof] for subM in M])
        return flatM.ravel()

    def Lump(self, M):
        """ Lump the sparse matrix to get a vector which each element is the sum of each row. """
        LM = np.zeros(self.ndof)
        for iNode in range(self.nNodes):
            for jDof in range(self.dof):
                LM[self.dof*iNode+jDof] = np.sum(self.Flatten(M[self.indptr[iNode]:self.indptr[iNode+1]], jDof))
        return LM

    def ExpandDofs(self, nodes):
        baseArray = np.arange(self.dof)
        return np.array([node*self.dof+baseArray for node in nodes]).ravel()


    def OperationCounting(self):
        nOpts = 0

        for i in range(self.nNodes):
            nOpts += 18 * (self.indptr[i+1]-self.indptr[i])

        return nOpts

    def WriteToMMFile(self, A, filename):
        findptr = [0]
        findices = []
        flatLHS = []
        # Extend the sparse structure info, indptr and indices.
        for iNode in range(self.nNodes):
            extDofs = self.ExpandDofs(self.indices[self.indptr[iNode]:self.indptr[iNode+1]])
            for jDof in range(3):
                findices.extend(extDofs)
                findptr.extend([len(findices)])
                # Extend each submatrix.
                flatLHS.extend(self.Flatten(A[self.indptr[iNode]:self.indptr[iNode+1]], jDof))

        A = csr_matrix((flatLHS, findices, findptr), shape=(self.ndof, self.ndof), dtype=np.float64)
        io.mmwrite(filename, A)

