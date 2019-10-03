import numpy as np
import scipy.sparse as sparse
#from scipy.sparse import csr_matrix
from ctypes import POINTER,c_void_p,c_int,c_char,c_double,byref,cdll
from timeit import default_timer as timer

class MKL:

    def __init__(self):

        self.mkl = cdll.LoadLibrary("libmkl_rt.so")
        self.timeConsuming = 0.0
        self.timeCore = 0.0
        self.counting = 0

    def SpMV(self, A, x, n):

        start = timer()

        # mkl = cdll.LoadLibrary("libmkl_rt.so")
        mkl = self.mkl

        SpMV = mkl.mkl_cspblas_dcsrgemv
        # Dissecting the "cspblas_dcsrgemv" name:
        # "c" - for "c-blas" like interface (as opposed to fortran)
        #    Also means expects sparse arrays to use 0-based indexing, which python does
        # "sp"  for sparse
        # "d"   for double-precision
        # "csr" for compressed row format
        # "ge"  for "general", e.g., the matrix has no special structure such as symmetry
        # "mv"  for "matrix-vector" multiply

        # The data of the matrix
        data    = A.data.ctypes.data_as(POINTER(c_double))
        indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
        indices = A.indices.ctypes.data_as(POINTER(c_int))

        # Allocate output, using same conventions as input
        y = np.empty(n, dtype=np.double)

        # # Check input
        # if x.dtype.type is not np.double:
        #     x = x.astype(np.double, copy=True)

        np_x = x.ctypes.data_as(POINTER(c_double))
        np_y = y.ctypes.data_as(POINTER(c_double))

        subend = timer()

        # now call MKL. This returns the answer in np_y, which links to y
        SpMV(byref(c_char(b"N")), byref(c_int(n)), data, indptr, indices, np_x, np_y)

        end = timer()
        self.timeConsuming += end - start
        self.timeCore += end - subend
        self.counting += 1

        return y

    def SpMV_Raw(self, A, indptr, indices, x, n):

        findptr = [0]
        findices = []
        flatLHS = []
        # Extend the sparse structure info, indptr and indices.
        for iNode in range(int(n/3)):
            extDofs = MKL.GenerateDof(indices[indptr[iNode]:indptr[iNode+1]], 3)
            for jDof in range(3):
                findices.extend(extDofs)
                findptr.extend([len(findices)])
                # Extend each submatrix.
                flatLHS.extend(MKL.Flatten(A[indptr[iNode]:indptr[iNode+1]], jDof))

        A = sparse.csr_matrix((flatLHS, findices, findptr), shape=(n,n), dtype=np.float64)

        return self.SpMV(A, x, n)

    def SpMV_Smp(self, A, indptr, indices, x, n, nSmp):

        Ku = np.zeros((n, nSmp))

        for i in range(nSmp):

            Ku[:,i] = self.SpMV_Raw(A[:,i,:,:], indptr, indices, x[:,i], n)

        return Ku

    def SpMV_Smp1(self, A, indptr, indices, x, n, nSmp):

        Ku = np.zeros((n, nSmp))

        for i in range(nSmp):

            Ku[:,i] = self.SpMV_Raw(A[:,i,:,:], indptr, indices, x, n)

        return Ku

    @staticmethod
    def Flatten(M, jDof):
        flatM = np.array([subM[jDof] for subM in M])
        return flatM.ravel()

    @staticmethod
    def GenerateDof(nodeIds, dof):
        # nodeIds = np.array(nodeIds)

        # tDofs = dof * len(nodeIds)
        # indices = np.arange(0, tDofs, dof)

        # dofIndices = np.zeros(tDofs, dtype=np.int64)
        # for i in range(dof):
        #     dofIndices[indices+i] = dof * nodeIds + i

        dofIndices = []
        dofArray = np.arange(dof)
        for inode,node in enumerate(nodeIds):
            dofIndices.extend(dof*node + dofArray)

        return np.array(dofIndices, dtype=int)
