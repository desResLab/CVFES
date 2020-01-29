# cython: language_level=3, boundscheck=False
# include math method from C libs.
import numpy as np

cimport numpy as np
from libc.math cimport sqrt
from libc.stdio cimport printf
from cython.parallel import prange
cimport cython
cimport openmp

cdef long nPts # number of nodes in shape
cdef long ndim # number of dimensions
cdef long nSmp # number of samples


# (n,m) (m,d)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dot(double[:,::1] A, double[:,::1] B, double[:,::1] C,
              long n, long m, long d):

    cdef long i, j, k

    for i in range(n):
        for j in range(d):
            C[i,j] = 0.0
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j]


# (n,m) (m,1)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dot1(double[:,::1] A, double[::1] B, double[::1] C,
               long n, long m):

    cdef long i, j, k

    for i in range(n):
        C[i] = 0.0
        for j in range(m):
            C[i] += A[i,j]*B[j]


# (n,m) (n,d)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dotT(double[:,::1] A, double[:,::1] B, double[:,::1] C,
               long n, long m, long d):

    cdef long i, j, k

    for i in range(m):
        for j in range(d):
            C[i,j] = 0.0
            for k in range(n):
                C[i,j] += A[k,i]*B[k,j]


# (n,m) (n,1)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dotT1(double[:,::1] A, double[::1] B, double[::1] C,
               long n, long m):

    cdef long i, j, k

    for i in range(m):
        C[i] = 0.0
        for j in range(n):
            C[i] += A[j,i]*B[j]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void CoordinateTransformationUpdate(double[:,:,::1] nodes, long s, long[::1] eNIds,
                                         double[:,::1] edges, double[:,::1] T,
                                         double[:,::1] TT):

    cdef double edgenorm = 0.0

    edges[0,0] = nodes[s,eNIds[2],0] - nodes[s,eNIds[1],0]
    edges[0,1] = nodes[s,eNIds[2],1] - nodes[s,eNIds[1],1]
    edges[0,2] = nodes[s,eNIds[2],2] - nodes[s,eNIds[1],2]
    edges[1,0] = nodes[s,eNIds[0],0] - nodes[s,eNIds[2],0]
    edges[1,1] = nodes[s,eNIds[0],1] - nodes[s,eNIds[2],1]
    edges[1,2] = nodes[s,eNIds[0],2] - nodes[s,eNIds[2],2]

    edgenorm = sqrt(edges[0,0]*edges[0,0] + edges[0,1]*edges[0,1] + edges[0,2]*edges[0,2])
    # edgenorm = 1.0 / edgenorm
    T[0,0] = edges[0,0]/edgenorm
    T[0,1] = edges[0,1]/edgenorm
    T[0,2] = edges[0,2]/edgenorm

    edgenorm = edges[1,0]*T[0,0] + edges[1,1]*T[0,1] + edges[1,2]*T[0,2]
    T[1,0] = edges[1,0] - edgenorm*T[0,0]
    T[1,1] = edges[1,1] - edgenorm*T[0,1]
    T[1,2] = edges[1,2] - edgenorm*T[0,2]
    edgenorm = sqrt(T[1,0]*T[1,0] + T[1,1]*T[1,1] + T[1,2]*T[1,2])
    # edgenorm = 1.0 / edgenorm
    T[1,0] = T[1,0]/edgenorm
    T[1,1] = T[1,1]/edgenorm
    T[1,2] = T[1,2]/edgenorm

    # set the 3rd to be cross product of first two
    T[2,0] = T[0,1]*T[1,2] - T[0,2]*T[1,1] # cx = aybz - azby
    T[2,1] = T[0,2]*T[1,0] - T[0,0]*T[1,2] # cy = azbx - axbz
    T[2,2] = T[0,0]*T[1,1] - T[0,1]*T[1,0] # cz = axby - aybx

    # enlarge the T to be big
    TT[0,0] = TT[3,3] = TT[6,6] = T[0,0]
    TT[0,1] = TT[3,4] = TT[6,7] = T[0,1]
    TT[0,2] = TT[3,5] = TT[6,8] = T[0,2]

    TT[1,0] = TT[4,3] = TT[7,6] = T[1,0]
    TT[1,1] = TT[4,4] = TT[7,7] = T[1,1]
    TT[1,2] = TT[4,5] = TT[7,8] = T[1,2]

    TT[2,0] = TT[5,3] = TT[8,6] = T[2,0]
    TT[2,1] = TT[5,4] = TT[8,7] = T[2,1]
    TT[2,2] = TT[5,5] = TT[8,8] = T[2,2]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedCalculateStress(double[:,:,::1] nodes, long[:,::1] elements,
                             double[:,::1] D, double[:,::1]aveElmGE,
                             double[:,::1] u, double[:,:,:,::1] stress):

    cdef long nElms = elements.shape[0]
    cdef long iElm, iSmp

    nPts = elements.shape[1]
    ndim = nodes.shape[2]
    nSmp = nodes.shape[0]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    cdef double[:,::1] TT = np.zeros((9,9), dtype=np.float)
    cdef double[:,::1] lNodes = np.empty((nPts, 2), dtype=np.float)
    cdef double[:,::1] B = np.zeros((5, 9), dtype=np.float)
    cdef double[::1] gU = np.empty(9, dtype=np.float)
    cdef double[::1] lU = np.empty(9, dtype=np.float)
    cdef double[::1] lStress = np.empty(5, dtype=np.float)
    cdef double[:,::1] stressTensor = np.empty((3,3), dtype=np.float)
    cdef double[:,::1] glbStressTensor = np.empty((3,3), dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    # used for temporary value
    cdef double[:,::1] DB = np.empty((5,9), dtype=np.float)
    cdef double[:,::1] TST = np.empty((3,3), dtype=np.float) # T.transpose dot stressTensor

    cdef double area
    cdef double y23, y31, y12, x32, x13, x21
    cdef double c1, c2, c3, c4
    cdef long i, j, m, n

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for iSmp in range(nSmp):

            for i in range(nPts):
                gU[i*3] = u[eNIds[i]*3,iSmp]
                gU[i*3+1] = u[eNIds[i]*3+1,iSmp]
                gU[i*3+2] = u[eNIds[i]*3+2,iSmp]

            # Get the transformation matrix T.
            CoordinateTransformationUpdate(nodes, iSmp, eNIds, edges, T, TT)

            # Transform the triangular to sheer.
            lNodes[0,0] = nodes[iSmp,eNIds[0],0]*T[0,0] + nodes[iSmp,eNIds[0],1]*T[0,1] + nodes[iSmp,eNIds[0],2]*T[0,2]
            lNodes[0,1] = nodes[iSmp,eNIds[0],0]*T[1,0] + nodes[iSmp,eNIds[0],1]*T[1,1] + nodes[iSmp,eNIds[0],2]*T[1,2]
            lNodes[1,0] = nodes[iSmp,eNIds[1],0]*T[0,0] + nodes[iSmp,eNIds[1],1]*T[0,1] + nodes[iSmp,eNIds[1],2]*T[0,2]
            lNodes[1,1] = nodes[iSmp,eNIds[1],0]*T[1,0] + nodes[iSmp,eNIds[1],1]*T[1,1] + nodes[iSmp,eNIds[1],2]*T[1,2]
            lNodes[2,0] = nodes[iSmp,eNIds[2],0]*T[0,0] + nodes[iSmp,eNIds[2],1]*T[0,1] + nodes[iSmp,eNIds[2],2]*T[0,2]
            lNodes[2,1] = nodes[iSmp,eNIds[2],0]*T[1,0] + nodes[iSmp,eNIds[2],1]*T[1,1] + nodes[iSmp,eNIds[2],2]*T[1,2]

            # Calculate area and matrix B.
            area = ((lNodes[1,0]*lNodes[2,1] - lNodes[2,0]*lNodes[1,1])
                    - (lNodes[0,0]*lNodes[2,1] - lNodes[2,0]*lNodes[0,1])
                    + (lNodes[0,0]*lNodes[1,1] - lNodes[1,0]*lNodes[0,1]))*0.5

            y23 = lNodes[1,1] - lNodes[2,1]
            y31 = lNodes[2,1] - lNodes[0,1]
            y12 = lNodes[0,1] - lNodes[1,1]
            x32 = lNodes[2,0] - lNodes[1,0]
            x13 = lNodes[0,0] - lNodes[2,0]
            x21 = lNodes[1,0] - lNodes[0,0]

            c1 = (2.0*area)
            B[0,0] = B[2,1] = B[3,2] = y23/c1
            B[0,3] = B[2,4] = B[3,5] = y31/c1
            B[0,6] = B[2,7] = B[3,8] = y12/c1
            B[1,1] = B[2,0] = B[4,2] = x32/c1
            B[1,4] = B[2,3] = B[4,5] = x13/c1
            B[1,7] = B[2,6] = B[4,8] = x21/c1

            # Calculate the local u (displacement) vector.
            dot1(TT, gU, lU, 9, 9)

            # Calculate local stress vector.
            dot(D, B, DB, 5, 5, 9)
            dot1(DB, lU, lStress, 5, 9)

            lStress[0] = lStress[0]*aveElmGE[iElm,iSmp]
            lStress[1] = lStress[1]*aveElmGE[iElm,iSmp]
            lStress[2] = lStress[2]*aveElmGE[iElm,iSmp]
            lStress[3] = lStress[3]*aveElmGE[iElm,iSmp]
            lStress[4] = lStress[4]*aveElmGE[iElm,iSmp]

            # Fill up the stress tensor and calculate the final global stress.
            stressTensor[0,0] = lStress[0]
            stressTensor[0,1] = lStress[2]
            stressTensor[0,2] = lStress[3]
            stressTensor[1,0] = lStress[2]
            stressTensor[1,1] = lStress[1]
            stressTensor[1,2] = lStress[4]
            stressTensor[2,0] = lStress[3]
            stressTensor[2,1] = lStress[4]
            stressTensor[2,2] = 0.0

            dotT(T, stressTensor, TST, 3, 3, 3)
            dot(TST, T, glbStressTensor, 3, 3, 3)

            # 'Assemble' back to the big stress matrix.
            for m in range(3):
                for n in range(3):
                    stress[iElm,iSmp,m,n] = glbStressTensor[m,n]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ShearTransform(double[:,:,::1] nodes, long[::1] eNIds,
                         double[:,:,::1] elmCtrlPts, double[:,::1] edges,
                         double[:,::1] T, long iElm, long iSmp):

    cdef double edgenorm = 0.0
    cdef double ctrlVec[3]

    edges[0,0] = nodes[iSmp,eNIds[2],0] - nodes[iSmp,eNIds[1],0]
    edges[0,1] = nodes[iSmp,eNIds[2],1] - nodes[iSmp,eNIds[1],1]
    edges[0,2] = nodes[iSmp,eNIds[2],2] - nodes[iSmp,eNIds[1],2]
    edges[1,0] = nodes[iSmp,eNIds[0],0] - nodes[iSmp,eNIds[2],0]
    edges[1,1] = nodes[iSmp,eNIds[0],1] - nodes[iSmp,eNIds[2],1]
    edges[1,2] = nodes[iSmp,eNIds[0],2] - nodes[iSmp,eNIds[2],2]

    edgenorm = sqrt(edges[0,0]*edges[0,0] + edges[0,1]*edges[0,1] + edges[0,2]*edges[0,2])
    # edgenorm = 1.0 / edgenorm
    edges[0,0] = edges[0,0]/edgenorm
    edges[0,1] = edges[0,1]/edgenorm
    edges[0,2] = edges[0,2]/edgenorm

    edgenorm = edges[1,0]*edges[0,0] + edges[1,1]*edges[0,1] + edges[1,2]*edges[0,2]
    edges[1,0] = edges[1,0] - edgenorm*edges[0,0]
    edges[1,1] = edges[1,1] - edgenorm*edges[0,1]
    edges[1,2] = edges[1,2] - edgenorm*edges[0,2]
    edgenorm = sqrt(edges[1,0]*edges[1,0] + edges[1,1]*edges[1,1] + edges[1,2]*edges[1,2])
    edges[1,0] = edges[1,0]/edgenorm
    edges[1,1] = edges[1,1]/edgenorm
    edges[1,2] = edges[1,2]/edgenorm

    # get norm from cross product of first two
    T[1,0] = edges[0,1]*edges[1,2] - edges[0,2]*edges[1,1] # cx = aybz - azby
    T[1,1] = edges[0,2]*edges[1,0] - edges[0,0]*edges[1,2] # cy = azbx - axbz
    T[1,2] = edges[0,0]*edges[1,1] - edges[0,1]*edges[1,0] # cz = axby - aybx

    # get the 'Z' direction unit vector
    ctrlVec[0] = elmCtrlPts[iElm,0,0] - elmCtrlPts[iElm,1,0]
    ctrlVec[1] = elmCtrlPts[iElm,0,1] - elmCtrlPts[iElm,1,1]
    ctrlVec[2] = elmCtrlPts[iElm,0,2] - elmCtrlPts[iElm,1,2]
    edgenorm = ctrlVec[0]*T[1,0] + ctrlVec[1]*T[1,1] + ctrlVec[2]*T[1,2]
    T[2,0] = ctrlVec[0] - edgenorm*T[1,0]
    T[2,1] = ctrlVec[1] - edgenorm*T[1,1]
    T[2,2] = ctrlVec[2] - edgenorm*T[1,2]
    edgenorm = sqrt(T[2,0]*T[2,0] + T[2,1]*T[2,1] + T[2,2]*T[2,2])
    T[2,0] = T[2,0]/edgenorm
    T[2,1] = T[2,1]/edgenorm
    T[2,2] = T[2,2]/edgenorm

    # get the third component base unit vec by cross product
    T[0,0] = T[1,1]*T[2,2] - T[1,2]*T[2,1]
    T[0,1] = T[1,2]*T[2,0] - T[1,0]*T[2,2]
    T[0,2] = T[1,0]*T[2,1] - T[1,1]*T[2,0]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def TranformStress(double[:,:,::1] nodes, long[:,::1] elements, double[:,:,::1] elmCtrlPts,
                   double[:,:,:,::1] stress, double[:,:,:,::1] tStress):

    cdef long nElms = elements.shape[0]
    cdef long iElm, iSmp

    nPts = elements.shape[1]
    ndim = nodes.shape[2]
    nSmp = nodes.shape[0]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    # tmp variables
    cdef double[:,::1] TS = np.empty((nPts, nPts), dtype=np.float)

    cdef long i, j, m, n

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for iSmp in range(nSmp):

            # Get the transformation matrix T.
            ShearTransform(nodes, eNIds, elmCtrlPts, edges, T, iElm, iSmp)

            # Transform Ss = T*S*T'.
            TS[0,0] = T[0,0]*stress[iElm,iSmp,0,0] + T[0,1]*stress[iElm,iSmp,1,0] + T[0,2]*stress[iElm,iSmp,2,0]
            TS[0,1] = T[0,0]*stress[iElm,iSmp,0,1] + T[0,1]*stress[iElm,iSmp,1,1] + T[0,2]*stress[iElm,iSmp,2,1]
            TS[0,2] = T[0,0]*stress[iElm,iSmp,0,2] + T[0,1]*stress[iElm,iSmp,1,2] + T[0,2]*stress[iElm,iSmp,2,2]
            TS[1,0] = T[1,0]*stress[iElm,iSmp,0,0] + T[1,1]*stress[iElm,iSmp,1,0] + T[1,2]*stress[iElm,iSmp,2,0]
            TS[1,1] = T[1,0]*stress[iElm,iSmp,0,1] + T[1,1]*stress[iElm,iSmp,1,1] + T[1,2]*stress[iElm,iSmp,2,1]
            TS[1,2] = T[1,0]*stress[iElm,iSmp,0,2] + T[1,1]*stress[iElm,iSmp,1,2] + T[1,2]*stress[iElm,iSmp,2,2]
            TS[2,0] = T[2,0]*stress[iElm,iSmp,0,0] + T[2,1]*stress[iElm,iSmp,1,0] + T[2,2]*stress[iElm,iSmp,2,0]
            TS[2,1] = T[2,0]*stress[iElm,iSmp,0,1] + T[2,1]*stress[iElm,iSmp,1,1] + T[2,2]*stress[iElm,iSmp,2,1]
            TS[2,2] = T[2,0]*stress[iElm,iSmp,0,2] + T[2,1]*stress[iElm,iSmp,1,2] + T[2,2]*stress[iElm,iSmp,2,2]

            tStress[iElm,iSmp,0,0] = TS[0,0]*T[0,0] + TS[0,1]*T[0,1] + TS[0,2]*T[0,2]
            tStress[iElm,iSmp,0,1] = TS[0,0]*T[1,0] + TS[0,1]*T[1,1] + TS[0,2]*T[1,2]
            tStress[iElm,iSmp,0,2] = TS[0,0]*T[2,0] + TS[0,1]*T[2,1] + TS[0,2]*T[2,2]
            tStress[iElm,iSmp,1,0] = TS[1,0]*T[0,0] + TS[1,1]*T[0,1] + TS[1,2]*T[0,2]
            tStress[iElm,iSmp,1,1] = TS[1,0]*T[1,0] + TS[1,1]*T[1,1] + TS[1,2]*T[1,2]
            tStress[iElm,iSmp,1,2] = TS[1,0]*T[2,0] + TS[1,1]*T[2,1] + TS[1,2]*T[2,2]
            tStress[iElm,iSmp,2,0] = TS[2,0]*T[0,0] + TS[2,1]*T[0,1] + TS[2,2]*T[0,2]
            tStress[iElm,iSmp,2,1] = TS[2,0]*T[1,0] + TS[2,1]*T[1,1] + TS[2,2]*T[1,2]
            tStress[iElm,iSmp,2,2] = TS[2,0]*T[2,0] + TS[2,1]*T[2,1] + TS[2,2]*T[2,2]
