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
cdef void CoordinateTransformation(double[:,::1] nodes, long[::1] eNIds,
                                   double[:,::1] edges, double[:,::1] T,
                                   double[:,::1] TT):

    cdef double edgenorm = 0.0

    edges[0,0] = nodes[eNIds[2],0] - nodes[eNIds[1],0]
    edges[0,1] = nodes[eNIds[2],1] - nodes[eNIds[1],1]
    edges[0,2] = nodes[eNIds[2],2] - nodes[eNIds[1],2]
    edges[1,0] = nodes[eNIds[0],0] - nodes[eNIds[2],0]
    edges[1,1] = nodes[eNIds[0],1] - nodes[eNIds[2],1]
    edges[1,2] = nodes[eNIds[0],2] - nodes[eNIds[2],2]

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
cdef void assembling(long[::1] eNIds, double[:,:,::1] lM,
                     double[:,:,::1] lK, double[::1] lf,
                     long[::1] indptr, long[::1] indices,
                     double[:,:,:,::1] M, double[:,:,:,::1] K, double[::1] f):

    nSmp = lM.shape[0]

    cdef long a, b, s
    cdef long row, col, left, right, ptr
    cdef long il, ir, ib

    for a in range(3):
        row = eNIds[a]
        il = row*3
        ir = a*3
        # Assemble the RHS.
        f[il] += lf[ir]
        f[il+1] += lf[ir+1]
        f[il+2] += lf[ir+2]

        for b in range(3):
            col = eNIds[b]
            ib = b*3
            # Search and assemble.
            left = indptr[row]
            right = indptr[row+1]
            ptr = (left + right) / 2

            while indices[ptr] != col:
                if indices[ptr] > col:
                    right = ptr
                else:
                    left = ptr
                ptr = (left + right) / 2

            for s in range(nSmp):

                M[ptr,s,0,0] += lM[s,ir,ib]
                M[ptr,s,0,1] += lM[s,ir,ib+1]
                M[ptr,s,0,2] += lM[s,ir,ib+2]
                M[ptr,s,1,0] += lM[s,ir+1,ib]
                M[ptr,s,1,1] += lM[s,ir+1,ib+1]
                M[ptr,s,1,2] += lM[s,ir+1,ib+2]
                M[ptr,s,2,0] += lM[s,ir+2,ib]
                M[ptr,s,2,1] += lM[s,ir+2,ib+1]
                M[ptr,s,2,2] += lM[s,ir+2,ib+2]

                K[ptr,s,0,0] += lK[s,ir,ib]
                K[ptr,s,0,1] += lK[s,ir,ib+1]
                K[ptr,s,0,2] += lK[s,ir,ib+2]
                K[ptr,s,1,0] += lK[s,ir+1,ib]
                K[ptr,s,1,1] += lK[s,ir+1,ib+1]
                K[ptr,s,1,2] += lK[s,ir+1,ib+2]
                K[ptr,s,2,0] += lK[s,ir+2,ib]
                K[ptr,s,2,1] += lK[s,ir+2,ib+1]
                K[ptr,s,2,2] += lK[s,ir+2,ib+2]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lumping(long[::1] indptr,
                  double[:,:,:,::1] M, double[:,::1] lumpM):

    nSmp = M.shape[1]

    cdef long nNodes = len(indptr) - 1
    cdef long i, j, ik, jk, s

    for i in range(nNodes):
        for j in range(indptr[i], indptr[i+1]):
            for ik in range(3):
                for jk in range(3):
                    for s in range(nSmp):
                        lumpM[3*i+ik,s] += M[j,s,ik,jk]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedSolidAssemble(double[:,::1] nodes, long[:,::1] elements,
                      double[::1] coefs, double[::1] wt, double[:,:,::1] tM,
                      double[:,::1] D, double[:,::1] coefK, double[:,:,::1] elmGThick,
                      long[::1] indptr, long[::1] indices,
                      double[:,:,:,::1] M, double[:,:,:,::1] K,
                      double[::1] f, double[:,::1] lumpM):

    cdef long nElms = elements.shape[0]
    cdef long iElm, iSmp

    nPts = elements.shape[1]
    ndim = nodes.shape[1]
    nSmp = elmGThick.shape[1]

    cdef double trac = coefs[0]
    cdef long nGps = wt.shape[0]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    cdef double[:,::1] TT = np.zeros((9,9), dtype=np.float)
    cdef double[:,::1] lNodes = np.empty((nPts, 2), dtype=np.float)
    # cdef double[:,::1] D = np.zeros((5, 5), dtype=np.float)
    cdef double[:,::1] B = np.zeros((5, 9), dtype=np.float)
    # traction array used for calc f
    cdef double[::1] tracArr = np.zeros(9, dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    # used for temporary value
    cdef double[:,::1] BTD = np.empty((9,5), dtype=np.float)
    cdef double[:,::1] BTDB = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] TTPM = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] TTPK = np.empty((9,9), dtype=np.float)

    cdef double[:,::1] lM = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] lK = np.empty((9,9), dtype=np.float)
    cdef double[::1] lf = np.zeros(9, dtype=np.float)

    cdef double[:,::1] gM = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] gK = np.empty((9,9), dtype=np.float)
    cdef double[::1] gf = np.zeros(9, dtype=np.float)

    cdef double[:,:,::1] sgM = np.empty((nSmp,9,9), dtype=np.float)
    cdef double[:,:,::1] sgK = np.empty((nSmp,9,9), dtype=np.float)

    cdef double area
    cdef double y23, y31, y12, x32, x13, x21
    cdef double c1, c2, c3, c4
    cdef long i, j, k

    # Calculate the traction vector prepared for f.
    tracArr[2] = tracArr[5] = tracArr[8] = trac/3.0

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        # Get the transformation matrix T.
        CoordinateTransformation(nodes, eNIds, edges, T, TT)

        # Transform the triangular to sheer.
        lNodes[0,0] = nodes[eNIds[0],0]*T[0,0] + nodes[eNIds[0],1]*T[0,1] + nodes[eNIds[0],2]*T[0,2]
        lNodes[0,1] = nodes[eNIds[0],0]*T[1,0] + nodes[eNIds[0],1]*T[1,1] + nodes[eNIds[0],2]*T[1,2]
        lNodes[1,0] = nodes[eNIds[1],0]*T[0,0] + nodes[eNIds[1],1]*T[0,1] + nodes[eNIds[1],2]*T[0,2]
        lNodes[1,1] = nodes[eNIds[1],0]*T[1,0] + nodes[eNIds[1],1]*T[1,1] + nodes[eNIds[1],2]*T[1,2]
        lNodes[2,0] = nodes[eNIds[2],0]*T[0,0] + nodes[eNIds[2],1]*T[0,1] + nodes[eNIds[2],2]*T[0,2]
        lNodes[2,1] = nodes[eNIds[2],0]*T[1,0] + nodes[eNIds[2],1]*T[1,1] + nodes[eNIds[2],2]*T[1,2]

        # Calculate area and matrix B.
        area = ((lNodes[1,0]*lNodes[2,1] - lNodes[2,0]*lNodes[1,1])
                - (lNodes[0,0]*lNodes[2,1] - lNodes[2,0]*lNodes[0,1])
                + (lNodes[0,0]*lNodes[1,1] - lNodes[1,0]*lNodes[0,1]))*0.5

        # if area <= 0.0:
        #     printf("Element %ld area not valid!\n", iElm)

        # if iElm in [184204, 184227, 184239, 118128, 122060, 123569, 122747, 123560]:
        #     printf("Element %ld area is %.*e\n", iElm, 17, area)

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

        # Calculate local mass matrix.
        # Calculate local stiffness matrix.
        dotT(B, D, BTD, 5, 9, 5)
        dot(BTD, B, BTDB, 9, 5, 9)

        # Calculate for samples and assemble them.
        for iSmp in range(nSmp):
            for i in range(9):
                for j in range(9):
                    lK[i,j] = BTDB[i,j]*coefK[iElm,iSmp]*area
                    lM[i,j] = 0.0
                    for k in range(nGps):
                        # lM[i,j] = (tM[0,i,j]*elmGThick[iElm,iSmp,0]*wt[0] \
                        #          + tM[1,i,j]*elmGThick[iElm,iSmp,1]*wt[1] \
                        #          + tM[2,i,j]*elmGThick[iElm,iSmp,2]*wt[2])*area
                        lM[i,j] += tM[k,i,j]*elmGThick[iElm,iSmp,k]*wt[k]
                    lM[i,j] = lM[i,j]*area

            # Transform back to global coord.
            dotT(TT, lM, TTPM, 9, 9, 9)
            dot(TTPM, TT, gM, 9, 9, 9)
            dotT(TT, lK, TTPK, 9, 9, 9)
            dot(TTPK, TT, gK, 9, 9, 9)

            # Assembling.
            for i in range(9):
                for j in range(9):
                    sgM[iSmp,i,j] = gM[i,j]
                    sgK[iSmp,i,j] = gK[i,j]

        # Prepare the lf matrix.
        lf[2] = tracArr[2]*area
        lf[5] = tracArr[5]*area
        lf[8] = tracArr[8]*area

        # Transform back to global coord.
        dotT1(TT, lf, gf, 9, 9)

        # Assembling.
        assembling(eNIds, sgM, sgK, gf, indptr, indices, M, K, f)

        # if iElm == 925:
        #     print('lNodes: {}'.format(np.asarray(lNodes)))
        #     print('B: {}'.format(np.asarray(B)))
        #     print('D: {}'.format(np.asarray(D)))
        #     print('lK: {}'.format(np.asarray(lK)))
        #     print('gK: {}'.format(np.asarray(gK)))

    # Lump the mass matrix.
    lumping(indptr, M, lumpM)


# ----------------------------- Assembling With Damping ----------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assemblingWithDamping(long[::1] eNIds, double[:,:,::1] lM, double[:,:,::1] lC,
                                double[:,:,::1] lK, double[::1] lf,
                                long[::1] indptr, long[::1] indices,
                                double[:,:,:,::1] M, double[:,:,:,::1] C,
                                double[:,:,:,::1] K, double[::1] f):

    nSmp = lM.shape[0]

    cdef long a, b, s
    cdef long row, col, left, right, ptr
    cdef long il, ir, ib

    for a in range(3):
        row = eNIds[a]
        il = row*3
        ir = a*3
        # Assemble the RHS.
        f[il] += lf[ir]
        f[il+1] += lf[ir+1]
        f[il+2] += lf[ir+2]

        for b in range(3):
            col = eNIds[b]
            ib = b*3
            # Search and assemble.
            left = indptr[row]
            right = indptr[row+1]
            ptr = (left + right) / 2

            while indices[ptr] != col:
                if indices[ptr] > col:
                    right = ptr
                else:
                    left = ptr
                ptr = (left + right) / 2

            for s in range(nSmp):

                M[ptr,s,0,0] += lM[s,ir,ib]
                M[ptr,s,0,1] += lM[s,ir,ib+1]
                M[ptr,s,0,2] += lM[s,ir,ib+2]
                M[ptr,s,1,0] += lM[s,ir+1,ib]
                M[ptr,s,1,1] += lM[s,ir+1,ib+1]
                M[ptr,s,1,2] += lM[s,ir+1,ib+2]
                M[ptr,s,2,0] += lM[s,ir+2,ib]
                M[ptr,s,2,1] += lM[s,ir+2,ib+1]
                M[ptr,s,2,2] += lM[s,ir+2,ib+2]

                C[ptr,s,0,0] += lC[s,ir,ib]
                C[ptr,s,0,1] += lC[s,ir,ib+1]
                C[ptr,s,0,2] += lC[s,ir,ib+2]
                C[ptr,s,1,0] += lC[s,ir+1,ib]
                C[ptr,s,1,1] += lC[s,ir+1,ib+1]
                C[ptr,s,1,2] += lC[s,ir+1,ib+2]
                C[ptr,s,2,0] += lC[s,ir+2,ib]
                C[ptr,s,2,1] += lC[s,ir+2,ib+1]
                C[ptr,s,2,2] += lC[s,ir+2,ib+2]

                K[ptr,s,0,0] += lK[s,ir,ib]
                K[ptr,s,0,1] += lK[s,ir,ib+1]
                K[ptr,s,0,2] += lK[s,ir,ib+2]
                K[ptr,s,1,0] += lK[s,ir+1,ib]
                K[ptr,s,1,1] += lK[s,ir+1,ib+1]
                K[ptr,s,1,2] += lK[s,ir+1,ib+2]
                K[ptr,s,2,0] += lK[s,ir+2,ib]
                K[ptr,s,2,1] += lK[s,ir+2,ib+1]
                K[ptr,s,2,2] += lK[s,ir+2,ib+2]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lumpingWithDamping(long[::1] indptr,
                  double[:,:,:,::1] M, double[:,::1] lumpM,
                  double[:,:,:,::1] C, double[:,::1] lumpC):

    nSmp = M.shape[1]

    cdef long nNodes = len(indptr) - 1
    cdef long i, j, ik, jk, s

    for i in range(nNodes):
        for j in range(indptr[i], indptr[i+1]):
            for ik in range(3):
                for jk in range(3):
                    for s in range(nSmp):
                        lumpM[3*i+ik,s] += M[j,s,ik,jk]
                        lumpC[3*i+ik,s] += C[j,s,ik,jk]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedSolidAssembleWithDamping(double[:,::1] nodes, long[:,::1] elements,
                      double[::1] coefs, double[::1] wt, double[:,:,::1] tM,
                      double[:,::1] D, double[:,::1] coefK, double[:,:,::1] elmGThick,
                      long[::1] indptr, long[::1] indices,
                      double[:,:,:,::1] M, double[:,:,:,::1] C, double[:,:,:,::1] K,
                      double[::1] f, double[:,::1] lumpM, double[:,::1] lumpC):

    cdef long nElms = elements.shape[0]
    cdef long iElm, iSmp

    nPts = elements.shape[1]
    ndim = nodes.shape[1]
    nSmp = elmGThick.shape[1]

    cdef double trac = coefs[0]
    cdef double damp = coefs[1]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    cdef double[:,::1] TT = np.zeros((9,9), dtype=np.float)
    cdef double[:,::1] lNodes = np.empty((nPts, 2), dtype=np.float)
    # cdef double[:,::1] D = np.zeros((5, 5), dtype=np.float)
    cdef double[:,::1] B = np.zeros((5, 9), dtype=np.float)
    # traction array used for calc f
    cdef double[::1] tracArr = np.zeros(9, dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    # used for temporary value
    cdef double[:,::1] BTD = np.empty((9,5), dtype=np.float)
    cdef double[:,::1] BTDB = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] TTPM = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] TTPC = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] TTPK = np.empty((9,9), dtype=np.float)

    cdef double[:,::1] lM = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] lC = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] lK = np.empty((9,9), dtype=np.float)
    cdef double[::1] lf = np.zeros(9, dtype=np.float)

    cdef double[:,::1] gM = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] gC = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] gK = np.empty((9,9), dtype=np.float)
    cdef double[::1] gf = np.zeros(9, dtype=np.float)

    cdef double[:,:,::1] sgM = np.empty((nSmp,9,9), dtype=np.float)
    cdef double[:,:,::1] sgC = np.empty((nSmp,9,9), dtype=np.float)
    cdef double[:,:,::1] sgK = np.empty((nSmp,9,9), dtype=np.float)

    cdef double area
    cdef double y23, y31, y12, x32, x13, x21
    cdef double c1, c2, c3, c4
    cdef long i, j

    # Calculate the traction vector prepared for f.
    tracArr[2] = tracArr[5] = tracArr[8] = trac/3.0

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        # Get the transformation matrix T.
        CoordinateTransformation(nodes, eNIds, edges, T, TT)

        # Transform the triangular to sheer.
        lNodes[0,0] = nodes[eNIds[0],0]*T[0,0] + nodes[eNIds[0],1]*T[0,1] + nodes[eNIds[0],2]*T[0,2]
        lNodes[0,1] = nodes[eNIds[0],0]*T[1,0] + nodes[eNIds[0],1]*T[1,1] + nodes[eNIds[0],2]*T[1,2]
        lNodes[1,0] = nodes[eNIds[1],0]*T[0,0] + nodes[eNIds[1],1]*T[0,1] + nodes[eNIds[1],2]*T[0,2]
        lNodes[1,1] = nodes[eNIds[1],0]*T[1,0] + nodes[eNIds[1],1]*T[1,1] + nodes[eNIds[1],2]*T[1,2]
        lNodes[2,0] = nodes[eNIds[2],0]*T[0,0] + nodes[eNIds[2],1]*T[0,1] + nodes[eNIds[2],2]*T[0,2]
        lNodes[2,1] = nodes[eNIds[2],0]*T[1,0] + nodes[eNIds[2],1]*T[1,1] + nodes[eNIds[2],2]*T[1,2]

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

        # Calculate local mass matrix.
        # Calculate local stiffness matrix.
        dotT(B, D, BTD, 5, 9, 5)
        dot(BTD, B, BTDB, 9, 5, 9)

        # Calculate for samples and assemble them.
        for iSmp in range(nSmp):
            for i in range(9):
                for j in range(9):
                    lM[i,j] = (tM[0,i,j]*elmGThick[iElm,iSmp,0]*wt[0] \
                             + tM[1,i,j]*elmGThick[iElm,iSmp,1]*wt[1] \
                             + tM[2,i,j]*elmGThick[iElm,iSmp,2]*wt[2])*area
                    # TODO:: divide by density
                    lC[i,j] = lM[i,j]*damp
                    lK[i,j] = BTDB[i,j]*coefK[iElm,iSmp]*area

            # Transform back to global coord.
            dotT(TT, lM, TTPM, 9, 9, 9)
            dot(TTPM, TT, gM, 9, 9, 9)
            dotT(TT, lC, TTPC, 9, 9, 9)
            dot(TTPC, TT, gC, 9, 9, 9)
            dotT(TT, lK, TTPK, 9, 9, 9)
            dot(TTPK, TT, gK, 9, 9, 9)

            # Assembling.
            for i in range(9):
                for j in range(9):
                    sgM[iSmp,i,j] = gM[i,j]
                    sgC[iSmp,i,j] = gC[i,j]
                    sgK[iSmp,i,j] = gK[i,j]

        # Prepare the lf matrix.
        lf[2] = tracArr[2]*area
        lf[5] = tracArr[5]*area
        lf[8] = tracArr[8]*area

        # Transform back to global coord.
        dotT1(TT, lf, gf, 9, 9)

        # Assembling.
        assemblingWithDamping(eNIds, sgM, sgC, sgK, gf, indptr, indices, M, C, K, f)

        # if iElm == 925:
        #     print('lNodes: {}'.format(np.asarray(lNodes)))
        #     print('B: {}'.format(np.asarray(B)))
        #     print('D: {}'.format(np.asarray(D)))
        #     print('lK: {}'.format(np.asarray(lK)))
        #     print('gK: {}'.format(np.asarray(gK)))

    # Lump the mass matrix.
    lumpingWithDamping(indptr, M, lumpM, C, lumpC)


# ----------------------------- Assembling Updating ----------------------------------------------
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
cdef void assemblingUpdate(long[::1] eNIds,
                           double[:,:,::1] lK, double[:,::1] lf,
                           long[::1] indptr, long[::1] indices,
                           double[:,:,:,::1] K, double[:,::1] f):

    nSmp = lK.shape[0]

    cdef long a, b, s
    cdef long row, col, left, right, ptr
    cdef long il, ir, ib

    for a in range(3):
        row = eNIds[a]
        il = row*3
        ir = a*3

        for s in range(nSmp):
            # Assemble the RHS.
            f[il,s] += lf[s,ir]
            f[il+1,s] += lf[s,ir+1]
            f[il+2,s] += lf[s,ir+2]

        for b in range(3):
            col = eNIds[b]
            ib = b*3
            # Search and assemble.
            left = indptr[row]
            right = indptr[row+1]
            ptr = (left + right) / 2

            while indices[ptr] != col:
                if indices[ptr] > col:
                    right = ptr
                else:
                    left = ptr
                ptr = (left + right) / 2

            for s in range(nSmp):

                K[ptr,s,0,0] += lK[s,ir,ib]
                K[ptr,s,0,1] += lK[s,ir,ib+1]
                K[ptr,s,0,2] += lK[s,ir,ib+2]
                K[ptr,s,1,0] += lK[s,ir+1,ib]
                K[ptr,s,1,1] += lK[s,ir+1,ib+1]
                K[ptr,s,1,2] += lK[s,ir+1,ib+2]
                K[ptr,s,2,0] += lK[s,ir+2,ib]
                K[ptr,s,2,1] += lK[s,ir+2,ib+1]
                K[ptr,s,2,2] += lK[s,ir+2,ib+2]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedSolidAssembleUpdate(double[:,:,::1] nodes, long[:,::1] elements,
                                 double[::1] coefs, double[:,::1] D, double[:,::1] coefK,
                                 long[::1] indptr, long[::1] indices,
                                 double[:,:,:,::1] K, double[:,::1] f):
    cdef long nElms = elements.shape[0]
    cdef long iElm, iSmp

    nPts = elements.shape[1]
    ndim = nodes.shape[2]
    nSmp = coefK.shape[1]

    cdef double trac = coefs[0]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    cdef double[:,::1] TT = np.zeros((9,9), dtype=np.float)
    cdef double[:,::1] lNodes = np.empty((nPts, 2), dtype=np.float)
    # cdef double[:,::1] D = np.zeros((5, 5), dtype=np.float)
    cdef double[:,::1] B = np.zeros((5, 9), dtype=np.float)
    # traction array used for calc f
    cdef double[::1] tracArr = np.zeros(9, dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    # used for temporary value
    cdef double[:,::1] BTD = np.empty((9,5), dtype=np.float)
    cdef double[:,::1] BTDB = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] TTPK = np.empty((9,9), dtype=np.float)

    cdef double[:,::1] lK = np.empty((9,9), dtype=np.float)
    cdef double[::1] lf = np.zeros(9, dtype=np.float)

    cdef double[:,::1] gK = np.empty((9,9), dtype=np.float)
    cdef double[::1] gf = np.zeros(9, dtype=np.float)

    cdef double[:,:,::1] sgK = np.empty((nSmp,9,9), dtype=np.float)
    cdef double[:,::1] sgf = np.empty((nSmp,9), dtype=np.float)

    cdef double area
    cdef double y23, y31, y12, x32, x13, x21
    cdef double c1, c2, c3, c4
    cdef long i, j

    # Calculate the traction vector prepared for f.
    tracArr[2] = tracArr[5] = tracArr[8] = trac/3.0

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for iSmp in range(nSmp):
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

            # Prepare the lf matrix.
            lf[2] = tracArr[2]*area
            lf[5] = tracArr[5]*area
            lf[8] = tracArr[8]*area

            # Calculate local mass matrix.
            # Calculate local stiffness matrix.
            dotT(B, D, BTD, 5, 9, 5)
            dot(BTD, B, BTDB, 9, 5, 9)

            # Calculate for samples and assemble them.
            for i in range(9):
                for j in range(9):
                    lK[i,j] = BTDB[i,j]*coefK[iElm,iSmp]*area

            # Transform back to global coord.
            dotT(TT, lK, TTPK, 9, 9, 9)
            dot(TTPK, TT, gK, 9, 9, 9)
            dotT1(TT, lf, gf, 9, 9)

            # Assembling.
            for i in range(9):
                sgf[iSmp,i] = gf[i]
                for j in range(9):
                    sgK[iSmp,i,j] = gK[i,j]

        # Assembling.
        assemblingUpdate(eNIds, sgK, sgf, indptr, indices, K, f)


# ----------------------------- Assembling Updating with Damping force ----------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assemblingUpdateWithDampingForce(long[::1] eNIds,
                                           double[:,:,::1] lK, double[:,::1] lf, double[:,::1] ldf,
                                           long[::1] indptr, long[::1] indices,
                                           double[:,:,:,::1] K, double[:,::1] f, double[:,::1] df):

    nSmp = lK.shape[0]

    cdef long a, b, s
    cdef long row, col, left, right, ptr
    cdef long il, ir, ib

    for a in range(3):
        row = eNIds[a]
        il = row*3
        ir = a*3

        for s in range(nSmp):
            # Assemble the RHS.
            f[il,s] += lf[s,ir]
            f[il+1,s] += lf[s,ir+1]
            f[il+2,s] += lf[s,ir+2]

            # Assemble the RHS.
            df[il,s] += ldf[s,ir]
            df[il+1,s] += ldf[s,ir+1]
            df[il+2,s] += ldf[s,ir+2]

        for b in range(3):
            col = eNIds[b]
            ib = b*3
            # Search and assemble.
            left = indptr[row]
            right = indptr[row+1]
            ptr = (left + right) / 2

            while indices[ptr] != col:
                if indices[ptr] > col:
                    right = ptr
                else:
                    left = ptr
                ptr = (left + right) / 2

            for s in range(nSmp):

                K[ptr,s,0,0] += lK[s,ir,ib]
                K[ptr,s,0,1] += lK[s,ir,ib+1]
                K[ptr,s,0,2] += lK[s,ir,ib+2]
                K[ptr,s,1,0] += lK[s,ir+1,ib]
                K[ptr,s,1,1] += lK[s,ir+1,ib+1]
                K[ptr,s,1,2] += lK[s,ir+1,ib+2]
                K[ptr,s,2,0] += lK[s,ir+2,ib]
                K[ptr,s,2,1] += lK[s,ir+2,ib+1]
                K[ptr,s,2,2] += lK[s,ir+2,ib+2]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedSolidAssembleUpdateWithDampingForce(double[:,:,::1] nodes, long[:,::1] elements,
                                                 double[::1] coefs, double[:,::1] D, double[:,::1] coefK,
                                                 double[:,::1] acc, double[:,::1] mass,
                                                 long[::1] indptr, long[::1] indices,
                                                 double[:,:,:,::1] K, double[:,::1] f, double[:,::1] df):
    cdef long nElms = elements.shape[0]
    cdef long iElm, iSmp

    nPts = elements.shape[1]
    ndim = nodes.shape[2]
    nSmp = coefK.shape[1]

    cdef double trac = coefs[0]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    cdef double[:,::1] TT = np.zeros((9,9), dtype=np.float)
    cdef double[:,::1] lNodes = np.empty((nPts, 2), dtype=np.float)
    # cdef double[:,::1] D = np.zeros((5, 5), dtype=np.float)
    cdef double[:,::1] B = np.zeros((5, 9), dtype=np.float)
    # traction array used for calc f
    cdef double[::1] tracArr = np.zeros(9, dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    # used for temporary value
    cdef double[:,::1] BTD = np.empty((9,5), dtype=np.float)
    cdef double[:,::1] BTDB = np.empty((9,9), dtype=np.float)
    cdef double[:,::1] TTPK = np.empty((9,9), dtype=np.float)

    cdef double[:,::1] lK = np.empty((9,9), dtype=np.float)
    cdef double[::1] lf = np.zeros(9, dtype=np.float)

    cdef double[:,::1] gK = np.empty((9,9), dtype=np.float)
    cdef double[::1] gf = np.zeros(9, dtype=np.float)

    cdef double[:,:,::1] sgK = np.empty((nSmp,9,9), dtype=np.float)
    cdef double[:,::1] sgf = np.empty((nSmp,9), dtype=np.float)

    # things related with damping force
    cdef double[::1] gAcc = np.empty(9, dtype=np.float)
    cdef double[::1] lAcc = np.empty(9, dtype=np.float)
    cdef double[::1] ldf = np.zeros(9, dtype=np.float)
    cdef double[::1] gdf = np.zeros(9, dtype=np.float)
    cdef double[:,::1] sgdf = np.empty((nSmp,9), dtype=np.float)

    cdef double area
    cdef double y23, y31, y12, x32, x13, x21
    cdef double c1, c2, c3, c4
    cdef long i, j

    # Calculate the traction vector prepared for f.
    tracArr[2] = tracArr[5] = tracArr[8] = trac/3.0

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for iSmp in range(nSmp):

            for i in range(nPts):
                gAcc[i*3] = acc[eNIds[i]*3,iSmp]
                gAcc[i*3+1] = acc[eNIds[i]*3+1,iSmp]
                gAcc[i*3+2] = acc[eNIds[i]*3+2,iSmp]

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

            # Prepare the lf matrix.
            lf[2] = tracArr[2]*area
            lf[5] = tracArr[5]*area
            lf[8] = tracArr[8]*area

            # Calculating the local acc.
            dot1(TT, gAcc, lAcc, 9, 9)
            ldf[2] = -0.1*lAcc[2]*mass[iElm,iSmp]
            ldf[5] = -0.1*lAcc[5]*mass[iElm,iSmp]
            ldf[8] = -0.1*lAcc[8]*mass[iElm,iSmp]

            # Calculate local mass matrix.
            # Calculate local stiffness matrix.
            dotT(B, D, BTD, 5, 9, 5)
            dot(BTD, B, BTDB, 9, 5, 9)

            # Calculate for samples and assemble them.
            for i in range(9):
                for j in range(9):
                    lK[i,j] = BTDB[i,j]*coefK[iElm,iSmp]*area

            # Transform back to global coord.
            dotT(TT, lK, TTPK, 9, 9, 9)
            dot(TTPK, TT, gK, 9, 9, 9)
            dotT1(TT, lf, gf, 9, 9)
            dotT1(TT, ldf, gdf, 9, 9)

            # Assembling.
            for i in range(9):
                sgf[iSmp,i] = gf[i]
                sgdf[iSmp,i] = gdf[i]
                for j in range(9):
                    sgK[iSmp,i,j] = gK[i,j]

        # Assembling.
        assemblingUpdateWithDampingForce(eNIds, sgK, sgf, sgdf, indptr, indices, K, f, df)

# ----------------------------- Assembling Updating f ----------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assemblingUpdatef(long[::1] eNIds, double[:,::1] lf, double[:,::1] f):

    nSmp = lf.shape[0]

    cdef long a, s
    cdef long il, ir

    for a in range(3):
        il = eNIds[a]*3
        ir = a*3

        for s in range(nSmp):
            # Assemble the RHS.
            f[il,s] += lf[s,ir]
            f[il+1,s] += lf[s,ir+1]
            f[il+2,s] += lf[s,ir+2]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedSolidAssembleUpdatef(double[:,:,::1] nodes, long[:,::1] elements,
                                  double[::1] coefs, double[:,::1] f):

    cdef long nElms = elements.shape[0]
    cdef long iElm, iSmp

    nPts = elements.shape[1]
    ndim = nodes.shape[2]
    nSmp = nodes.shape[0]

    cdef double trac = coefs[0]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    cdef double[:,::1] TT = np.zeros((9,9), dtype=np.float)
    cdef double[:,::1] lNodes = np.empty((nPts, 2), dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    # traction array used for calc f
    cdef double[::1] tracArr = np.zeros(9, dtype=np.float)

    cdef double[::1] lf = np.zeros(9, dtype=np.float)
    cdef double[::1] gf = np.zeros(9, dtype=np.float)
    cdef double[:,::1] sgf = np.empty((nSmp,9), dtype=np.float)

    cdef double area
    cdef long i

    # Calculate the traction vector prepared for f.
    tracArr[2] = tracArr[5] = tracArr[8] = trac/3.0

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for iSmp in range(nSmp):
            # Get the transformation matrix T.
            CoordinateTransformationUpdate(nodes, iSmp, eNIds, edges, T, TT)

            # Transform the triangular to sheer.
            lNodes[0,0] = nodes[iSmp,eNIds[0],0]*T[0,0] + nodes[iSmp,eNIds[0],1]*T[0,1] + nodes[iSmp,eNIds[0],2]*T[0,2]
            lNodes[0,1] = nodes[iSmp,eNIds[0],0]*T[1,0] + nodes[iSmp,eNIds[0],1]*T[1,1] + nodes[iSmp,eNIds[0],2]*T[1,2]
            lNodes[1,0] = nodes[iSmp,eNIds[1],0]*T[0,0] + nodes[iSmp,eNIds[1],1]*T[0,1] + nodes[iSmp,eNIds[1],2]*T[0,2]
            lNodes[1,1] = nodes[iSmp,eNIds[1],0]*T[1,0] + nodes[iSmp,eNIds[1],1]*T[1,1] + nodes[iSmp,eNIds[1],2]*T[1,2]
            lNodes[2,0] = nodes[iSmp,eNIds[2],0]*T[0,0] + nodes[iSmp,eNIds[2],1]*T[0,1] + nodes[iSmp,eNIds[2],2]*T[0,2]
            lNodes[2,1] = nodes[iSmp,eNIds[2],0]*T[1,0] + nodes[iSmp,eNIds[2],1]*T[1,1] + nodes[iSmp,eNIds[2],2]*T[1,2]

            # Calculate area.
            area = ((lNodes[1,0]*lNodes[2,1] - lNodes[2,0]*lNodes[1,1])
                    - (lNodes[0,0]*lNodes[2,1] - lNodes[2,0]*lNodes[0,1])
                    + (lNodes[0,0]*lNodes[1,1] - lNodes[1,0]*lNodes[0,1]))*0.5

            # Prepare the lf matrix.
            lf[2] = tracArr[2]*area
            lf[5] = tracArr[5]*area
            lf[8] = tracArr[8]*area

            # Transform back to global coord.
            dotT1(TT, lf, gf, 9, 9)

            # Assembling.
            for i in range(9):
                sgf[iSmp,i] = gf[i]

        # Assembling.
        assemblingUpdatef(eNIds, sgf, f)


# ----------------------------- Calculate Stresses ----------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedCalculateStress(double[:,:,::1] nodes, long[:,::1] elements,
                             double[:,::1] D, double[:,::1]aveElmGE,
                             double[:,::1] u, double[:,:,::1] stress):

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
    cdef long i, j

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
            stress[iElm,iSmp,0] = glbStressTensor[0,0] # xx
            stress[iElm,iSmp,1] = glbStressTensor[1,1] # yy
            stress[iElm,iSmp,2] = glbStressTensor[0,1] # xy
            stress[iElm,iSmp,3] = glbStressTensor[0,2] # xz
            stress[iElm,iSmp,4] = glbStressTensor[1,2] # yz


# ----------------------------------- Other Utility Functions ----------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def D(double[::1] coefs, double[:,::1] D):

    cdef double E = coefs[0]
    cdef double v = coefs[1]
    cdef double k = coefs[2]
    cdef double c1, c2, c3

    # Calculate the constant matrix D.
    c1 = 0.5*(1-v)
    c2 = k*c1
    c3 = (1-v*v)
    D[0,0] = D[1,1] = E/c3
    D[1,0] = D[0,1] = v*E/c3
    D[2,2] = c1*E/c3
    D[3,3] = D[4,4] = c2*E/c3


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def MultiplyByVector(long[::1] indptr, long[::1] indices,
                     double[:,:,:,::1] M, double[:,::1] v,
                     double[:,::1] res):

    cdef long nNodes = len(indptr) - 1
    cdef long a, ia, b
    cdef long r, c

    cdef long nSamples = M.shape[1]
    cdef long s
    cdef int num_threads

    for a in prange(nNodes, nogil=True):
        # if a == 0:
        #     num_threads = openmp.omp_get_num_threads()
        #     printf('How many threads are using: %d\n', num_threads)

        r = a*3
        for ia in range(indptr[a], indptr[a+1]):
            b = indices[ia]
            c = b*3
            for s in range(nSamples):
                res[  r,s] += M[ia,s,0,0]*v[c,s] + M[ia,s,0,1]*v[c+1,s] + M[ia,s,0,2]*v[c+2,s]
                res[r+1,s] += M[ia,s,1,0]*v[c,s] + M[ia,s,1,1]*v[c+1,s] + M[ia,s,1,2]*v[c+2,s]
                res[r+2,s] += M[ia,s,2,0]*v[c,s] + M[ia,s,2,1]*v[c+1,s] + M[ia,s,2,2]*v[c+2,s]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def MultiplyBy1DVector(long[::1] indptr, long[::1] indices,
                       double[:,:,:,::1] M, double[::1] v,
                       double[:,::1] res):

    cdef long nNodes = len(indptr) - 1
    cdef long a, ia, b
    cdef long r, c

    cdef long nSamples = M.shape[1]
    cdef long s

    for a in prange(nNodes, nogil=True):
        r = a*3
        for ia in range(indptr[a], indptr[a+1]):
            b = indices[ia]
            c = b*3
            for s in range(nSamples):
                res[  r,s] += M[ia,s,0,0]*v[c] + M[ia,s,0,1]*v[c+1] + M[ia,s,0,2]*v[c+2]
                res[r+1,s] += M[ia,s,1,0]*v[c] + M[ia,s,1,1]*v[c+1] + M[ia,s,1,2]*v[c+2]
                res[r+2,s] += M[ia,s,2,0]*v[c] + M[ia,s,2,1]*v[c+1] + M[ia,s,2,2]*v[c+2]
