# cython: language_level=3, boundscheck=False
# include math method from C libs.
import numpy as np

cimport numpy as np
from libc.math cimport sqrt
cimport cython

cdef long nPts # number of nodes in shape
cdef long ndim # number of dimensions
cdef double eps = 2.220446049250313e-16

# >>> import sys
# >>> sys.float_info.epsilon
# 2.220446049250313e-16


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedFluidBoundaryAssemble(double[:,::1] nodes, long[::1] glbNodeIds,
                                   long[:,::1] elements, double[::1] elementAreas,
                                   double[::1] h, double[:,::1] lN, double[::1] w,
                                   double[:,::1] RHS):

    cdef long nElms = elements.shape[0]

    cdef long nPts = 3 # elements.shape[1]
    cdef long ndim = 3 # nodes.shape[1]
    cdef long nGps = lN.shape[0]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[::1] lR = np.empty(nPts, dtype=np.float)
    cdef double[:,::1] v = np.empty((2, ndim), dtype=np.float)
    cdef double[::1] elmNorm = np.empty(ndim, dtype=np.float)
    cdef double wGp, hh, elm_norm

    cdef long iElm, iGp, a

    for iElm in range(nElms):

        for a in range(nPts):
            eNIds[a] = glbNodeIds[elements[iElm,a]]
            lR[a] = 0.0

        # Calculate the element's unit normal vector.
        v[0,0] = nodes[eNIds[1],0] - nodes[eNIds[0],0]
        v[0,1] = nodes[eNIds[1],1] - nodes[eNIds[0],1]
        v[0,2] = nodes[eNIds[1],2] - nodes[eNIds[0],2]
        v[1,0] = nodes[eNIds[2],0] - nodes[eNIds[0],0]
        v[1,1] = nodes[eNIds[2],1] - nodes[eNIds[0],1]
        v[1,2] = nodes[eNIds[2],2] - nodes[eNIds[0],2]
        elmNorm[0] = v[0,1]*v[1,2] - v[0,2]*v[1,1]
        elmNorm[1] = v[1,0]*v[0,2] - v[0,0]*v[1,2]
        elmNorm[2] = v[0,0]*v[1,1] - v[0,1]*v[1,0]
        elm_norm = 1.0 / sqrt(elmNorm[0]*elmNorm[0]+elmNorm[1]*elmNorm[1]+elmNorm[2]*elmNorm[2])
        elmNorm[0] = elmNorm[0]*elm_norm
        elmNorm[1] = elmNorm[1]*elm_norm
        elmNorm[2] = elmNorm[2]*elm_norm

        for iGp in range(nGps):
            hh = lN[iGp,0]*h[elements[iElm,0]] + lN[iGp,1]*h[elements[iElm,1]] + lN[iGp,2]*h[elements[iElm,2]]
            wGp = w[iGp] * elementAreas[iElm] * hh
            for a in range(nPts):
                lR[a] -= lN[iGp,a]*wGp

        # Assembling.
        for a in range(nPts):
            RHS[eNIds[a],0] += lR[a]*elmNorm[0]
            RHS[eNIds[a],1] += lR[a]*elmNorm[1]
            RHS[eNIds[a],2] += lR[a]*elmNorm[2]


# jacobian = [[x[1]-x[0], x[2]-x[0], x[3]-x[0]],
#             [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
#             [z[1]-z[0], z[2]-z[0], z[3]-z[0]]]
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double getGlbDerivatives(double[:,::1] nodes, long[::1] eNIds,
                              double[:,::1] lDN, double[:,::1] DN,
                              double[:,::1] G, double[:,::1] jac,
                              double[:,::1] cof, double[:,::1] invJac):

    cdef long a = eNIds[0]
    cdef long b = eNIds[1]
    cdef long c = eNIds[2]
    cdef long d = eNIds[3]

    cdef double detJ, iDetJ

    jac[0,0] = nodes[b,0] - nodes[a,0]
    jac[0,1] = nodes[c,0] - nodes[a,0]
    jac[0,2] = nodes[d,0] - nodes[a,0]
    jac[1,0] = nodes[b,1] - nodes[a,1]
    jac[1,1] = nodes[c,1] - nodes[a,1]
    jac[1,2] = nodes[d,1] - nodes[a,1]
    jac[2,0] = nodes[b,2] - nodes[a,2]
    jac[2,1] = nodes[c,2] - nodes[a,2]
    jac[2,2] = nodes[d,2] - nodes[a,2]

    # jac[0,0] = nodes[a,0]*lDN[0,0] + nodes[b,0]*lDN[1,0] + nodes[c,0]*lDN[2,0] + nodes[d,0]*lDN[3,0]
    # jac[0,1] = nodes[a,0]*lDN[0,1] + nodes[b,0]*lDN[1,1] + nodes[c,0]*lDN[2,1] + nodes[d,0]*lDN[3,1]
    # jac[0,2] = nodes[a,0]*lDN[0,2] + nodes[b,0]*lDN[1,2] + nodes[c,0]*lDN[2,2] + nodes[d,0]*lDN[3,2]
    # jac[1,0] = nodes[a,1]*lDN[0,0] + nodes[b,1]*lDN[1,0] + nodes[c,1]*lDN[2,0] + nodes[d,1]*lDN[3,0]
    # jac[1,1] = nodes[a,1]*lDN[0,1] + nodes[b,1]*lDN[1,1] + nodes[c,1]*lDN[2,1] + nodes[d,1]*lDN[3,1]
    # jac[1,2] = nodes[a,1]*lDN[0,2] + nodes[b,1]*lDN[1,2] + nodes[c,1]*lDN[2,2] + nodes[d,1]*lDN[3,2]
    # jac[2,0] = nodes[a,2]*lDN[0,0] + nodes[b,2]*lDN[1,0] + nodes[c,2]*lDN[2,0] + nodes[d,2]*lDN[3,0]
    # jac[2,1] = nodes[a,2]*lDN[0,1] + nodes[b,2]*lDN[1,1] + nodes[c,2]*lDN[2,1] + nodes[d,2]*lDN[3,1]
    # jac[2,2] = nodes[a,2]*lDN[0,2] + nodes[b,2]*lDN[1,2] + nodes[c,2]*lDN[2,2] + nodes[d,2]*lDN[3,2]

    # +0,0  -0,1  +0,2 --- 0,0  1,0  2,0
    # -1,0  +1,1  -1,2 --- 0,1  1,1  2,1
    # +2,0  -2,1  +2,2 --- 0,2  1,2  2,2
    cof[0,0] = jac[1,1]*jac[2,2] - jac[2,1]*jac[1,2]
    cof[0,1] = jac[2,0]*jac[1,2] - jac[1,0]*jac[2,2]
    cof[0,2] = jac[1,0]*jac[2,1] - jac[2,0]*jac[1,1]
    cof[1,0] = jac[2,1]*jac[0,2] - jac[0,1]*jac[2,2]
    cof[1,1] = jac[0,0]*jac[2,2] - jac[2,0]*jac[0,2]
    cof[1,2] = jac[2,0]*jac[0,1] - jac[0,0]*jac[2,1]
    cof[2,0] = jac[0,1]*jac[1,2] - jac[1,1]*jac[0,2]
    cof[2,1] = jac[1,0]*jac[0,2] - jac[0,0]*jac[1,2]
    cof[2,2] = jac[0,0]*jac[1,1] - jac[1,0]*jac[0,1]

    detJ = jac[0,0]*cof[0,0] + jac[0,1]*cof[0,1] + jac[0,2]*cof[0,2]
    iDetJ = 1.0 / detJ

    invJac[0,0] = cof[0,0] * iDetJ
    invJac[0,1] = cof[1,0] * iDetJ
    invJac[0,2] = cof[2,0] * iDetJ
    invJac[1,0] = cof[0,1] * iDetJ
    invJac[1,1] = cof[1,1] * iDetJ
    invJac[1,2] = cof[2,1] * iDetJ
    invJac[2,0] = cof[0,2] * iDetJ
    invJac[2,1] = cof[1,2] * iDetJ
    invJac[2,2] = cof[2,2] * iDetJ

    # DN = trans(invJ)lDN
    DN[0,0] = lDN[0,0]*invJac[0,0] + lDN[1,0]*invJac[1,0] + lDN[2,0]*invJac[2,0]
    DN[0,1] = lDN[0,1]*invJac[0,0] + lDN[1,1]*invJac[1,0] + lDN[2,1]*invJac[2,0]
    DN[0,2] = lDN[0,2]*invJac[0,0] + lDN[1,2]*invJac[1,0] + lDN[2,2]*invJac[2,0]
    DN[0,3] = lDN[0,3]*invJac[0,0] + lDN[1,3]*invJac[1,0] + lDN[2,3]*invJac[2,0]

    DN[1,0] = lDN[0,0]*invJac[0,1] + lDN[1,0]*invJac[1,1] + lDN[2,0]*invJac[2,1]
    DN[1,1] = lDN[0,1]*invJac[0,1] + lDN[1,1]*invJac[1,1] + lDN[2,1]*invJac[2,1]
    DN[1,2] = lDN[0,2]*invJac[0,1] + lDN[1,2]*invJac[1,1] + lDN[2,2]*invJac[2,1]
    DN[1,3] = lDN[0,3]*invJac[0,1] + lDN[1,3]*invJac[1,1] + lDN[2,3]*invJac[2,1]

    DN[2,0] = lDN[0,0]*invJac[0,2] + lDN[1,0]*invJac[1,2] + lDN[2,0]*invJac[2,2]
    DN[2,1] = lDN[0,1]*invJac[0,2] + lDN[1,1]*invJac[1,2] + lDN[2,1]*invJac[2,2]
    DN[2,2] = lDN[0,2]*invJac[0,2] + lDN[1,2]*invJac[1,2] + lDN[2,2]*invJac[2,2]
    DN[2,3] = lDN[0,3]*invJac[0,2] + lDN[1,3]*invJac[1,2] + lDN[2,3]*invJac[2,2]


    G[0,0] = invJac[0,0]*invJac[0,0] + invJac[1,0]*invJac[1,0] + invJac[2,0]*invJac[2,0]
    G[0,1] = invJac[0,0]*invJac[0,1] + invJac[1,0]*invJac[1,1] + invJac[2,0]*invJac[2,1]
    G[0,2] = invJac[0,0]*invJac[0,2] + invJac[1,0]*invJac[1,2] + invJac[2,0]*invJac[2,2]
    # G[1,0] = invJac[0,1]*invJac[0,0] + invJac[1,1]*invJac[1,0] + invJac[2,1]*invJac[2,0]
    G[1,1] = invJac[0,1]*invJac[0,1] + invJac[1,1]*invJac[1,1] + invJac[2,1]*invJac[2,1]
    G[1,2] = invJac[0,1]*invJac[0,2] + invJac[1,1]*invJac[1,2] + invJac[2,1]*invJac[2,2]
    # G[2,0] = invJac[0,2]*invJac[0,0] + invJac[1,2]*invJac[1,0] + invJac[2,2]*invJac[2,0]
    # G[2,1] = invJac[0,2]*invJac[0,1] + invJac[1,2]*invJac[1,1] + invJac[2,2]*invJac[2,1]
    G[2,2] = invJac[0,2]*invJac[0,2] + invJac[1,2]*invJac[1,2] + invJac[2,2]*invJac[2,2]
    G[1,0] = G[0,1]
    G[2,0] = G[0,2]
    G[2,1] = G[1,2]

    return detJ / 6.0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assembling(long[::1] eNIds, double[:,:,::1] lLHS, double[:,::1] lR,
                     long[::1] indptr, long[::1] indices,
                     double[:,:,::1] LHS, double[:,::1] RHS):

    cdef long a, b
    cdef long row, col, left, right, ptr

    for a in range(4):
        row = eNIds[a]
        # Assemble the RHS.
        RHS[row,0] += lR[0,a]
        RHS[row,1] += lR[1,a]
        RHS[row,2] += lR[2,a]
        RHS[row,3] += lR[3,a]

        for b in range(4):
            col = eNIds[b]
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

            LHS[ptr,0,0] += lLHS[0,a,b]
            LHS[ptr,0,1] += lLHS[1,a,b]
            LHS[ptr,0,2] += lLHS[2,a,b]
            LHS[ptr,0,3] += lLHS[3,a,b]
            LHS[ptr,1,0] += lLHS[4,a,b]
            LHS[ptr,1,1] += lLHS[5,a,b]
            LHS[ptr,1,2] += lLHS[6,a,b]
            LHS[ptr,1,3] += lLHS[7,a,b]
            LHS[ptr,2,0] += lLHS[8,a,b]
            LHS[ptr,2,1] += lLHS[9,a,b]
            LHS[ptr,2,2] += lLHS[10,a,b]
            LHS[ptr,2,3] += lLHS[11,a,b]
            LHS[ptr,3,0] += lLHS[12,a,b]
            LHS[ptr,3,1] += lLHS[13,a,b]
            LHS[ptr,3,2] += lLHS[14,a,b]
            LHS[ptr,3,3] += lLHS[15,a,b]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedFluidAssemble(double[:,::1] nodes, long[:,::1] elements,
                           double[:,::1] interDu, double[:,::1] interU,
                           double[::1] interP, double[:,::1] f,
                           double[::1] coefs, double[:,::1] lN,
                           double[:,::1] lDN, double[::1] w,
                           long[::1] indptr, long[::1] indices,
                           double[:,:,::1] LHS, double[:,::1] RHS):

    cdef long nElms = elements.shape[0]

    cdef long nPts = 4 # elements.shape[1]
    cdef long ndim = 3 # nodes.shape[1]

    cdef double am = coefs[0]
    cdef double af = coefs[1]
    cdef double gamma = coefs[2]
    cdef double dt = coefs[3]
    cdef double rho = coefs[4] # density
    cdef double mu = coefs[5] # dynamic viscocity
    cdef double ci = coefs[6] # Ci

    cdef double nu = mu / rho # kinematic viscocity
    cdef double mr = am * rho
    cdef double fgt = af * gamma * dt
    cdef double mdfgt = am / fgt

    # Shape functions.
    # lN, lDN

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] DN = np.empty((ndim, nPts), dtype=np.float)
    cdef double[::1] DivN = np.empty(nPts, dtype=np.float)
    cdef double[:,::1] G = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] uh = np.empty(ndim, dtype=np.float)
    cdef double[:,::1] gradUh = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] duh = np.empty(ndim, dtype=np.float)
    cdef double ph
    cdef double[::1] gradPh = np.empty(ndim, dtype=np.float)
    cdef double[::1] fh = np.empty(ndim, dtype=np.float)

    cdef double[::1] uhDN = np.empty(nPts, dtype=np.float)
    cdef double[::1] rM = np.empty(ndim, dtype=np.float)
    cdef double[:,::1] rN = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] rV = np.empty(ndim, dtype=np.float)

    cdef double[:,:,::1] lLHS = np.empty((16,nPts,nPts), dtype=np.float)
    cdef double[:,::1] lR = np.empty((4,nPts), dtype=np.float)

    cdef double jac[3][3]
    cdef double cof[3][3]
    cdef double invJac[3][3]

    cdef double Ve # volume

    cdef double wGp, wGpV
    cdef double uhGuh
    cdef double DNDN
    cdef double GG, trG
    cdef double tauSP, tauM
    cdef double tauC
    cdef double trGradUh # Rc
    cdef double wr, wrl, wl
    cdef double T1, T2, T3

    cdef double c1 = 4.0 / (dt * dt)
    cdef double c2 = ci * nu * nu
    cdef double c3 = 1.0 / rho

    cdef long nGp = 4 # w.shape[0]
    cdef long iGp
    cdef int i, j, k, a, b


    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for i in range(4):
            for j in range(nPts):
                lR[i,j] = 0.0

        for i in range(16):
            for j in range(nPts):
                for k in range(nPts):
                    lLHS[i,j,k] = 0.0

        Ve = getGlbDerivatives(nodes, eNIds, lDN, DN, G, jac, cof, invJac)

        # print "vol ", vol
        # print "DN ", np.asarray(DN)
        # print "G ", np.asarray(G)

        # For tau_SUPS
        GG =  G[0,0]*G[0,0] + G[0,1]*G[0,1] + G[0,2]*G[0,2] \
            + G[1,0]*G[1,0] + G[1,1]*G[1,1] + G[1,2]*G[1,2] \
            + G[2,0]*G[2,0] + G[2,1]*G[2,1] + G[2,2]*G[2,2]

        tauSP = c1 + c2*GG
        trG = G[0,0] + G[1,1] + G[2,2]

        # print "GG ", GG
        # print "trG ", trG

        # gradDuh
        gradUh[0,0] = interU[eNIds[0],0]*DN[0,0] + interU[eNIds[1],0]*DN[0,1] \
                        + interU[eNIds[2],0]*DN[0,2] + interU[eNIds[3],0]*DN[0,3]
        gradUh[0,1] = interU[eNIds[0],0]*DN[1,0] + interU[eNIds[1],0]*DN[1,1] \
                        + interU[eNIds[2],0]*DN[1,2] + interU[eNIds[3],0]*DN[1,3]
        gradUh[0,2] = interU[eNIds[0],0]*DN[2,0] + interU[eNIds[1],0]*DN[2,1] \
                        + interU[eNIds[2],0]*DN[2,2] + interU[eNIds[3],0]*DN[2,3]
        gradUh[1,0] = interU[eNIds[0],1]*DN[0,0] + interU[eNIds[1],1]*DN[0,1] \
                        + interU[eNIds[2],1]*DN[0,2] + interU[eNIds[3],1]*DN[0,3]
        gradUh[1,1] = interU[eNIds[0],1]*DN[1,0] + interU[eNIds[1],1]*DN[1,1] \
                        + interU[eNIds[2],1]*DN[1,2] + interU[eNIds[3],1]*DN[1,3]
        gradUh[1,2] = interU[eNIds[0],1]*DN[2,0] + interU[eNIds[1],1]*DN[2,1] \
                        + interU[eNIds[2],1]*DN[2,2] + interU[eNIds[3],1]*DN[2,3]
        gradUh[2,0] = interU[eNIds[0],2]*DN[0,0] + interU[eNIds[1],2]*DN[0,1] \
                        + interU[eNIds[2],2]*DN[0,2] + interU[eNIds[3],2]*DN[0,3]
        gradUh[2,1] = interU[eNIds[0],2]*DN[1,0] + interU[eNIds[1],2]*DN[1,1] \
                        + interU[eNIds[2],2]*DN[1,2] + interU[eNIds[3],2]*DN[1,3]
        gradUh[2,2] = interU[eNIds[0],2]*DN[2,0] + interU[eNIds[1],2]*DN[2,1] \
                        + interU[eNIds[2],2]*DN[2,2] + interU[eNIds[3],2]*DN[2,3]

        trGradUh = gradUh[0,0] + gradUh[1,1] + gradUh[2,2]

        # print "gradDuh ", np.asarray(gradDuh)
        # print "trGradDuh ", trGradDuh

        # gradPh
        gradPh[0] = interP[eNIds[0]]*DN[0,0] + interP[eNIds[1]]*DN[0,1] \
                    + interP[eNIds[2]]*DN[0,2] + interP[eNIds[3]]*DN[0,3]
        gradPh[1] = interP[eNIds[0]]*DN[1,0] + interP[eNIds[1]]*DN[1,1] \
                    + interP[eNIds[2]]*DN[1,2] + interP[eNIds[3]]*DN[1,3]
        gradPh[2] = interP[eNIds[0]]*DN[2,0] + interP[eNIds[1]]*DN[2,1] \
                    + interP[eNIds[2]]*DN[2,2] + interP[eNIds[3]]*DN[2,3]

        # print "gradPh ", np.asarray(gradPh)

        # For Rc
        DivN[0] = DN[0,0] + DN[1,0] + DN[2,0]
        DivN[1] = DN[0,1] + DN[1,1] + DN[2,1]
        DivN[2] = DN[0,2] + DN[1,2] + DN[2,2]
        DivN[3] = DN[0,3] + DN[1,3] + DN[2,3]


        # Loop through gaussian points. nGp
        for iGp in range(nGp):

            wGp = w[iGp]
            wGpV = wGp * Ve

            # print "wGp ", wGp

            wr = wGpV * rho
            wrl = wr * fgt
            wl = wGpV * fgt

            uh[0] = interU[eNIds[0],0]*lN[iGp,0] + interU[eNIds[1],0]*lN[iGp,1] \
                        + interU[eNIds[2],0]*lN[iGp,2] + interU[eNIds[3],0]*lN[iGp,3]
            uh[1] = interU[eNIds[0],1]*lN[iGp,0] + interU[eNIds[1],1]*lN[iGp,1] \
                        + interU[eNIds[2],1]*lN[iGp,2] + interU[eNIds[3],1]*lN[iGp,3]
            uh[2] = interU[eNIds[0],2]*lN[iGp,0] + interU[eNIds[1],2]*lN[iGp,1] \
                        + interU[eNIds[2],2]*lN[iGp,2] + interU[eNIds[3],2]*lN[iGp,3]

            duh[0] = interDu[eNIds[0],0]*lN[iGp,0] + interDu[eNIds[1],0]*lN[iGp,1] \
                        + interDu[eNIds[2],0]*lN[iGp,2] + interDu[eNIds[3],0]*lN[iGp,3]
            duh[1] = interDu[eNIds[0],1]*lN[iGp,0] + interDu[eNIds[1],1]*lN[iGp,1] \
                        + interDu[eNIds[2],1]*lN[iGp,2] + interDu[eNIds[3],1]*lN[iGp,3]
            duh[2] = interDu[eNIds[0],2]*lN[iGp,0] + interDu[eNIds[1],2]*lN[iGp,1] \
                        + interDu[eNIds[2],2]*lN[iGp,2] + interDu[eNIds[3],2]*lN[iGp,3]

            ph = interP[eNIds[0]]*lN[iGp,0] + interP[eNIds[1]]*lN[iGp,1] \
                        + interP[eNIds[2]]*lN[iGp,2] + interP[eNIds[3]]*lN[iGp,3]

            fh[0] = f[eNIds[0],0]*lN[iGp,0] + f[eNIds[1],0]*lN[iGp,1] \
                        + f[eNIds[2],0]*lN[iGp,2] + f[eNIds[3],0]*lN[iGp,3]
            fh[1] = f[eNIds[0],1]*lN[iGp,0] + f[eNIds[1],1]*lN[iGp,1] \
                        + f[eNIds[2],1]*lN[iGp,2] + f[eNIds[3],1]*lN[iGp,3]
            fh[2] = f[eNIds[0],2]*lN[iGp,0] + f[eNIds[1],2]*lN[iGp,1] \
                        + f[eNIds[2],2]*lN[iGp,2] + f[eNIds[3],2]*lN[iGp,3]

            # print "duh ", np.asarray(duh)
            # print "dduh ", np.asarray(dduh)
            # print "ph ", ph

            # tauM := tau_SUPS
            uhGuh = uh[0]*(uh[0]*G[0,0] + uh[1]*G[0,1] + uh[2]*G[0,2]) \
                    + uh[1]*(uh[0]*G[1,0] + uh[1]*G[1,1] + uh[2]*G[1,2]) \
                    + uh[2]*(uh[0]*G[2,0] + uh[1]*G[2,1] + uh[2]*G[2,2])
            tauM = 1.0 / sqrt(tauSP + uhGuh)

            # print "duhGduh ", duhGduh
            # print "tauM ", tauM

            # tauC := v_LSIC
            tauC = 1.0 / (trG * tauM)

            # print "tauC ", tauC

            # for Rm
            rV[0] = duh[0] + uh[0]*gradUh[0,0] + uh[1]*gradUh[0,1] + uh[2]*gradUh[0,2] - fh[0]
            rV[1] = duh[1] + uh[0]*gradUh[1,0] + uh[1]*gradUh[1,1] + uh[2]*gradUh[1,2] - fh[1]
            rV[2] = duh[2] + uh[0]*gradUh[2,0] + uh[1]*gradUh[2,1] + uh[2]*gradUh[2,2] - fh[2]
            # tauM * Rm / rho
            rM[0] = tauM*(rV[0] + gradPh[0]*c3)
            rM[1] = tauM*(rV[1] + gradPh[1]*c3)
            rM[2] = tauM*(rV[2] + gradPh[2]*c3)
            # Nm-6
            rV[0] -= rM[0]*gradUh[0,0] + rM[1]*gradUh[0,1] + rM[2]*gradUh[0,2]
            rV[1] -= rM[0]*gradUh[1,0] + rM[1]*gradUh[1,1] + rM[2]*gradUh[1,2]
            rV[2] -= rM[0]*gradUh[2,0] + rM[1]*gradUh[2,1] + rM[2]*gradUh[2,2]

            # print "rV ", np.asarray(rV)

            # Stress tensor (sigma) + Nm-7
            # Use rN to store stress tensor (sigma) for temporary + Nm-7
            # T1 = tauC*trGradDuh - ph*c3
            T1 = -ph*c3

            rN[0,0] = nu*(gradUh[0,0]+gradUh[0,0]) - rM[0]*rM[0] + T1
            rN[0,1] = nu*(gradUh[0,1]+gradUh[1,0]) - rM[0]*rM[1]
            rN[0,2] = nu*(gradUh[0,2]+gradUh[2,0]) - rM[0]*rM[2]

            rN[1,0] = rN[0,1]
            rN[1,1] = nu*(gradUh[1,1]+gradUh[1,1]) - rM[1]*rM[1] + T1
            rN[1,2] = nu*(gradUh[1,2]+gradUh[2,1]) - rM[1]*rM[2]

            rN[2,0] = rN[0,2]
            rN[2,1] = rN[1,2]
            rN[2,2] = nu*(gradUh[2,2]+gradUh[2,2]) - rM[2]*rM[2] + T1

            # print "rN ", np.asarray(rN)

            # for K-2, Rc
            uhDN[0] = uh[0]*DN[0,0] + uh[1]*DN[1,0] + uh[2]*DN[2,0]
            uhDN[1] = uh[0]*DN[0,1] + uh[1]*DN[1,1] + uh[2]*DN[2,1]
            uhDN[2] = uh[0]*DN[0,2] + uh[1]*DN[1,2] + uh[2]*DN[2,2]
            uhDN[3] = uh[0]*DN[0,3] + uh[1]*DN[1,3] + uh[2]*DN[2,3]


            for a in range(nPts):

                T1 = tauC*trGradUh

                lR[0,a] += wr*(rV[0]*lN[iGp,a] + rN[0,0]*DN[0,a] + rN[0,1]*DN[1,a] + rN[0,2]*DN[2,a] \
                               + uhDN[a]*rM[0] + T1*DN[0,a])
                lR[1,a] += wr*(rV[1]*lN[iGp,a] + rN[1,0]*DN[0,a] + rN[1,1]*DN[1,a] + rN[1,2]*DN[2,a] \
                               + uhDN[a]*rM[1] + T1*DN[1,a])
                lR[2,a] += wr*(rV[2]*lN[iGp,a] + rN[2,0]*DN[0,a] + rN[2,1]*DN[1,a] + rN[2,2]*DN[2,a] \
                               + uhDN[a]*rM[2] + T1*DN[2,a])

                lR[3,a] += wGpV*(rM[0]*DN[0,a]+rM[1]*DN[1,a]+rM[2]*DN[2,a])


            for a in range(nPts):
                for b in range(nPts):

                    # DN(b) cross_prod DN(a)
                    rN[0,0] = DN[0,b]*DN[0,a]
                    rN[0,1] = DN[0,b]*DN[1,a]
                    rN[0,2] = DN[0,b]*DN[2,a]
                    rN[1,0] = DN[1,b]*DN[0,a]
                    rN[1,1] = DN[1,b]*DN[1,a]
                    rN[1,2] = DN[1,b]*DN[2,a]
                    rN[2,0] = DN[2,b]*DN[0,a]
                    rN[2,1] = DN[2,b]*DN[1,a]
                    rN[2,2] = DN[2,b]*DN[2,a]

                    # print "rN ", np.asarray(rN)

                    # DN(a) dot_prod DN(b)
                    DNDN = DN[0,a]*DN[0,b] + DN[1,a]*DN[1,b] + DN[2,a]*DN[2,b]

                    # print "DNDN ", np.asarray(DNDN)

                    # K-2, K-3, K-4, K-6
                    T1 = tauM*uhDN[a]*(lN[iGp,b]*mdfgt + uhDN[b]) \
                        + lN[iGp,a]*uhDN[b] + nu*DNDN
                    T2 = tauM*uhDN[a]
                    T3 = tauM*(mdfgt*lN[iGp,b] + uhDN[b])

                    # print "T123 ", T1, T2, T3
                    # print "Debug: nu ", nu, " mdfgt ", mdfgt
                    # print "Debug: upDN ", np.asarray(upDN)
                    # print "Debug: duhDN ", np.asarray(duhDN)
                    # print "Debug: uaDN ", np.asarray(uaDN)

                    # K  dM/dU
                    lLHS[0,a,b] += wrl*((nu + tauC)*rN[0,0] + T1)
                    lLHS[1,a,b] += wrl*(nu*rN[0,1] + tauC*rN[1,0])
                    lLHS[2,a,b] += wrl*(nu*rN[0,2] + tauC*rN[2,0])

                    lLHS[4,a,b] += wrl*(nu*rN[1,0] + tauC*rN[0,1])
                    lLHS[5,a,b] += wrl*((nu + tauC)*rN[1,1] + T1)
                    lLHS[6,a,b] += wrl*(nu*rN[1,2] + tauC*rN[2,1])

                    lLHS[8,a,b] += wrl*(nu*rN[2,0] + tauC*rN[0,2])
                    lLHS[9,a,b] += wrl*(nu*rN[2,1] + tauC*rN[1,2])
                    lLHS[10,a,b] += wrl*((nu + tauC)*rN[2,2] + T1)

                    # G  dM/dP
                    lLHS[3,a,b] -= wGpV*(DN[0,a]*lN[iGp,b] - DN[0,b]*T2)
                    lLHS[7,a,b] -= wGpV*(DN[1,a]*lN[iGp,b] - DN[1,b]*T2)
                    lLHS[11,a,b] -= wGpV*(DN[2,a]*lN[iGp,b] - DN[2,b]*T2)

                    # D  dC/dU
                    lLHS[12,a,b] += wl*(lN[iGp,a]*DN[0,b] + DN[0,a]*T3)
                    lLHS[13,a,b] += wl*(lN[iGp,a]*DN[1,b] + DN[1,a]*T3)
                    lLHS[14,a,b] += wl*(lN[iGp,a]*DN[2,b] + DN[2,a]*T3)

                    # L
                    lLHS[15,a,b] += wGpV*tauM*DNDN*c3

        # Process the items do not depend on Gaussian points.
        T1 = trGradUh * Ve * 0.25 # Nc-1
        T2 = mr * Ve * 0.05 # K-1

        for a in range(nPts):
            lR[3,a] += T1
            for b in range(nPts):
                # K
                lLHS[0,a,b] += T2
                lLHS[5,a,b] += T2
                lLHS[10,a,b] += T2


        # Do the assembling!
        assembling(eNIds, lLHS, lR, indptr, indices, LHS, RHS)

        # print np.asarray(lR)
        # print np.asarray(lLHS[:,0,0])

        # for i in range(nPts):
        #     if eNIds[i] == 2323:
        #         print np.asarray(lR)

        # for a in range(nPts):
        #     for b in range(nPts):
        #         print np.asarray(lLHS[:,a,b])

