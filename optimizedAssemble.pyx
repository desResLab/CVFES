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
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int iszero(a):
    # cdef double eps = 2.220446049250313e-15
    return abs(a) < 10.0*eps*max(a, eps)


# jacobian = [[x[1]-x[0], x[2]-x[0], x[3]-x[0]],
#             [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
#             [z[1]-z[0], z[2]-z[0], z[3]-z[0]]]
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double getGlbDerivatives(double[:,::1] nodes, long[::1] eNIds,
                              double[:,::1] lDN, double[:,::1] glbDN,
                              double[:,::1] G, double[:,::1] jac,
                              double[:,::1] invJac, double[:,::1] cof):

    cdef long a = eNIds[0]
    cdef long b = eNIds[1]
    cdef long c = eNIds[2]
    cdef long d = eNIds[3]

    # cdef double[:,::1] jac = np.empty((3, 3), dtype=np.float) # jacobian
    # cdef double[:,::1] invJac = np.empty((3, 3), dtype=np.float) # inverse of jacobian
    # cdef double[:,::1] cof = np.empty((3,3), dtype=np.float) # cofactors
    cdef double detJ, iDetJ

    # jac[0,0] = nodes[b,0] - nodes[a,0]
    # jac[0,1] = nodes[c,0] - nodes[a,0]
    # jac[0,2] = nodes[d,0] - nodes[a,0]
    # jac[1,0] = nodes[b,1] - nodes[a,1]
    # jac[1,1] = nodes[c,1] - nodes[a,1]
    # jac[1,2] = nodes[d,1] - nodes[a,1]
    # jac[2,0] = nodes[b,2] - nodes[a,2]
    # jac[2,1] = nodes[c,2] - nodes[a,2]
    # jac[2,2] = nodes[d,2] - nodes[a,2]

    jac[0,0] = nodes[a,0]*lDN[0,0] + nodes[b,0]*lDN[1,0] + nodes[c,0]*lDN[2,0] + nodes[d,0]*lDN[3,0]
    jac[0,1] = nodes[a,0]*lDN[0,1] + nodes[b,0]*lDN[1,1] + nodes[c,0]*lDN[2,1] + nodes[d,0]*lDN[3,1]
    jac[0,2] = nodes[a,0]*lDN[0,2] + nodes[b,0]*lDN[1,2] + nodes[c,0]*lDN[2,2] + nodes[d,0]*lDN[3,2]
    jac[1,0] = nodes[a,1]*lDN[0,0] + nodes[b,1]*lDN[1,0] + nodes[c,1]*lDN[2,0] + nodes[d,1]*lDN[3,0]
    jac[1,1] = nodes[a,1]*lDN[0,1] + nodes[b,1]*lDN[1,1] + nodes[c,1]*lDN[2,1] + nodes[d,1]*lDN[3,1]
    jac[1,2] = nodes[a,1]*lDN[0,2] + nodes[b,1]*lDN[1,2] + nodes[c,1]*lDN[2,2] + nodes[d,1]*lDN[3,2]
    jac[2,0] = nodes[a,2]*lDN[0,0] + nodes[b,2]*lDN[1,0] + nodes[c,2]*lDN[2,0] + nodes[d,2]*lDN[3,0]
    jac[2,1] = nodes[a,2]*lDN[0,1] + nodes[b,2]*lDN[1,1] + nodes[c,2]*lDN[2,1] + nodes[d,2]*lDN[3,1]
    jac[2,2] = nodes[a,2]*lDN[0,2] + nodes[b,2]*lDN[1,2] + nodes[c,2]*lDN[2,2] + nodes[d,2]*lDN[3,2]

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

    glbDN[0,0] = lDN[0,0]*invJac[0,0] + lDN[0,1]*invJac[1,0] + lDN[0,2]*invJac[2,0]
    glbDN[0,1] = lDN[0,0]*invJac[0,1] + lDN[0,1]*invJac[1,1] + lDN[0,2]*invJac[2,1]
    glbDN[0,2] = lDN[0,0]*invJac[0,2] + lDN[0,1]*invJac[1,2] + lDN[0,2]*invJac[2,2]
    glbDN[1,0] = lDN[1,0]*invJac[0,0] + lDN[1,1]*invJac[1,0] + lDN[1,2]*invJac[2,0]
    glbDN[1,1] = lDN[1,0]*invJac[0,1] + lDN[1,1]*invJac[1,1] + lDN[1,2]*invJac[2,1]
    glbDN[1,2] = lDN[1,0]*invJac[0,2] + lDN[1,1]*invJac[1,2] + lDN[1,2]*invJac[2,2]
    glbDN[2,0] = lDN[2,0]*invJac[0,0] + lDN[2,1]*invJac[1,0] + lDN[2,2]*invJac[2,0]
    glbDN[2,1] = lDN[2,0]*invJac[0,1] + lDN[2,1]*invJac[1,1] + lDN[2,2]*invJac[2,1]
    glbDN[2,2] = lDN[2,0]*invJac[0,2] + lDN[2,1]*invJac[1,2] + lDN[2,2]*invJac[2,2]
    glbDN[3,0] = lDN[3,0]*invJac[0,0] + lDN[3,1]*invJac[1,0] + lDN[3,2]*invJac[2,0]
    glbDN[3,1] = lDN[3,0]*invJac[0,1] + lDN[3,1]*invJac[1,1] + lDN[3,2]*invJac[2,1]
    glbDN[3,2] = lDN[3,0]*invJac[0,2] + lDN[3,1]*invJac[1,2] + lDN[3,2]*invJac[2,2]

    return detJ

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
def OptimizedAssemble(double[:,::1] nodes, long[:,::1] elements,
                      double[:,::1] interDDu, double[:,::1] interDu,
                      double[::1] interP, double[::1] coefs,
                      double[:,::1] N, double[:,:,::1] NN,
                      double[:,::1] lDN, double[::1] w,
                      long[::1] indptr, long[::1] indices,
                      double[:,:,::1] LHS, double[:,::1] RHS):

    cdef long nElms = elements.shape[0]
    cdef long iElm

    nPts = elements.shape[1]
    ndim = nodes.shape[1]

    cdef double am = coefs[0]
    cdef double af = coefs[1]
    cdef double gamma = coefs[2]
    cdef double dt = coefs[3]
    cdef double rho = coefs[4]
    cdef double mu = coefs[5]
    cdef double ci = coefs[6]

    cdef double nu = mu / rho
    cdef double mr = am * rho
    cdef double fgt = af * gamma * dt
    cdef double mdfgt = am / fgt

    # print "af ", af, am, gamma
    # print "mdfgt ", mdfgt, dt

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] DN = np.empty((nPts, ndim), dtype=np.float)
    cdef double[:,::1] G = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] duh = np.empty(ndim, dtype=np.float)
    cdef double[:,::1] gradDuh = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] dduh = np.empty(ndim, dtype=np.float)
    cdef double[::1] gradPh = np.empty(ndim, dtype=np.float)
    cdef double[::1] duhDN = np.empty(nPts, dtype=np.float)
    cdef double[::1] upDN = np.empty(nPts, dtype=np.float)
    cdef double[::1] uaDN = np.empty(nPts, dtype=np.float)
    cdef double[::1] up = np.empty(ndim, dtype=np.float)
    cdef double[::1] ua = np.empty(ndim, dtype=np.float)
    cdef double[:,::1] rM = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] rV = np.empty(ndim, dtype=np.float)

    cdef double[:,:,::1] lLHS = np.empty((16,nPts,nPts), dtype=np.float)
    cdef double[:,::1] lR = np.empty((4,nPts), dtype=np.float)

    cdef double jac[3][3]
    cdef double invJac[3][3]
    cdef double cof[3][3]

    cdef long nGp = w.shape[0]
    cdef long iGp
    cdef int i, j, k, a, b

    cdef double ph
    cdef double detJ
    cdef double wGp
    cdef double duhGduh
    cdef double DNDN
    cdef double GG, trG
    cdef double tauSP, tauM
    cdef double tauB, tauC
    cdef double trGradDuh
    cdef double T1, T2, T3
    cdef double wr, wrl, wl

    cdef double c1 = 4.0 / (dt * dt)
    cdef double c2 = ci * nu * nu
    cdef double c3 = 1.0 / rho

    for iElm in range(nElms):

        # if iElm != 742:
        #     continue

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for i in range(4):
            for j in range(nPts):
                lR[i,j] = 0.0

        for i in range(16):
            for j in range(nPts):
                for k in range(nPts):
                    lLHS[i,j,k] = 0.0

        detJ = getGlbDerivatives(nodes, eNIds, lDN, DN, G, jac, invJac, cof)

        # print "jac ", detJ
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
        gradDuh[0,0] = interDu[eNIds[0],0]*DN[0,0] + interDu[eNIds[1],0]*DN[1,0] \
                        + interDu[eNIds[2],0]*DN[2,0] + interDu[eNIds[3],0]*DN[3,0]
        gradDuh[1,0] = interDu[eNIds[0],0]*DN[0,1] + interDu[eNIds[1],0]*DN[1,1] \
                        + interDu[eNIds[2],0]*DN[2,1] + interDu[eNIds[3],0]*DN[3,1]
        gradDuh[2,0] = interDu[eNIds[0],0]*DN[0,2] + interDu[eNIds[1],0]*DN[1,2] \
                        + interDu[eNIds[2],0]*DN[2,2] + interDu[eNIds[3],0]*DN[3,2]
        gradDuh[0,1] = interDu[eNIds[0],1]*DN[0,0] + interDu[eNIds[1],1]*DN[1,0] \
                        + interDu[eNIds[2],1]*DN[2,0] + interDu[eNIds[3],1]*DN[3,0]
        gradDuh[1,1] = interDu[eNIds[0],1]*DN[0,1] + interDu[eNIds[1],1]*DN[1,1] \
                        + interDu[eNIds[2],1]*DN[2,1] + interDu[eNIds[3],1]*DN[3,1]
        gradDuh[2,1] = interDu[eNIds[0],1]*DN[0,2] + interDu[eNIds[1],1]*DN[1,2] \
                        + interDu[eNIds[2],1]*DN[2,2] + interDu[eNIds[3],1]*DN[3,2]
        gradDuh[0,2] = interDu[eNIds[0],2]*DN[0,0] + interDu[eNIds[1],2]*DN[1,0] \
                        + interDu[eNIds[2],2]*DN[2,0] + interDu[eNIds[3],2]*DN[3,0]
        gradDuh[1,2] = interDu[eNIds[0],2]*DN[0,1] + interDu[eNIds[1],2]*DN[1,1] \
                        + interDu[eNIds[2],2]*DN[2,1] + interDu[eNIds[3],2]*DN[3,1]
        gradDuh[2,2] = interDu[eNIds[0],2]*DN[0,2] + interDu[eNIds[1],2]*DN[1,2] \
                        + interDu[eNIds[2],2]*DN[2,2] + interDu[eNIds[3],2]*DN[3,2]

        trGradDuh = gradDuh[0,0] + gradDuh[1,1] + gradDuh[2,2]

        # print "gradDuh ", np.asarray(gradDuh)
        # print "trGradDuh ", trGradDuh

        # gradPh
        gradPh[0] = interP[eNIds[0]]*DN[0,0] + interP[eNIds[1]]*DN[1,0] \
                    + interP[eNIds[2]]*DN[2,0] + interP[eNIds[3]]*DN[3,0]
        gradPh[1] = interP[eNIds[0]]*DN[0,1] + interP[eNIds[1]]*DN[1,1] \
                    + interP[eNIds[2]]*DN[2,1] + interP[eNIds[3]]*DN[3,1]
        gradPh[2] = interP[eNIds[0]]*DN[0,2] + interP[eNIds[1]]*DN[1,2] \
                    + interP[eNIds[2]]*DN[2,2] + interP[eNIds[3]]*DN[3,2]

        # print "gradPh ", np.asarray(gradPh)

        # Loop through gaussian points. nGp
        for iGp in range(nGp):

            wGp = w[iGp] * detJ

            # print "wGp ", wGp

            wr = wGp * rho
            wrl = wr * fgt
            wl = wGp * fgt

            duh[0] = interDu[eNIds[0],0]*N[iGp,0] + interDu[eNIds[1],0]*N[iGp,1] \
                        + interDu[eNIds[2],0]*N[iGp,2] + interDu[eNIds[3],0]*N[iGp,3]
            duh[1] = interDu[eNIds[0],1]*N[iGp,0] + interDu[eNIds[1],1]*N[iGp,1] \
                        + interDu[eNIds[2],1]*N[iGp,2] + interDu[eNIds[3],1]*N[iGp,3]
            duh[2] = interDu[eNIds[0],2]*N[iGp,0] + interDu[eNIds[1],2]*N[iGp,1] \
                        + interDu[eNIds[2],2]*N[iGp,2] + interDu[eNIds[3],2]*N[iGp,3]

            dduh[0] = interDDu[eNIds[0],0]*N[iGp,0] + interDDu[eNIds[1],0]*N[iGp,1] \
                        + interDDu[eNIds[2],0]*N[iGp,2] + interDDu[eNIds[3],0]*N[iGp,3]
            dduh[1] = interDDu[eNIds[0],1]*N[iGp,0] + interDDu[eNIds[1],1]*N[iGp,1] \
                        + interDDu[eNIds[2],1]*N[iGp,2] + interDDu[eNIds[3],1]*N[iGp,3]
            dduh[2] = interDDu[eNIds[0],2]*N[iGp,0] + interDDu[eNIds[1],2]*N[iGp,1] \
                        + interDDu[eNIds[2],2]*N[iGp,2] + interDDu[eNIds[3],2]*N[iGp,3]

            ph = interP[eNIds[0]]*N[iGp,0] + interP[eNIds[1]]*N[iGp,1] \
                        + interP[eNIds[2]]*N[iGp,2] + interP[eNIds[3]]*N[iGp,3]

            # print "duh ", np.asarray(duh)
            # print "dduh ", np.asarray(dduh)
            # print "ph ", ph

            # tau_SUPS  tauM
            duhGduh = duh[0]*duh[0]*G[0,0] + duh[0]*duh[1]*G[0,1] + duh[0]*duh[2]*G[0,2] \
                    + duh[1]*duh[0]*G[1,0] + duh[1]*duh[1]*G[1,1] + duh[1]*duh[2]*G[1,2] \
                    + duh[2]*duh[0]*G[2,0] + duh[2]*duh[1]*G[2,1] + duh[2]*duh[2]*G[2,2]
            tauM = 1.0 / sqrt(tauSP + duhGduh)

            # print "duhGduh ", duhGduh
            # print "tauM ", tauM

            # up
            up[0] = -tauM*(dduh[0] + gradPh[0]*c3 + duh[0]*gradDuh[0,0] + duh[1]*gradDuh[1,0] + duh[2]*gradDuh[2,0])
            up[1] = -tauM*(dduh[1] + gradPh[1]*c3 + duh[0]*gradDuh[0,1] + duh[1]*gradDuh[1,1] + duh[2]*gradDuh[2,1])
            up[2] = -tauM*(dduh[2] + gradPh[2]*c3 + duh[0]*gradDuh[0,2] + duh[1]*gradDuh[1,2] + duh[2]*gradDuh[2,2])

            # print "up ", np.asarray(up)

            # tauC  v_LSIC
            tauC = 1.0 / (trG * tauM * 16.0)

            # print "tauC ", tauC

            # tauB
            tauB = up[0]*up[0]*G[0,0] + up[0]*up[1]*G[0,1] + up[0]*up[2]*G[0,2] \
                 + up[1]*up[0]*G[1,0] + up[1]*up[1]*G[1,1] + up[1]*up[2]*G[1,2] \
                 + up[2]*up[0]*G[2,0] + up[2]*up[1]*G[2,1] + up[2]*up[2]*G[2,2]

            if iszero(tauB):
                tauB = eps
            tauB = 1.0 / sqrt(tauB)

            # print "tauB ", tauB

            # u + up
            ua[0] = duh[0] + up[0]
            ua[1] = duh[1] + up[1]
            ua[2] = duh[2] + up[2]

            # print "ua ", np.asarray(ua)

            # for Rm
            rV[0] = tauB*(up[0]*gradDuh[0,0] + up[1]*gradDuh[1,0] + up[2]*gradDuh[2,0])
            rV[1] = tauB*(up[0]*gradDuh[0,1] + up[1]*gradDuh[1,1] + up[2]*gradDuh[2,1])
            rV[2] = tauB*(up[0]*gradDuh[0,2] + up[1]*gradDuh[1,2] + up[2]*gradDuh[2,2])

            # print "rV ", np.asarray(rV)

            T1 = tauC*trGradDuh - ph*c3

            rM[0,0] = nu*(gradDuh[0,0]+gradDuh[0,0]) - up[0]*ua[0] + rV[0]*up[0] + T1
            rM[1,0] = nu*(gradDuh[0,1]+gradDuh[1,0]) - up[1]*ua[0] + rV[1]*up[0]
            rM[2,0] = nu*(gradDuh[0,2]+gradDuh[2,0]) - up[2]*ua[0] + rV[2]*up[0]

            rM[0,1] = nu*(gradDuh[1,0]+gradDuh[0,1]) - up[0]*ua[1] + rV[0]*up[1]
            rM[1,1] = nu*(gradDuh[1,1]+gradDuh[1,1]) - up[1]*ua[1] + rV[1]*up[1] + T1
            rM[2,1] = nu*(gradDuh[1,2]+gradDuh[2,1]) - up[2]*ua[1] + rV[2]*up[1]

            rM[0,2] = nu*(gradDuh[2,0]+gradDuh[0,2]) - up[0]*ua[2] + rV[0]*up[2]
            rM[1,2] = nu*(gradDuh[2,1]+gradDuh[1,2]) - up[1]*ua[2] + rV[1]*up[2]
            rM[2,2] = nu*(gradDuh[2,2]+gradDuh[2,2]) - up[2]*ua[2] + rV[2]*up[2] + T1

            # print "rM ", np.asarray(rM)

            # for Rm
            rV[0] = dduh[0] + ua[0]*gradDuh[0,0] + ua[1]*gradDuh[1,0] + ua[2]*gradDuh[2,0]
            rV[1] = dduh[1] + ua[0]*gradDuh[0,1] + ua[1]*gradDuh[1,1] + ua[2]*gradDuh[2,1]
            rV[2] = dduh[2] + ua[0]*gradDuh[0,2] + ua[1]*gradDuh[1,2] + ua[2]*gradDuh[2,2]

            # print "rV ", np.asarray(rV)

            # # for Rc
            # duhDN[0] = duh[0]*DN[0,0] + duh[1]*DN[0,1] + duh[2]*DN[0,2]
            # duhDN[1] = duh[0]*DN[1,0] + duh[1]*DN[1,1] + duh[2]*DN[1,2]
            # duhDN[2] = duh[0]*DN[2,0] + duh[1]*DN[2,1] + duh[2]*DN[2,2]
            # duhDN[3] = duh[0]*DN[3,0] + duh[1]*DN[3,1] + duh[2]*DN[3,2]

            # upDN[0] = up[0]*DN[0,0] + up[1]*DN[0,1] + up[2]*DN[0,2]
            # upDN[1] = up[0]*DN[1,0] + up[1]*DN[1,1] + up[2]*DN[1,2]
            # upDN[2] = up[0]*DN[2,0] + up[1]*DN[2,1] + up[2]*DN[2,2]
            # upDN[3] = up[0]*DN[3,0] + up[1]*DN[3,1] + up[2]*DN[3,2]

            # uaDN[0] = duhDN[0] + upDN[0]
            # uaDN[1] = duhDN[1] + upDN[1]
            # uaDN[2] = duhDN[2] + upDN[2]
            # uaDN[3] = duhDN[3] + upDN[3]

            for a in range(nPts):

                duhDN[a] = duh[0]*DN[a,0] + duh[1]*DN[a,1] + duh[2]*DN[a,2]
                upDN[a] = up[0]*DN[a,0] + up[1]*DN[a,1] + up[2]*DN[a,2]
                uaDN[a] = duhDN[a] + upDN[a]

                lR[0,a] += wr*(rV[0]*N[iGp,a] + rM[0,0]*DN[a,0] + rM[0,1]*DN[a,1] + rM[0,2]*DN[a,2])
                lR[1,a] += wr*(rV[1]*N[iGp,a] + rM[1,0]*DN[a,0] + rM[1,1]*DN[a,1] + rM[1,2]*DN[a,2])
                lR[2,a] += wr*(rV[2]*N[iGp,a] + rM[2,0]*DN[a,0] + rM[2,1]*DN[a,1] + rM[2,2]*DN[a,2])
                lR[3,a] += wGp*(trGradDuh*N[iGp,a] - upDN[a])

            for a in range(nPts):
                for b in range(nPts):

                    rM[0,0] = DN[b,0]*DN[a,0]
                    rM[0,1] = DN[b,0]*DN[a,1]
                    rM[0,2] = DN[b,0]*DN[a,2]
                    rM[1,0] = DN[b,1]*DN[a,0]
                    rM[1,1] = DN[b,1]*DN[a,1]
                    rM[1,2] = DN[b,1]*DN[a,2]
                    rM[2,0] = DN[b,2]*DN[a,0]
                    rM[2,1] = DN[b,2]*DN[a,1]
                    rM[2,2] = DN[b,2]*DN[a,2]

                    # print "rM ", np.asarray(rM)

                    DNDN = DN[a,0]*DN[b,0] + DN[a,1]*DN[b,1] + DN[a,2]*DN[b,2]

                    # print "DNDN ", np.asarray(DNDN)

                    T1 = nu*DNDN + tauB*upDN[a]*upDN[b] \
                        + N[iGp,a]*(mdfgt*N[iGp,b] + uaDN[b]) \
                        + tauM*uaDN[a]*(mdfgt*N[iGp,b] + duhDN[b])
                    T2 = tauM*duhDN[a]
                    T3 = tauM*(mdfgt*N[iGp,b] + duhDN[b])

                    # print "T123 ", T1, T2, T3
                    # print "Debug: nu ", nu, " mdfgt ", mdfgt
                    # print "Debug: upDN ", np.asarray(upDN)
                    # print "Debug: duhDN ", np.asarray(duhDN)
                    # print "Debug: uaDN ", np.asarray(uaDN)

                    # K  dM/dU
                    lLHS[0,a,b] += wrl*((nu + tauC)*rM[0,0] + T1)
                    lLHS[1,a,b] += wrl*(nu*rM[0,1] + tauC*rM[1,0])
                    lLHS[2,a,b] += wrl*(nu*rM[0,2] + tauC*rM[2,0])

                    lLHS[4,a,b] += wrl*(nu*rM[1,0] + tauC*rM[0,1])
                    lLHS[5,a,b] += wrl*((nu + tauC)*rM[1,1] + T1)
                    lLHS[6,a,b] += wrl*(nu*rM[1,2] + tauC*rM[2,1])

                    lLHS[8,a,b] += wrl*(nu*rM[2,0] + tauC*rM[0,2])
                    lLHS[9,a,b] += wrl*(nu*rM[2,1] + tauC*rM[1,2])
                    lLHS[10,a,b] += wrl*((nu + tauC)*rM[2,2] + T1)

                    # G  dM/dP
                    lLHS[3,a,b] -= wl*(DN[a,0]*N[iGp,b] - DN[b,0]*T2)
                    lLHS[7,a,b] -= wl*(DN[a,1]*N[iGp,b] - DN[b,1]*T2)
                    lLHS[11,a,b] -= wl*(DN[a,2]*N[iGp,b] - DN[b,2]*T2)

                    # D  dC/dU
                    lLHS[12,a,b] += wl*(N[iGp,a]*DN[b,0] + DN[a,0]*T3)
                    lLHS[13,a,b] += wl*(N[iGp,a]*DN[b,1] + DN[a,1]*T3)
                    lLHS[14,a,b] += wl*(N[iGp,a]*DN[b,2] + DN[a,2]*T3)

                    # L
                    lLHS[15,a,b] += wl*tauM*DNDN*c3

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

