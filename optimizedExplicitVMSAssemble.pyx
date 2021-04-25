# cython: language_level=3, boundscheck=False
# include math method from C libs.
import numpy as np

cimport numpy as np
from libc.math cimport sqrt
cimport cython


# jacobian = [[x[1]-x[0], x[2]-x[0], x[3]-x[0]],
#             [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
#             [z[1]-z[0], z[2]-z[0], z[3]-z[0]]]
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double getGlbDerivatives(
    double[:,::1] nodes, long[::1] eNIds, double[:,::1] lDN, double[:,::1] DN,
    double[:,::1] jac, double[:,::1] cof, double[:,::1] invJac):

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

    return detJ / 6.0


# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double inverseM(double[:,::1] lM, double[:,::1] cofLM, double[:,::1] invLM):

#     cdef double detM, iDetM

#     # +0,0  -0,1  +0,2
#     # -1,0  +1,1  -1,2
#     # +2,0  -2,1  +2,2
#     cofLM[0,0] = lM[1,1]*lM[2,2] - lM[1,2]*lM[2,1]
#     cofLM[0,1] = lM[2,0]*lM[1,2] - lM[1,0]*lM[2,2]
#     cofLM[0,2] = lM[1,0]*lM[2,1] - lM[1,1]*lM[2,0]
#     cofLM[1,0] = lM[2,1]*lM[0,2] - lM[0,1]*lM[2,2]
#     cofLM[1,1] = lM[0,0]*lM[2,2] - lM[2,0]*lM[0,2]
#     cofLM[1,2] = lM[2,0]*lM[0,1] - lM[0,0]*lM[2,1]
#     cofLM[2,0] = lM[0,1]*lM[1,2] - lM[1,1]*lM[0,2]
#     cofLM[2,1] = lM[1,0]*lM[0,2] - lM[0,0]*lM[1,2]
#     cofLM[2,2] = lM[0,0]*lM[1,1] - lM[1,0]*lM[0,1]

#     detM = lM[0,0]*cofLM[0,0] + lM[0,1]*cofLM[0,1] + lM[0,2]*cofLM[0,2]
#     iDetM = 1.0 / detM

#     invLM[0,0] = cofLM[0,0] * iDetM
#     invLM[0,1] = cofLM[1,0] * iDetM
#     invLM[0,2] = cofLM[2,0] * iDetM
#     invLM[1,0] = cofLM[0,1] * iDetM
#     invLM[1,1] = cofLM[1,1] * iDetM
#     invLM[1,2] = cofLM[2,1] * iDetM
#     invLM[2,0] = cofLM[0,2] * iDetM
#     invLM[2,1] = cofLM[1,2] * iDetM
#     invLM[2,2] = cofLM[2,2] * iDetM



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void initialAssembling(long[::1] eNIds, double[:,::1] lM, double[:,::1] LHS):
    
    cdef int nPts = eNIds.shape[0]
    cdef int nDofs = 4
    cdef int a, b, k

    for a in range(nPts):
        for b in range(nPts):
            for k in range(nDofs):
                LHS[nDofs*eNIds[a]+k,nDofs*eNIds[b]+k] += lM[nDofs*a+k,nDofs*b+k]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedExplicitVMSInitialAssemble(double[:,::1] nodes, long[:,::1] elements,
    double[::1] w, double[:,::1] lN, double[:,::1] lDN,
    double[:,:,::1] DNs, double[::1] volumes, double[:,::1] LHS, double[:,:,::1] lMs):

    cdef long nElm = elements.shape[0]
    cdef long nPts = 4
    cdef long nDim = 3
    cdef long nGp = 4 # w.shape[0]
    cdef long nElmDofs = 16 # nPts*(nV+nP)

    cdef double Ve, wGp
    cdef double tmpM

    cdef long eNIds[4]
    cdef double[:,::1] DN = np.empty((nDim, nPts), dtype=np.float)

    cdef double[:,::1] lM = np.zeros((nElmDofs, nElmDofs), dtype=np.float)

    cdef double jac[3][3]
    cdef double invJac[3][3]
    cdef double cof[3][3]

    cdef long iElm, iGp
    cdef long i, j, k
    cdef long a, b

    for iElm in range(nElm):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for i in range(nPts):
            for j in range(nPts):
                for k in range(4):
                    lM[4*i+k,4*j+k] = 0.0 # only clear the value spot

        # Get the global derivatives and the volume of the tetrahedron element
        Ve = getGlbDerivatives(nodes, eNIds, lDN, DN, jac, cof, invJac)

        for i in range(nDim):
            for j in range(nPts):
                DNs[iElm,i,j] = DN[i,j]
        
        volumes[iElm] = Ve

        for iGp in range(nGp):
            wGp = w[iGp] * Ve

            for a in range(nPts):
                for b in range(nPts):
                    tmpM = lN[iGp,a] * lN[iGp,b] * wGp
                    for k in range(4):
                        lM[4*a+k,4*b+k] += tmpM
                    for k in range(3):
                        lMs[iElm,3*a+k,3*b+k] += tmpM

        initialAssembling(eNIds, lM, LHS)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void matrixVecMltp(double[:,:,::1] invLM, double[:,::1] R, double[:,:,::1] subValue, long iElm):

    cdef int a, i, j

    for a in range(4):
        for i in range(4):
            for j in range(3):
                subValue[iElm,a,0] += invLM[iElm,3*a,3*i+j]*R[i,j]
                subValue[iElm,a,1] += invLM[iElm,3*a+1,3*i+j]*R[i,j]
                subValue[iElm,a,2] += invLM[iElm,3*a+2,3*i+j]*R[i,j]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assembling(long[::1] eNIds, double[:,::1] lRHS, double[:,::1] lRes, double[::1] RHS, double[::1] Res,
                     double[:,::1] lMResT1, double[:,::1] lMResT2, double[:,::1] lMResT3, double[:,::1] lMResT4, double[:,::1] lMResT5,
                     double[::1] lPResT1, double[::1] lPResT2,
                     double[::1] mRT1, double[::1] mRT2, double[::1] mRT3, double[::1] mRT4, double[::1] mRT5,
                     double[::1] pRT1, double[::1] pRT2):

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef void assembling(long[::1] eNIds, double[:,::1] lRHS, double[:,::1] lRes,
#                      double[::1] RHS, double[::1] Res):
    
    cdef int nPts = eNIds.shape[0]
    cdef int a, b

    for a in range(nPts):
        for b in range(4): # Dof
            RHS[eNIds[a]*4+b] += lRHS[a,b]
            Res[eNIds[a]*4+b] += lRes[a,b]

        # only for debugging
        for b in range(3):
            mRT1[eNIds[a]*3+b] += lMResT1[a,b]
            mRT2[eNIds[a]*3+b] += lMResT2[a,b]
            mRT3[eNIds[a]*3+b] += lMResT3[a,b]
            mRT4[eNIds[a]*3+b] += lMResT4[a,b]
            mRT5[eNIds[a]*3+b] += lMResT5[a,b]
        
        pRT1[eNIds[a]] += lPResT1[a]
        pRT2[eNIds[a]] += lPResT2[a]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedExplicitVMSAssemble(
    double[:,::1] nodes, long[:,::1] elements,
    double[:,::1] du, double[::1] p, double[:,::1] hdu, double[::1] hp,
    double[:,:,::1] sdu, double[:,:,::1] nsdu, double[:,::1] f, double[::1] hs,
    double[::1] w, double[:,::1] lN, double[:,:,::1] DNs, double[::1] volumes,
    double[:,:,::1] invLMs, double[::1] coefs, double[::1] RHS, double[::1] R,
    double[::1] mRT1, double[::1] mRT2, double[::1] mRT3, double[::1] mRT4, double[::1] mRT5,
    double[::1] pRT1, double[::1] pRT2):

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def OptimizedExplicitVMSAssemble(
#     double[:,::1] nodes, long[:,::1] elements,
#     double[:,::1] du, double[::1] p, double[:,::1] hdu, double[::1] hp,
#     double[:,:,::1] sdu, double[:,:,::1] nsdu,
#     double[:,::1] f, double[::1] hs,
#     double[::1] w, double[:,::1] lN, double[:,:,::1] DNs, double[::1] coefs,
#     double[::1] RHS, double[::1] R):

    cdef long nElms = elements.shape[0]

    cdef long nPts = 4 # elements.shape[1]
    cdef long ndim = 3 # nodes.shape[1]

    # Explicit VMS solver parameters
    cdef double c1 = coefs[0]
    cdef double c2 = coefs[1]
    cdef double nu = coefs[2]
    cdef double dt = coefs[3]
    cdef double invEpsilon = coefs[4]
    cdef double h = 0.0

    cdef double wGp
    cdef double trGradU, trGradHu
    cdef double varT1, varT2
    cdef double ph, hph
    
    cdef long[::1] eNIds = np.empty(nPts, dtype=long)

    cdef double[:,::1] av = np.zeros((nPts, ndim), dtype=np.float)
    cdef double[::1] ah = np.zeros(ndim, dtype=np.float)
    cdef double[::1] duh = np.zeros(ndim, dtype=np.float)
    cdef double[::1] sduh = np.empty(ndim, dtype=np.float)
    cdef double[::1] fh = np.empty(ndim, dtype=np.float)

    cdef double[:,::1] gradU = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] gradP = np.empty(ndim, dtype=np.float)
    cdef double[:,::1] gradHu = np.empty((ndim, ndim), dtype=np.float)
    # cdef double[:,::1] symGradHu = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] gradHp = np.empty(ndim, dtype=np.float)

    cdef double[::1] ahGradHu = np.empty(ndim, dtype=np.float)
    cdef double[::1] T1 = np.empty(ndim, dtype=np.float)

    cdef double[:,::1] lRHS = np.empty((nPts, 4), dtype=np.float)
    cdef double[:,::1] lRes = np.empty((nPts, 4), dtype=np.float)
    
    cdef double max_norm_av, tau_u_inv, tau_t, Ru
    cdef double[:,::1] atau = np.zeros((nPts, ndim), dtype=np.float)
    cdef double[::1] norm_atau = np.zeros(nPts, dtype=np.float)
    cdef double[::1] avGradU = np.empty(ndim, dtype=np.float)
    cdef double[::1] subgridF = np.zeros(ndim, dtype=np.float)
    cdef double[:,::1] lNsdu = np.zeros((nPts, 3), dtype=np.float)

    # for debugging only
    cdef double[:,::1] lMomentumResT1 = np.empty((nPts, 3), dtype=np.float)
    cdef double[:,::1] lMomentumResT2 = np.empty((nPts, 3), dtype=np.float)
    cdef double[:,::1] lMomentumResT3 = np.empty((nPts, 3), dtype=np.float)
    cdef double[:,::1] lMomentumResT4 = np.empty((nPts, 3), dtype=np.float)
    cdef double[:,::1] lMomentumResT5 = np.empty((nPts, 3), dtype=np.float)
    cdef double[::1] lPressureResT1 = np.empty(nPts, dtype=np.float)
    cdef double[::1] lPressureResT2 = np.empty(nPts, dtype=np.float)

    cdef long iElm
    cdef long nGp = 4 # w.shape[0]
    cdef long iGp
    cdef int i, j, k, a, b

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for i in range(nPts):
            lNsdu[i,0] = 0.0
            lNsdu[i,1] = 0.0
            lNsdu[i,2] = 0.0

            lPressureResT1[i] = 0.0
            lPressureResT2[i] = 0.0

            for j in range(4):
                lRHS[i,j] = 0.0
                lRes[i,j] = 0.0
                
                lMomentumResT1[i,j] = 0.0
                lMomentumResT2[i,j] = 0.0
                lMomentumResT3[i,j] = 0.0
                lMomentumResT4[i,j] = 0.0
                lMomentumResT5[i,j] = 0.0

        # gradU
        gradU[0,0] = du[eNIds[0],0]*DNs[iElm,0,0] + du[eNIds[1],0]*DNs[iElm,0,1] + du[eNIds[2],0]*DNs[iElm,0,2] + du[eNIds[3],0]*DNs[iElm,0,3]
        gradU[0,1] = du[eNIds[0],0]*DNs[iElm,1,0] + du[eNIds[1],0]*DNs[iElm,1,1] + du[eNIds[2],0]*DNs[iElm,1,2] + du[eNIds[3],0]*DNs[iElm,1,3]
        gradU[0,2] = du[eNIds[0],0]*DNs[iElm,2,0] + du[eNIds[1],0]*DNs[iElm,2,1] + du[eNIds[2],0]*DNs[iElm,2,2] + du[eNIds[3],0]*DNs[iElm,2,3]
        gradU[1,0] = du[eNIds[0],1]*DNs[iElm,0,0] + du[eNIds[1],1]*DNs[iElm,0,1] + du[eNIds[2],1]*DNs[iElm,0,2] + du[eNIds[3],1]*DNs[iElm,0,3]
        gradU[1,1] = du[eNIds[0],1]*DNs[iElm,1,0] + du[eNIds[1],1]*DNs[iElm,1,1] + du[eNIds[2],1]*DNs[iElm,1,2] + du[eNIds[3],1]*DNs[iElm,1,3]
        gradU[1,2] = du[eNIds[0],1]*DNs[iElm,2,0] + du[eNIds[1],1]*DNs[iElm,2,1] + du[eNIds[2],1]*DNs[iElm,2,2] + du[eNIds[3],1]*DNs[iElm,2,3]
        gradU[2,0] = du[eNIds[0],2]*DNs[iElm,0,0] + du[eNIds[1],2]*DNs[iElm,0,1] + du[eNIds[2],2]*DNs[iElm,0,2] + du[eNIds[3],2]*DNs[iElm,0,3]
        gradU[2,1] = du[eNIds[0],2]*DNs[iElm,1,0] + du[eNIds[1],2]*DNs[iElm,1,1] + du[eNIds[2],2]*DNs[iElm,1,2] + du[eNIds[3],2]*DNs[iElm,1,3]
        gradU[2,2] = du[eNIds[0],2]*DNs[iElm,2,0] + du[eNIds[1],2]*DNs[iElm,2,1] + du[eNIds[2],2]*DNs[iElm,2,2] + du[eNIds[3],2]*DNs[iElm,2,3]

        trGradU = gradU[0,0] + gradU[1,1] + gradU[2,2]

        # gradP
        gradP[0] = p[eNIds[0]]*DNs[iElm,0,0] + p[eNIds[1]]*DNs[iElm,0,1] + p[eNIds[2]]*DNs[iElm,0,2] + p[eNIds[3]]*DNs[iElm,0,3]
        gradP[1] = p[eNIds[0]]*DNs[iElm,1,0] + p[eNIds[1]]*DNs[iElm,1,1] + p[eNIds[2]]*DNs[iElm,1,2] + p[eNIds[3]]*DNs[iElm,1,3]
        gradP[2] = p[eNIds[0]]*DNs[iElm,2,0] + p[eNIds[1]]*DNs[iElm,2,1] + p[eNIds[2]]*DNs[iElm,2,2] + p[eNIds[3]]*DNs[iElm,2,3]

        # gradHu
        gradHu[0,0] = hdu[eNIds[0],0]*DNs[iElm,0,0] + hdu[eNIds[1],0]*DNs[iElm,0,1] + hdu[eNIds[2],0]*DNs[iElm,0,2] + hdu[eNIds[3],0]*DNs[iElm,0,3]
        gradHu[0,1] = hdu[eNIds[0],0]*DNs[iElm,1,0] + hdu[eNIds[1],0]*DNs[iElm,1,1] + hdu[eNIds[2],0]*DNs[iElm,1,2] + hdu[eNIds[3],0]*DNs[iElm,1,3]
        gradHu[0,2] = hdu[eNIds[0],0]*DNs[iElm,2,0] + hdu[eNIds[1],0]*DNs[iElm,2,1] + hdu[eNIds[2],0]*DNs[iElm,2,2] + hdu[eNIds[3],0]*DNs[iElm,2,3]
        gradHu[1,0] = hdu[eNIds[0],1]*DNs[iElm,0,0] + hdu[eNIds[1],1]*DNs[iElm,0,1] + hdu[eNIds[2],1]*DNs[iElm,0,2] + hdu[eNIds[3],1]*DNs[iElm,0,3]
        gradHu[1,1] = hdu[eNIds[0],1]*DNs[iElm,1,0] + hdu[eNIds[1],1]*DNs[iElm,1,1] + hdu[eNIds[2],1]*DNs[iElm,1,2] + hdu[eNIds[3],1]*DNs[iElm,1,3]
        gradHu[1,2] = hdu[eNIds[0],1]*DNs[iElm,2,0] + hdu[eNIds[1],1]*DNs[iElm,2,1] + hdu[eNIds[2],1]*DNs[iElm,2,2] + hdu[eNIds[3],1]*DNs[iElm,2,3]
        gradHu[2,0] = hdu[eNIds[0],2]*DNs[iElm,0,0] + hdu[eNIds[1],2]*DNs[iElm,0,1] + hdu[eNIds[2],2]*DNs[iElm,0,2] + hdu[eNIds[3],2]*DNs[iElm,0,3]
        gradHu[2,1] = hdu[eNIds[0],2]*DNs[iElm,1,0] + hdu[eNIds[1],2]*DNs[iElm,1,1] + hdu[eNIds[2],2]*DNs[iElm,1,2] + hdu[eNIds[3],2]*DNs[iElm,1,3]
        gradHu[2,2] = hdu[eNIds[0],2]*DNs[iElm,2,0] + hdu[eNIds[1],2]*DNs[iElm,2,1] + hdu[eNIds[2],2]*DNs[iElm,2,2] + hdu[eNIds[3],2]*DNs[iElm,2,3]

        trGradHu = gradHu[0,0] + gradHu[1,1] + gradHu[2,2]

        # for i in range(3):
        #     for j in range(3):
        #         symGradHu[i,j] = gradHu[i,j] + gradHu[j,i]

        # gradHp
        gradHp[0] = hp[eNIds[0]]*DNs[iElm,0,0] + hp[eNIds[1]]*DNs[iElm,0,1] + hp[eNIds[2]]*DNs[iElm,0,2] + hp[eNIds[3]]*DNs[iElm,0,3]
        gradHp[1] = hp[eNIds[0]]*DNs[iElm,1,0] + hp[eNIds[1]]*DNs[iElm,1,1] + hp[eNIds[2]]*DNs[iElm,1,2] + hp[eNIds[3]]*DNs[iElm,1,3]
        gradHp[2] = hp[eNIds[0]]*DNs[iElm,2,0] + hp[eNIds[1]]*DNs[iElm,2,1] + hp[eNIds[2]]*DNs[iElm,2,2] + hp[eNIds[3]]*DNs[iElm,2,3]

        
        # Calc a elementwise
        h = hs[iElm]
        for i in range(nPts):
            for j in range(ndim):
                av[i,j] = hdu[eNIds[i],j] + sdu[iElm,i,j]
                atau[i,j] = du[eNIds[i],j] + sdu[iElm,i,j]
            # norm_av[i] = sqrt(av[i,0]**2 + av[i,1]**2 + av[i,2]**2)
            norm_atau[i] = sqrt(atau[i,0]**2 + atau[i,1]**2 + atau[i,2]**2)
        
        max_norm_av = max(norm_atau)
        tau_u_inv = c1*nu/(h**2) + c2*max_norm_av/h
        tau_t = 1.0 / (1.0/dt + tau_u_inv)

        # Evaluate velocity sub-grid scales
        for i in range(nPts):
            for j in range(ndim):
                Ru = tau_t*(gradU[j,0]*av[i,0] + gradU[j,1]*av[i,1] + gradU[j,2]*av[i,2] + gradP[j])
                nsdu[iElm,i,j] = sdu[iElm,i,j]*tau_t/dt - Ru

        # for i in range(nPts):
        #     avGradU[0] = gradU[0,0]*av[i,0] + gradU[0,1]*av[i,1] + gradU[0,2]*av[i,2] + gradP[0]
        #     avGradU[1] = gradU[1,0]*av[i,0] + gradU[1,1]*av[i,1] + gradU[1,2]*av[i,2] + gradP[1]
        #     avGradU[2] = gradU[2,0]*av[i,0] + gradU[2,1]*av[i,1] + gradU[2,2]*av[i,2] + gradP[2]
        #     for j in range(ndim):
        #         nsdu[iElm,i,j] = sdu[iElm,i,j]*tau_t/dt - tau_t*avGradU[j]

        
        # Loop through Gaussian integration points and assemble
        for iGp in range(nGp):
            wGp = w[iGp] * volumes[iElm]

            # ah at Gaussion point iGp
            ah[0] = av[0,0]*lN[iGp,0] + av[1,0]*lN[iGp,1] + av[2,0]*lN[iGp,2] + av[3,0]*lN[iGp,3]
            ah[1] = av[0,1]*lN[iGp,0] + av[1,1]*lN[iGp,1] + av[2,1]*lN[iGp,2] + av[3,1]*lN[iGp,3]
            ah[2] = av[0,2]*lN[iGp,0] + av[1,2]*lN[iGp,1] + av[2,2]*lN[iGp,2] + av[3,2]*lN[iGp,3]

            # Evaluate velocity sub-grid scales
            subgridF[0] = gradU[0,0]*ah[0] + gradU[0,1]*ah[1] + gradU[0,2]*ah[2] + gradP[0]
            subgridF[1] = gradU[1,0]*ah[0] + gradU[1,1]*ah[1] + gradU[1,2]*ah[2] + gradP[1]
            subgridF[2] = gradU[2,0]*ah[0] + gradU[2,1]*ah[1] + gradU[2,2]*ah[2] + gradP[2]

            for i in range(nPts):
                lNsdu[i,0] += tau_t*subgridF[0]*lN[iGp,i]*wGp
                lNsdu[i,1] += tau_t*subgridF[1]*lN[iGp,i]*wGp
                lNsdu[i,2] += tau_t*subgridF[2]*lN[iGp,i]*wGp

            # # lLHS
            # for a in range(nPts):
            #     for b in range(nPts):
            #         lLHS[a] += lN[iGp,a]*lN[iGp,b]*wGp

            # duh at Gaussian point iGp
            duh[0] = du[eNIds[0],0]*lN[iGp,0] + du[eNIds[1],0]*lN[iGp,1] + du[eNIds[2],0]*lN[iGp,2] + du[eNIds[3],0]*lN[iGp,3]                    
            duh[1] = du[eNIds[0],1]*lN[iGp,0] + du[eNIds[1],1]*lN[iGp,1] + du[eNIds[2],1]*lN[iGp,2] + du[eNIds[3],1]*lN[iGp,3]
            duh[2] = du[eNIds[0],2]*lN[iGp,0] + du[eNIds[1],2]*lN[iGp,1] + du[eNIds[2],2]*lN[iGp,2] + du[eNIds[3],2]*lN[iGp,3]
            
            # fh at Gaussian point iGp
            fh[0] = f[eNIds[0],0]*lN[iGp,0] + f[eNIds[1],0]*lN[iGp,1] + f[eNIds[2],0]*lN[iGp,2] + f[eNIds[3],0]*lN[iGp,3]                    
            fh[1] = f[eNIds[0],1]*lN[iGp,0] + f[eNIds[1],1]*lN[iGp,1] + f[eNIds[2],1]*lN[iGp,2] + f[eNIds[3],1]*lN[iGp,3]
            fh[2] = f[eNIds[0],2]*lN[iGp,0] + f[eNIds[1],2]*lN[iGp,1] + f[eNIds[2],2]*lN[iGp,2] + f[eNIds[3],2]*lN[iGp,3]
            
            # ph at Gaussian point iGp
            ph = p[eNIds[0]]*lN[iGp,0] + p[eNIds[1]]*lN[iGp,1] + p[eNIds[2]]*lN[iGp,2] + p[eNIds[3]]*lN[iGp,3]
            
            # hph at Gaussian point iGp
            hph = hp[eNIds[0]]*lN[iGp,0] + hp[eNIds[1]]*lN[iGp,1] + hp[eNIds[2]]*lN[iGp,2] + hp[eNIds[3]]*lN[iGp,3]
            
            # sduh at Gaussian point iGp
            sduh[0] = sdu[iElm,0,0]*lN[iGp,0] + sdu[iElm,1,0]*lN[iGp,1] + sdu[iElm,2,0]*lN[iGp,2] + sdu[iElm,3,0]*lN[iGp,3]                    
            sduh[1] = sdu[iElm,0,1]*lN[iGp,0] + sdu[iElm,1,1]*lN[iGp,1] + sdu[iElm,2,1]*lN[iGp,2] + sdu[iElm,3,1]*lN[iGp,3]
            sduh[2] = sdu[iElm,0,2]*lN[iGp,0] + sdu[iElm,1,2]*lN[iGp,1] + sdu[iElm,2,2]*lN[iGp,2] + sdu[iElm,3,2]*lN[iGp,3]
            
            # ah dot GradHu
            ahGradHu[0] = ah[0]*gradHu[0,0] + ah[1]*gradHu[0,1] + ah[2]*gradHu[0,2]
            ahGradHu[1] = ah[0]*gradHu[1,0] + ah[1]*gradHu[1,1] + ah[2]*gradHu[1,2]
            ahGradHu[2] = ah[0]*gradHu[2,0] + ah[1]*gradHu[2,1] + ah[2]*gradHu[2,2]

            # lRHS
            for a in range(nPts):
                varT1 = ah[0]*DNs[iElm,0,a] + ah[1]*DNs[iElm,1,a] + ah[2]*DNs[iElm,2,a]
                varT2 = sduh[0]*DNs[iElm,0,a] + sduh[1]*DNs[iElm,1,a] + sduh[2]*DNs[iElm,2,a]

                T1[0] = nu*(gradHu[0,0]*DNs[iElm,0,a]+gradHu[0,1]*DNs[iElm,1,a]+gradHu[0,2]*DNs[iElm,2,a]) \
                        - hph*DNs[iElm,0,a] - sduh[0]*varT1
                T1[1] = nu*(gradHu[1,0]*DNs[iElm,0,a]+gradHu[1,1]*DNs[iElm,1,a]+gradHu[1,2]*DNs[iElm,2,a]) \
                        - hph*DNs[iElm,1,a] - sduh[1]*varT1
                T1[2] = nu*(gradHu[2,0]*DNs[iElm,0,a]+gradHu[2,1]*DNs[iElm,1,a]+gradHu[2,2]*DNs[iElm,2,a]) \
                        - hph*DNs[iElm,2,a] - sduh[2]*varT1

                lRHS[a,0] += duh[0]*lN[iGp,a]*wGp
                lRHS[a,1] += duh[1]*lN[iGp,a]*wGp
                lRHS[a,2] += duh[2]*lN[iGp,a]*wGp
                lRHS[a,3] += ph*lN[iGp,a]*wGp

                lRes[a,0] += wGp*((ahGradHu[0]-fh[0])*lN[iGp,a] + T1[0])
                lRes[a,1] += wGp*((ahGradHu[1]-fh[1])*lN[iGp,a] + T1[1])
                lRes[a,2] += wGp*((ahGradHu[2]-fh[2])*lN[iGp,a] + T1[2])
                lRes[a,3] += wGp*(trGradHu*lN[iGp,a]-varT2)*invEpsilon

                # only for debugging
                lMomentumResT1[a,0] += wGp*ahGradHu[0]*lN[iGp,a]
                lMomentumResT1[a,1] += wGp*ahGradHu[1]*lN[iGp,a]
                lMomentumResT1[a,2] += wGp*ahGradHu[2]*lN[iGp,a]

                lMomentumResT2[a,0] += wGp*nu*(gradHu[0,0]*DNs[iElm,0,a]+gradHu[0,1]*DNs[iElm,1,a]+gradHu[0,2]*DNs[iElm,2,a])
                lMomentumResT2[a,1] += wGp*nu*(gradHu[1,0]*DNs[iElm,0,a]+gradHu[1,1]*DNs[iElm,1,a]+gradHu[1,2]*DNs[iElm,2,a])
                lMomentumResT2[a,2] += wGp*nu*(gradHu[2,0]*DNs[iElm,0,a]+gradHu[2,1]*DNs[iElm,1,a]+gradHu[2,2]*DNs[iElm,2,a])

                lMomentumResT3[a,0] -= wGp*(hph)*DNs[iElm,0,a]
                lMomentumResT3[a,1] -= wGp*(hph)*DNs[iElm,1,a]
                lMomentumResT3[a,2] -= wGp*(hph)*DNs[iElm,2,a]

                lMomentumResT4[a,0] -= wGp*sduh[0]*varT1
                lMomentumResT4[a,1] -= wGp*sduh[1]*varT1
                lMomentumResT4[a,2] -= wGp*sduh[2]*varT1

                lMomentumResT5[a,0] -= wGp*fh[0]*lN[iGp,a]
                lMomentumResT5[a,1] -= wGp*fh[1]*lN[iGp,a]
                lMomentumResT5[a,2] -= wGp*fh[2]*lN[iGp,a]

                lPressureResT1[a] += wGp*trGradHu*lN[iGp,a]*invEpsilon
                lPressureResT2[a] -= wGp*varT2*invEpsilon

        # Update the u_subscale for next timestep.
        # for a in range(nPts):
        #     nsdu[iElm,a,0] += lNsdu[a,0] / lLHS[a]
        #     nsdu[iElm,a,1] += lNsdu[a,1] / lLHS[a]
        #     nsdu[iElm,a,2] += lNsdu[a,2] / lLHS[a]

        matrixVecMltp(invLMs, lNsdu, nsdu, iElm)

        # Assembling
        # assembling(eNIds, lRHS, lRes, RHS, R)

        # only for debugging
        assembling(eNIds, lRHS, lRes, RHS, R,
            lMomentumResT1, lMomentumResT2, lMomentumResT3, lMomentumResT4, lMomentumResT5,
            lPressureResT1, lPressureResT2,
            mRT1, mRT2, mRT3, mRT4, mRT5, pRT1, pRT2)



