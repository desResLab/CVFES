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

    return detJ / 6.0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void assembling(long[::1] eNIds, double[::1] lLHS, double[:,::1] lR,
                     double[::1] LHS, double[::1] RHS):
    
    cdef int nPts = eNIds.shape[0]
    cdef int a, b

    for a in range(nPts):
        for b in range(4): # Dof
            LHS[eNIds[a]*4+b] += lLHS[a]
            RHS[eNIds[a]*4+b] += lR[a,b]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedExplicitVMSAssemble(
    double[:,::1] nodes, long[:,::1] elements,
    double[:,::1] du, double[::1] p, double[:,::1] hdu, double[::1] hp,
    double[:,:,::1] sdu, double[:,:,::1] nsdu, double[::1] hs, double[:,::1] f,
    double[:,::1] lN, double[:,::1] lDN, double[::1] w, double[::1] coefs,
    double[::1] LHS, double[::1] RHS):

    cdef long nElms = elements.shape[0]

    cdef long nPts = 4 # elements.shape[1]
    cdef long ndim = 3 # nodes.shape[1]

    # Explicit VMS solver parameters
    cdef double c1 = coefs[0]
    cdef double c2 = coefs[1]
    cdef double nu = coefs[2]
    cdef double dt = coefs[3]
    cdef double h = 0.0
    cdef double invEpsilon = coefs[4]

    cdef double Ve
    cdef double wGp
    cdef double norm_a
    cdef double tau_u, Ru
    cdef double trGradU, trGradHu
    cdef double tmpM, varT1, varT2
    cdef double ph, hph

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] DN = np.empty((ndim, nPts), dtype=np.float)
    cdef double[:,::1] av = np.zeros((nPts, ndim), dtype=np.float)
    cdef double[:,::1] tau_av = np.zeros((nPts, ndim), dtype=np.float)
    cdef double[::1] tau_t = np.zeros(nPts, dtype=np.float)
    cdef double[::1] ah = np.zeros(ndim, dtype=np.float)
    cdef double[::1] duh = np.zeros(ndim, dtype=np.float)
    cdef double[::1] subgridF = np.zeros(ndim, dtype=np.float)
    cdef double[:,::1] gradU = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] gradP = np.empty(ndim, dtype=np.float)
    cdef double[::1] hduh = np.empty(ndim, dtype=np.float)
    cdef double[::1] sduh = np.empty(ndim, dtype=np.float)
    cdef double[::1] fh = np.empty(ndim, dtype=np.float)
    cdef double[:,::1] gradHu = np.empty((ndim, ndim), dtype=np.float)
    cdef double[::1] gradHp = np.empty(ndim, dtype=np.float)
    cdef double[::1] ahGradHu = np.empty(ndim, dtype=np.float)
    cdef double[::1] T1 = np.empty(ndim, dtype=np.float)
    cdef double[::1] lLHS = np.empty(nPts, dtype=np.float)
    cdef double[:,::1] lRHS = np.empty((nPts, 4), dtype=np.float)

    cdef double jac[3][3]
    cdef double invJac[3][3]
    cdef double cof[3][3]

    cdef long iElm
    cdef long nGp = 4 # w.shape[0]
    cdef long iGp
    cdef int i, j, k, a, b

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = elements[iElm,i]

        for i in range(nPts):
            lLHS[i] = 0.0
            for j in range(4):
                lRHS[i,j] = 0.0

        Ve = getGlbDerivatives(nodes, eNIds, lDN, DN, jac, cof, invJac)

        # gradU
        gradU[0,0] = du[eNIds[0],0]*DN[0,0] + du[eNIds[1],0]*DN[0,1] \
                        + du[eNIds[2],0]*DN[0,2] + du[eNIds[3],0]*DN[0,3]
        gradU[0,1] = du[eNIds[0],0]*DN[1,0] + du[eNIds[1],0]*DN[1,1] \
                        + du[eNIds[2],0]*DN[1,2] + du[eNIds[3],0]*DN[1,3]
        gradU[0,2] = du[eNIds[0],0]*DN[2,0] + du[eNIds[1],0]*DN[2,1] \
                        + du[eNIds[2],0]*DN[2,2] + du[eNIds[3],0]*DN[2,3]
        gradU[1,0] = du[eNIds[0],1]*DN[0,0] + du[eNIds[1],1]*DN[0,1] \
                        + du[eNIds[2],1]*DN[0,2] + du[eNIds[3],1]*DN[0,3]
        gradU[1,1] = du[eNIds[0],1]*DN[1,0] + du[eNIds[1],1]*DN[1,1] \
                        + du[eNIds[2],1]*DN[1,2] + du[eNIds[3],1]*DN[1,3]
        gradU[1,2] = du[eNIds[0],1]*DN[2,0] + du[eNIds[1],1]*DN[2,1] \
                        + du[eNIds[2],1]*DN[2,2] + du[eNIds[3],1]*DN[2,3]
        gradU[2,0] = du[eNIds[0],2]*DN[0,0] + du[eNIds[1],2]*DN[0,1] \
                        + du[eNIds[2],2]*DN[0,2] + du[eNIds[3],2]*DN[0,3]
        gradU[2,1] = du[eNIds[0],2]*DN[1,0] + du[eNIds[1],2]*DN[1,1] \
                        + du[eNIds[2],2]*DN[1,2] + du[eNIds[3],2]*DN[1,3]
        gradU[2,2] = du[eNIds[0],2]*DN[2,0] + du[eNIds[1],2]*DN[2,1] \
                        + du[eNIds[2],2]*DN[2,2] + du[eNIds[3],2]*DN[2,3]

        trGradU = gradU[0,0] + gradU[1,1] + gradU[2,2]

        # gradP
        gradP[0] = p[eNIds[0]]*DN[0,0] + p[eNIds[1]]*DN[0,1] \
                    + p[eNIds[2]]*DN[0,2] + p[eNIds[3]]*DN[0,3]
        gradP[1] = p[eNIds[0]]*DN[1,0] + p[eNIds[1]]*DN[1,1] \
                    + p[eNIds[2]]*DN[1,2] + p[eNIds[3]]*DN[1,3]
        gradP[2] = p[eNIds[0]]*DN[2,0] + p[eNIds[1]]*DN[2,1] \
                    + p[eNIds[2]]*DN[2,2] + p[eNIds[3]]*DN[2,3]

        # gradHu
        gradHu[0,0] = hdu[eNIds[0],0]*DN[0,0] + hdu[eNIds[1],0]*DN[0,1] \
                        + hdu[eNIds[2],0]*DN[0,2] + hdu[eNIds[3],0]*DN[0,3]
        gradHu[0,1] = hdu[eNIds[0],0]*DN[1,0] + hdu[eNIds[1],0]*DN[1,1] \
                        + hdu[eNIds[2],0]*DN[1,2] + hdu[eNIds[3],0]*DN[1,3]
        gradHu[0,2] = hdu[eNIds[0],0]*DN[2,0] + hdu[eNIds[1],0]*DN[2,1] \
                        + hdu[eNIds[2],0]*DN[2,2] + hdu[eNIds[3],0]*DN[2,3]
        gradHu[1,0] = hdu[eNIds[0],1]*DN[0,0] + hdu[eNIds[1],1]*DN[0,1] \
                        + hdu[eNIds[2],1]*DN[0,2] + hdu[eNIds[3],1]*DN[0,3]
        gradHu[1,1] = hdu[eNIds[0],1]*DN[1,0] + hdu[eNIds[1],1]*DN[1,1] \
                        + hdu[eNIds[2],1]*DN[1,2] + hdu[eNIds[3],1]*DN[1,3]
        gradHu[1,2] = hdu[eNIds[0],1]*DN[2,0] + hdu[eNIds[1],1]*DN[2,1] \
                        + hdu[eNIds[2],1]*DN[2,2] + hdu[eNIds[3],1]*DN[2,3]
        gradHu[2,0] = hdu[eNIds[0],2]*DN[0,0] + hdu[eNIds[1],2]*DN[0,1] \
                        + hdu[eNIds[2],2]*DN[0,2] + hdu[eNIds[3],2]*DN[0,3]
        gradHu[2,1] = hdu[eNIds[0],2]*DN[1,0] + hdu[eNIds[1],2]*DN[1,1] \
                        + hdu[eNIds[2],2]*DN[1,2] + hdu[eNIds[3],2]*DN[1,3]
        gradHu[2,2] = hdu[eNIds[0],2]*DN[2,0] + hdu[eNIds[1],2]*DN[2,1] \
                        + hdu[eNIds[2],2]*DN[2,2] + hdu[eNIds[3],2]*DN[2,3]

        trGradHu = gradHu[0,0] + gradHu[1,1] + gradHu[2,2]

        # gradHp
        gradHp[0] = hp[eNIds[0]]*DN[0,0] + hp[eNIds[1]]*DN[0,1] \
                    + hp[eNIds[2]]*DN[0,2] + hp[eNIds[3]]*DN[0,3]
        gradHp[1] = hp[eNIds[0]]*DN[1,0] + hp[eNIds[1]]*DN[1,1] \
                    + hp[eNIds[2]]*DN[1,2] + hp[eNIds[3]]*DN[1,3]
        gradHp[2] = hp[eNIds[0]]*DN[2,0] + hp[eNIds[1]]*DN[2,1] \
                    + hp[eNIds[2]]*DN[2,2] + hp[eNIds[3]]*DN[2,3]


        # Add the nonlinear sub-grid scale component for the advective
        # velocity a
        h = hs[iElm]
        for i in range(nPts):
            for j in range(ndim):
                av[i,j] = hdu[eNIds[i],j] + sdu[iElm,i,j]
                tau_av[i,j] = du[eNIds[i],j] + sdu[iElm,i,j]

            # Calculate the stabilization parameters tau
            norm_a = sqrt(tau_av[i,0]**2.0 + tau_av[i,1]**2.0 + tau_av[i,2]**2.0)
            tau_u = 1.0 / (c1*nu/(h*h) + c2*norm_a/h)
            tau_t[i] = 1.0 / (1.0/dt + 1.0/tau_u)

        # Evaluate velocity sub-grid scales
        for i in range(nPts):
            for j in range(ndim):
                Ru = tau_t[i]*(av[i,0]*gradU[j,0] + av[i,1]*gradU[j,1] + av[i,2]*gradU[j,2] + gradP[j])
                nsdu[iElm,i,j] = sdu[iElm,i,j]*tau_t[i]/dt - Ru

        # Loop through Gaussian integration points and assemble
        for iGp in range(nGp):
            wGp = w[iGp] * Ve

            # ah at Gaussion point iGp
            ah[0] = av[0,0]*lN[iGp,0] + av[1,0]*lN[iGp,1] + av[2,0]*lN[iGp,2] + av[3,0]*lN[iGp,3]
            ah[1] = av[0,1]*lN[iGp,0] + av[1,1]*lN[iGp,1] + av[2,1]*lN[iGp,2] + av[3,1]*lN[iGp,3]
            ah[2] = av[0,2]*lN[iGp,0] + av[1,2]*lN[iGp,1] + av[2,2]*lN[iGp,2] + av[3,2]*lN[iGp,3]

            # Evaluate velocity sub-grid scales
            subgridF[0] = ah[0]*gradU[0,0] + ah[1]*gradU[0,1] + ah[2]*gradU[0,2] + gradP[0]
            subgridF[1] = ah[0]*gradU[1,0] + ah[1]*gradU[1,1] + ah[2]*gradU[1,2] + gradP[1]
            subgridF[2] = ah[0]*gradU[2,0] + ah[1]*gradU[2,1] + ah[2]*gradU[2,2] + gradP[2]

            for i in range(nPts):
                nsdu[iElm,i,0] += tau_t[i]*w[iGp]*subgridF[0]*lN[iGp,i]
                nsdu[iElm,i,1] += tau_t[i]*w[iGp]*subgridF[1]*lN[iGp,i]
                nsdu[iElm,i,2] += tau_t[i]*w[iGp]*subgridF[2]*lN[iGp,i]

            # lLHS
            for a in range(nPts):
                for b in range(nPts):
                    lLHS[a] += lN[iGp,a]*lN[iGp,b]*wGp

            # duh at Gaussian point iGp
            duh[0] = du[eNIds[0],0]*lN[iGp,0] + du[eNIds[1],0]*lN[iGp,1] \
                    + du[eNIds[2],0]*lN[iGp,2] + du[eNIds[3],0]*lN[iGp,3]                    
            duh[1] = du[eNIds[0],1]*lN[iGp,0] + du[eNIds[1],1]*lN[iGp,1] \
                    + du[eNIds[2],1]*lN[iGp,2] + du[eNIds[3],1]*lN[iGp,3]
            duh[2] = du[eNIds[0],2]*lN[iGp,0] + du[eNIds[1],2]*lN[iGp,1] \
                    + du[eNIds[2],2]*lN[iGp,2] + du[eNIds[3],2]*lN[iGp,3]
            # fh at Gaussian point iGp
            fh[0] = f[eNIds[0],0]*lN[iGp,0] + f[eNIds[1],0]*lN[iGp,1] \
                    + f[eNIds[2],0]*lN[iGp,2] + f[eNIds[3],0]*lN[iGp,3]                    
            fh[1] = f[eNIds[0],1]*lN[iGp,0] + f[eNIds[1],1]*lN[iGp,1] \
                    + f[eNIds[2],1]*lN[iGp,2] + f[eNIds[3],1]*lN[iGp,3]
            fh[2] = f[eNIds[0],2]*lN[iGp,0] + f[eNIds[1],2]*lN[iGp,1] \
                    + f[eNIds[2],2]*lN[iGp,2] + f[eNIds[3],2]*lN[iGp,3]
            # ph at Gaussian point iGp
            ph = p[eNIds[0]]*lN[iGp,0] + p[eNIds[1]]*lN[iGp,1] \
                + p[eNIds[2]]*lN[iGp,2] + p[eNIds[3]]*lN[iGp,3]
            
            # # hduh at Gaussian point iGp
            # hduh[0] = hdu[eNIds[0],0]*lN[iGp,0] + hdu[eNIds[1],0]*lN[iGp,1] \
            #         + hdu[eNIds[2],0]*lN[iGp,2] + hdu[eNIds[3],0]*lN[iGp,3]                    
            # hduh[1] = hdu[eNIds[0],1]*lN[iGp,0] + hdu[eNIds[1],1]*lN[iGp,1] \
            #         + hdu[eNIds[2],1]*lN[iGp,2] + hdu[eNIds[3],1]*lN[iGp,3]
            # hduh[2] = hdu[eNIds[0],2]*lN[iGp,0] + hdu[eNIds[1],2]*lN[iGp,1] \
            #         + hdu[eNIds[2],2]*lN[iGp,2] + hdu[eNIds[3],2]*lN[iGp,3]
            # hph at Gaussian point iGp
            hph = hp[eNIds[0]]*lN[iGp,0] + hp[eNIds[1]]*lN[iGp,1] \
                + hp[eNIds[2]]*lN[iGp,2] + hp[eNIds[3]]*lN[iGp,3]
            # sduh at Gaussian point iGp
            sduh[0] = sdu[iElm,0,0]*lN[iGp,0] + sdu[iElm,1,0]*lN[iGp,1] \
                    + sdu[iElm,2,0]*lN[iGp,2] + sdu[iElm,3,0]*lN[iGp,3]                    
            sduh[1] = sdu[iElm,0,1]*lN[iGp,0] + sdu[iElm,1,1]*lN[iGp,1] \
                    + sdu[iElm,2,1]*lN[iGp,2] + sdu[iElm,3,1]*lN[iGp,3]
            sduh[2] = sdu[iElm,0,2]*lN[iGp,0] + sdu[iElm,1,2]*lN[iGp,1] \
                    + sdu[iElm,2,2]*lN[iGp,2] + sdu[iElm,3,2]*lN[iGp,3]
            # ah dot GradHu
            ahGradHu[0] = ah[0]*gradHu[0,0] + ah[1]*gradHu[0,1] + ah[2]*gradHu[0,2]
            ahGradHu[1] = ah[0]*gradHu[1,0] + ah[1]*gradHu[1,1] + ah[2]*gradHu[1,2]
            ahGradHu[2] = ah[0]*gradHu[2,0] + ah[1]*gradHu[2,1] + ah[2]*gradHu[2,2]

            # lRHS
            for a in range(nPts):
                varT1 = ah[0]*DN[0,a] + ah[1]*DN[1,a] + ah[2]*DN[2,a]
                varT2 = sduh[0]*DN[0,a] + sduh[1]*DN[1,a] + sduh[2]*DN[2,a]

                T1[0] = nu*(gradHu[0,0]*DN[0,a]+gradHu[0,1]*DN[1,a]+gradHu[0,2]*DN[2,a]) \
                        - hph*DN[0,a] - sduh[0]*varT1
                T1[1] = nu*(gradHu[1,0]*DN[0,a]+gradHu[1,1]*DN[1,a]+gradHu[1,2]*DN[2,a]) \
                        - hph*DN[1,a] - sduh[1]*varT1
                T1[2] = nu*(gradHu[2,0]*DN[0,a]+gradHu[2,1]*DN[1,a]+gradHu[2,2]*DN[2,a]) \
                        - hph*DN[2,a] - sduh[2]*varT1

                lRHS[a,0] += duh[0]*lN[iGp,a]*wGp - dt*wGp*((ahGradHu[0]-fh[0])*lN[iGp,a] + T1[0])
                lRHS[a,1] += duh[1]*lN[iGp,a]*wGp - dt*wGp*((ahGradHu[1]-fh[1])*lN[iGp,a] + T1[1])
                lRHS[a,2] += duh[2]*lN[iGp,a]*wGp - dt*wGp*((ahGradHu[2]-fh[2])*lN[iGp,a] + T1[2])
                lRHS[a,3] += ph*lN[iGp,a]*wGp - dt*wGp*(trGradHu*lN[iGp,a]-varT2)*invEpsilon

        # Assembling
        assembling(eNIds, lLHS, lRHS, LHS, RHS)






