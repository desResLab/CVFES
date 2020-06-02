# cython: language_level=3, boundscheck=False
# include math method from C libs.
import numpy as np

cimport numpy as np
from libc.math cimport sqrt
cimport cython

cdef long nPts # number of nodes in shape
cdef long ndim # number of dimensions
cdef double eps = 2.220446049250313e-16


# jacobian = [[x[1]-x[0], x[2]-x[0], x[3]-x[0]],
#             [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
#             [z[1]-z[0], z[2]-z[0], z[3]-z[0]]]
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double getGlbDerivatives(double[:,::1] nodes, long[::1] eNIds,
                              double[:,::1] lDN, double[:,::1] DN,
                              double[:,::1] jac, double[:,::1] cof,
                              double[:,::1] invJac):

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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void getSurfaceNormal(double[:,::1] nodes, long[::1] eNIds,
                           double[:,::1] edges, double[:,::1] T):

    cdef double edgenorm = 0.0
    cdef double area = 0.0

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

    # calculate the area of the triangle
    area = (edges[0,1]*edges[1,2]-edges[0,2]*edges[1,1])**2 +\
           (edges[0,0]*edges[1,2]-edges[0,2]*edges[1,0])**2 +\
           (edges[0,0]*edges[1,1]-edges[0,1]*edges[1,0])**2
    return sqrt(area)


# elements - elements in the whole model that contains nodes on the wall/shell.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def BdyStressExport(double[:,::1] lumenNodes, long[:,::1] lumenElements,
                    long[::1] lumenWallNodeIds, double[:,::1] wallNodes, long[:,::1] wallElements,
                    double[:,::1] du, double[::1] p, double[:,::1] lDN, double[:,::1] wallStress):

    cdef long nElms = wallElements.shape[0]
    cdef long nWallNodes = lumenWallNodeIds.shape[0]

    cdef long nPts = 4 # elements.shape[1]
    cdef long ndim = 3 # nodes.shape[1]

    # For calculate the fluid/lumen element stress tensor.
    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] DN = np.empty((ndim, nPts), dtype=np.float)
    cdef double[:,::1] gradUh = np.empty((ndim, ndim), dtype=np.float)
    cdef double[:,:,::1] wallStressTensor = np.zeros((ndim, ndim), dtype=np.float)
    # For calculate the wall surface normal.
    cdef long[::1] eWallNIds = np.empty(3, dtype=long)
    cdef double[:,::1] T = np.empty((3, 3), dtype=np.float)
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)
    cdef double[:,::1] normals = np.zeros((nWallNodes, ndim), dtype=np.float)
    cdef double[::1] parT = np.zeros(ndim, dtype=np.float)

    cdef double jac[3][3]
    cdef double invJac[3][3]
    cdef double cof[3][3]

    cdef double mu = 0.04
    cdef double Ae = 0.0

    cdef long iElm
    cdef int i, j

    for iElm in range(nElms):

        for i in range(nPts):
            eNIds[i] = lumenElements[iElm,i]

        getGlbDerivatives(lumenNodes, eNIds, lDN, DN, jac, cof, invJac)

        # gradUh
        gradUh[0,0] = du[eNIds[0],0]*DN[0,0] + du[eNIds[1],0]*DN[0,1] \
                        + du[eNIds[2],0]*DN[0,2] + du[eNIds[3],0]*DN[0,3]
        gradUh[0,1] = du[eNIds[0],0]*DN[1,0] + du[eNIds[1],0]*DN[1,1] \
                        + du[eNIds[2],0]*DN[1,2] + du[eNIds[3],0]*DN[1,3]
        gradUh[0,2] = du[eNIds[0],0]*DN[2,0] + du[eNIds[1],0]*DN[2,1] \
                        + du[eNIds[2],0]*DN[2,2] + du[eNIds[3],0]*DN[2,3]
        gradUh[1,0] = du[eNIds[0],1]*DN[0,0] + du[eNIds[1],1]*DN[0,1] \
                        + du[eNIds[2],1]*DN[0,2] + du[eNIds[3],1]*DN[0,3]
        gradUh[1,1] = du[eNIds[0],1]*DN[1,0] + du[eNIds[1],1]*DN[1,1] \
                        + du[eNIds[2],1]*DN[1,2] + du[eNIds[3],1]*DN[1,3]
        gradUh[1,2] = du[eNIds[0],1]*DN[2,0] + du[eNIds[1],1]*DN[2,1] \
                        + du[eNIds[2],1]*DN[2,2] + du[eNIds[3],1]*DN[2,3]
        gradUh[2,0] = du[eNIds[0],2]*DN[0,0] + du[eNIds[1],2]*DN[0,1] \
                        + du[eNIds[2],2]*DN[0,2] + du[eNIds[3],2]*DN[0,3]
        gradUh[2,1] = du[eNIds[0],2]*DN[1,0] + du[eNIds[1],2]*DN[1,1] \
                        + du[eNIds[2],2]*DN[1,2] + du[eNIds[3],2]*DN[1,3]
        gradUh[2,2] = du[eNIds[0],2]*DN[2,0] + du[eNIds[1],2]*DN[2,1] \
                        + du[eNIds[2],2]*DN[2,2] + du[eNIds[3],2]*DN[2,3]

        # wall stress tensor
        for i in range(ndim):
            for j in range(ndim):
                wallStressTensor[i,j] = mu*(gradUh[i,j] + gradUh[j,i])
        
        # Get the normal of the wall element.
        for i in range(3):
            eWallNIds[i] = wallElements[iElm,i]
        
        Ae = getSurfaceNormal(wallNodes, eWallNIds, edges, T) # T[2,:] contains the normal

        parT[0] = T[2,0]*wallStressTensor[0,0] + T[2,1]*wallStressTensor[0,1] + T[2,2]*wallStressTensor[0,2]
        parT[1] = T[2,0]*wallStressTensor[1,0] + T[2,1]*wallStressTensor[1,1] + T[2,2]*wallStressTensor[1,2]
        parT[2] = T[2,0]*wallStressTensor[2,0] + T[2,1]*wallStressTensor[2,1] + T[2,2]*wallStressTensor[2,2]
        
        # Calculate Ti and add on to each node.
        for i in range(3):
            for j in range(ndim):
                wallStress[eWallNIds[i],j] += (parT[j] - T[j,j]*p[lumenWallNodeIds[eWallNIds[i]]])*Ae/3.0
        
        
