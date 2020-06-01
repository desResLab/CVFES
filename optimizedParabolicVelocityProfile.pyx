# cython: language_level=3, boundscheck=False
# include math method from C libs.
import numpy as np

cimport numpy as np
from libc.math cimport sqrt
from libc.stdio cimport printf
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void CoordinateTransformation(double[:,::1] nodes, long[::1] eNIds,
                                   double[:,::1] edges, double[:,::1] T):

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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def OptimizedParabolicVelocityProfile(
    double[:,::1] nodes, long[:,::1] elements, long[::1] glbNodeIds,
    double[:,::1] K, double[::1] f, double[::1] Ae):

    cdef long nElms = elements.shape[0]
    cdef long iElm

    nPts = elements.shape[1]
    ndim = nodes.shape[1]

    cdef long[::1] eNIds = np.empty(nPts, dtype=long)
    cdef long[::1] locENIds = np.empty(nPts, dtype=long)
    cdef double[:,::1] T = np.empty((nPts, nPts), dtype=np.float)
    cdef double[:,::1] TT = np.zeros((9,9), dtype=np.float)
    cdef double[:,::1] lNodes = np.empty((nPts, 2), dtype=np.float)
    cdef double[:,::1] DN = np.empty((2, nPts), dtype=np.float)
    # no used var for speed up
    cdef double[:,::1] edges = np.empty((2,3), dtype=np.float)

    cdef double area, inv_det
    cdef double y23, y31, x32, x13
    cdef long i, j


    for iElm in range(nElms):

        for i in range(nPts):
            locENIds[i] = elements[iElm,i]
            eNIds[i] = glbNodeIds[elements[iElm,i]]

        # Get the transformation matrix T.
        CoordinateTransformation(nodes, eNIds, edges, T)

        # Transform the triangular to sheer.
        lNodes[0,0] = nodes[eNIds[0],0]*T[0,0] + nodes[eNIds[0],1]*T[0,1] + nodes[eNIds[0],2]*T[0,2]
        lNodes[0,1] = nodes[eNIds[0],0]*T[1,0] + nodes[eNIds[0],1]*T[1,1] + nodes[eNIds[0],2]*T[1,2]
        lNodes[1,0] = nodes[eNIds[1],0]*T[0,0] + nodes[eNIds[1],1]*T[0,1] + nodes[eNIds[1],2]*T[0,2]
        lNodes[1,1] = nodes[eNIds[1],0]*T[1,0] + nodes[eNIds[1],1]*T[1,1] + nodes[eNIds[1],2]*T[1,2]
        lNodes[2,0] = nodes[eNIds[2],0]*T[0,0] + nodes[eNIds[2],1]*T[0,1] + nodes[eNIds[2],2]*T[0,2]
        lNodes[2,1] = nodes[eNIds[2],0]*T[1,0] + nodes[eNIds[2],1]*T[1,1] + nodes[eNIds[2],2]*T[1,2]

        # Calculate area.
        area = ((lNodes[1,0]*lNodes[2,1] - lNodes[2,0]*lNodes[1,1])
                - (lNodes[0,0]*lNodes[2,1] - lNodes[2,0]*lNodes[0,1])
                + (lNodes[0,0]*lNodes[1,1] - lNodes[1,0]*lNodes[0,1]))*0.5

        # Calculate global derivative of shape functions.
        y23 = lNodes[1,1] - lNodes[2,1]
        x32 = lNodes[2,0] - lNodes[1,0]
        y31 = lNodes[2,1] - lNodes[0,1]
        x13 = lNodes[0,0] - lNodes[2,0]
        inv_det = 1.0 / (x13*y23 - x32*y31)

        DN[0,0] = y23*inv_det
        DN[1,0] = y31*inv_det
        DN[0,1] = x32*inv_det
        DN[1,1] = x13*inv_det
        DN[0,2] = - (y23+x32)*inv_det
        DN[1,2] = - (y31+x13)*inv_det

        # Calculate k, f and assemble.
        # f = 1.0 is hard coded.
        for i in range(nPts):
            f[locENIds[i]] += 1.0
            for j in range(nPts):
                K[locENIds[i],locENIds[j]] += (DN[0,i]*DN[0,j] + DN[1,i]*DN[1,j])*area
        # Remember the area too.
        Ae[iElm] = area


