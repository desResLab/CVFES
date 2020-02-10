# cython: language_level=3, boundscheck=False
# include math method from C libs.
import numpy as np

cimport numpy as np
# from libc.stdio cimport printf
from cython.parallel import prange
cimport cython
cimport openmp


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def MultiplyByVector(int nNodes, int[::1] indptr, int[::1] indices,
                     double[::1] M, double[::1] v,
                     double[::1] res):

    # cdef long nNodes = len(indptr) - 1
    cdef int a, ia, b

    # cdef int num_threads

    for a in prange(nNodes, nogil=True): # , num_threads=1):
        # if a == 0:
        #     num_threads = openmp.omp_get_num_threads()
        #     printf('How many threads are using: %d\n', num_threads)

        for ia in range(indptr[a], indptr[a+1]):
            res[a] += M[ia]*v[indices[ia]]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def MultiplyByVectorOrigin(long[::1] indptr, long[::1] indices,
                           double[:,:,:,::1] M, double[:,::1] v,
                           double[:,::1] res):

    cdef long nNodes = len(indptr) - 1
    cdef long a, ia, b
    cdef long r, c

    cdef long nSamples = M.shape[1]
    cdef long s
    # cdef int num_threads

    for a in prange(nNodes, nogil=True): # , num_threads=1):
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
