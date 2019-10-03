# cython: language_level=3, boundscheck=False
# include math method from C libs.
import numpy as np

cimport numpy as np
# from libc.stdio cimport printf
from cython.parallel import prange
cimport cython
cimport openmp

@cython.boundscheck(False)
@cython.wraparound(False)
def Transfer(long[::1] indptr, long[::1] indices, double[:,:,:,::1] M, double[:,::1] nM):

    cdef long nNodes = len(indptr) - 1
    cdef long nSmp = M.shape[1]
    cdef long row, col, s
    cdef long i, j

    cdef long row_nnz = 0
    cdef long rb, ib, jb

    for row in range(nNodes):
        row_nnz = indptr[row+1] - indptr[row]
        rb = indptr[row]*9
        for col in range(indptr[row], indptr[row+1]):
            jb = (col-indptr[row])*3
            for s in range(nSmp):
                for i in range(3):
                    ib = i*3*row_nnz
                    for j in range(3):
                        nM[s, rb+ib+jb+j] = M[col, s, i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
def ExtendStructureInfo(long[::1] indptr, long[::1] indices, long[::1] exIndptr, long[::1] exIndices):

    cdef long nNodes = len(indptr) - 1
    cdef long row, col
    cdef long row_nnz

    cdef long i, j
    cdef long iind = 0
    cdef long iptr = 1
    exIndptr[0] = 0

    for row in range(nNodes):
        row_nnz = (indptr[row+1] - indptr[row])*3
        for i in range(3):
            for col in range(indptr[row], indptr[row+1]):
                for j in range(3):
                    exIndices[iind] = indices[col]*3 + j
                    iind += 1
            exIndptr[iptr] = exIndptr[iptr-1] + row_nnz
            iptr += 1


@cython.boundscheck(False)
@cython.wraparound(False)
def CalcGPUAssignment(long LOCAL_SIZE, long[::1] indptr, long[::1] rowBlocks):

    cdef long totalRows = len(indptr) - 1
    cdef long tmpsum = 0
    cdef long last_i = 0
    cdef long ctr = 1

    cdef long i

    rowBlocks[0] = 0

    for i in range(totalRows):
        tmpsum += indptr[i+1] - indptr[i]
        if tmpsum == LOCAL_SIZE:
            # This row fills up LOCAL_SIZE
            rowBlocks[ctr] = i + 1
            ctr += 1
            last_i = i
            tmpsum = 0
        elif tmpsum > LOCAL_SIZE:
            if i - last_i > 1:
                # This extra row will not fit
                rowBlocks[ctr] = i
                i -= 1
            elif i - last_i == 1:
                # This one row is too large
                rowBlocks[ctr] = i + 1
            ctr += 1
            last_i = i
            tmpsum = 0

    # Row indices from 0 to totalRows - 1
    rowBlocks[ctr] = totalRows
    return ctr + 1



