// CSR-Stream

__kernel void matrix_dot_vector(
        const long lM, const long nSmp, const long nV,
        const long nBlocks, __global const long *rowBlocks,
        __global const long *indptr, __global const long *indices,
        __global const double *M, __global const double *V,
        __global double *result, __local double* LDS)
{
    for (uint y = get_group_id(0); y < nSmp*nBlocks; y += get_num_groups(0))
    {
        // Working on iBlk of iSmp
        int iSmp = y / nBlocks;
        int iBlk = y % nBlocks;
        // Row pointer
        const __global double* locM = M + iSmp * lM;
        const __global double* locV = V + iSmp * nV;
        // Row start and end
        int startRow = rowBlocks[iBlk];
        int nextStartRow = rowBlocks[iBlk+1];
        int nnz = indptr[nextStartRow] - indptr[startRow];
        int first_col = indptr[startRow];

        // Stream nnz values into LDS
        for (uint i = get_local_id(0); i < nnz; i += get_local_size(0))
        {
            LDS[i] = locM[first_col+i] * locV[indices[first_col+i]];
        }

        // Scalar-style reduction from LDS
        double temp = 0;
        for (uint localRow = startRow + get_local_id(0); localRow < nextStartRow; localRow += get_local_size(0))
        {
            temp = 0;
            for (uint j = indptr[localRow] - first_col; j < indptr[localRow+1] - first_col; ++j)
            {
                temp += LDS[j];
            }
            result[iSmp*nV+localRow] = temp;
            // result[iSmp*nV+localRow] = 1.0;
        }
    }
}
