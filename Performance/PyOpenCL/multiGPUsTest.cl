__kernel void matrix_dot_vector(
        const long N, const long nSmp, long offset,
        __global long *indptr, __global long *indices,
        __global double *M, __global double *v,
        __global double *result)
{
    for (uint y = get_group_id(0)+offset; y < N; y += get_num_groups(0))
    {
        // Row start and end
        int row_s = indptr[y];
        int row_e = indptr[y+1];

        // The start of the row
        const __global double *row = M + row_s * nSmp * 9;

        // Local variable to store tmp result.
        double sum0, sum1, sum2;
        uint iy;

        for (uint iSmp = get_local_id(0); iSmp < nSmp; iSmp += get_local_size(0))
        {
            sum0 = sum1 = sum2 = 0.0;

            for (uint x = 0; x < row_e - row_s; x++)
            {
                iy = indices[row_s+x];

                sum0 += row[x*nSmp*9+iSmp*9] * v[iy*3*nSmp+iSmp] + row[x*nSmp*9+iSmp*9+1] * v[(iy*3+1)*nSmp+iSmp] + row[x*nSmp*9+iSmp*9+2] * v[(iy*3+2)*nSmp+iSmp];
                sum1 += row[x*nSmp*9+iSmp*9+3] * v[iy*3*nSmp+iSmp] + row[x*nSmp*9+iSmp*9+4] * v[(iy*3+1)*nSmp+iSmp] + row[x*nSmp*9+iSmp*9+5]* v[(iy*3+2)*nSmp+iSmp];
                sum2 += row[x*nSmp*9+iSmp*9+6] * v[iy*3*nSmp+iSmp] + row[x*nSmp*9+iSmp*9+7] * v[(iy*3+1)*nSmp+iSmp] + row[x*nSmp*9+iSmp*9+8]* v[(iy*3+2)*nSmp+iSmp];
            }

            result[y*3*nSmp+iSmp] = sum0;
            result[(y*3+1)*nSmp+iSmp] = sum1;
            result[(y*3+2)*nSmp+iSmp] = sum2;
        }
    }
}
