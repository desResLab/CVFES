__kernel void matrix_dot_vector(
        const long N, const long nSmp,
        __global const long *indptr, __global const long *indices,
        __global const double *M, __global const double *V,
        __global double *result)
{
    for (uint y = get_group_id(0); y < N; y += get_num_groups(0))
    {
        // Row start and end
        int row_s = indptr[y];
        int row_e = indptr[y+1];
        // Row pointer
        const __global double* row = M + row_s * 9 * nSmp;
        // Index of current element
        int j;

        // Each work-item accumulates as many products as necessary
        // into local variable "sum"
        for (uint s = get_local_id(0); s < nSmp; s += get_local_size(0))
        {
            double sum = 0.0;
            for (uint x = 0; x < row_e - row_s; ++x)
            {
                j = indices[row_s + x];

                sum += row[x*9*nSmp+s*9] * V[s*3*N+j*3] + row[x*9*nSmp+s*9+1] * V[s*3*N+j*3+1] + row[x*9*nSmp+s*9+2] * V[s*3*N+j*3+2];
                sum += row[x*9*nSmp+s*9+3] * V[s*3*N+j*3] + row[x*9*nSmp+s*9+4] * V[s*3*N+j*3+1] + row[x*9*nSmp+s*9+5] * V[s*3*N+j*3+2];
                sum += row[x*9*nSmp+s*9+6] * V[s*3*N+j*3] + row[x*9*nSmp+s*9+7] * V[s*3*N+j*3+1] + row[x*9*nSmp+s*9+8] * V[s*3*N+j*3+2];
            }

            /*
            result[s*3*N + y*3] = sum_a;
            result[s*3*N + y*3 + 1] = sum_b;
            result[s*3*N + y*3 + 2] = sum_c;*/
        }
    }
}
