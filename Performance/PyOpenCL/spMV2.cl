// Each working_group deals with one row at a time,
// each working_item deals with one sample,
// that means if the nSmp is too small(<local_size) a lot of computing resources is wasted.
// So in theory, spMV2 should not be faster than CSR-Stream(spMV4) for small nSmp, but this is not true
// in reality, and why?

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
        double sum_a = 0.0;
        double sum_b = 0.0;
        double sum_c = 0.0;
        for (uint s = get_local_id(0); s < nSmp; s += get_local_size(0))
        {
            sum_a = 0.0;
            sum_b = 0.0;
            sum_c = 0.0;

            for (uint x = 0; x < row_e - row_s; ++x)
            {
                j = indices[row_s + x];

                sum_a += row[x*9*nSmp+s*9] * V[s*3*N+j*3] + row[x*9*nSmp+s*9+1] * V[s*3*N+j*3+1] + row[x*9*nSmp+s*9+2] * V[s*3*N+j*3+2];
                sum_b += row[x*9*nSmp+s*9+3] * V[s*3*N+j*3] + row[x*9*nSmp+s*9+4] * V[s*3*N+j*3+1] + row[x*9*nSmp+s*9+5] * V[s*3*N+j*3+2];
                sum_c += row[x*9*nSmp+s*9+6] * V[s*3*N+j*3] + row[x*9*nSmp+s*9+7] * V[s*3*N+j*3+1] + row[x*9*nSmp+s*9+8] * V[s*3*N+j*3+2];
            }

            result[s*3*N + y*3] = sum_a;
            result[s*3*N + y*3 + 1] = sum_b;
            result[s*3*N + y*3 + 2] = sum_c;
        }
    }
}
