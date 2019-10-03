// organize memory accesses as optimally as possible.
// Not good as spMV2
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

                sum_a += row[x*9*nSmp+s*9] * V[j*3*nSmp+s] + row[x*9*nSmp+s*9+1] * V[(j*3+1)*nSmp+s] + row[x*9*nSmp+s*9+2] * V[(j*3+2)*nSmp+s];
                sum_b += row[x*9*nSmp+s*9+3] * V[j*3*nSmp+s] + row[x*9*nSmp+s*9+4] * V[(j*3+1)*nSmp+s] + row[x*9*nSmp+s*9+5] * V[(j*3+2)*nSmp+s];
                sum_c += row[x*9*nSmp+s*9+6] * V[j*3*nSmp+s] + row[x*9*nSmp+s*9+7] * V[(j*3+1)*nSmp+s] + row[x*9*nSmp+s*9+8] * V[(j*3+2)*nSmp+s];
            }

            result[y*3*nSmp + s] = sum_a;
            result[(y*3+1)*nSmp + s] = sum_b;
            result[(y*3+2)*nSmp + s] = sum_c;
        }
    }
}
