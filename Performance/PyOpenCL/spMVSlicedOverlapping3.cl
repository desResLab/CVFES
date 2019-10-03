// Each working_group deals with one row at a time,
// each working_item deals with one sample,
// that means if the nSmp is too small(<local_size) a lot of computing resources is wasted.
// So in theory, spMV2 should not be faster than CSR-Stream(spMV4) for small nSmp, but this is not true
// in reality, and why?

__kernel void matrix_dot_vector(
        long N, const long m,
        const long smpOffset, const long nSmp,
        __global long *indptr, __global long *indices,
        __global double *M, __global double *V,
        __global double *result)
{
    for (uint y = get_group_id(0); y < N; y += get_num_groups(0))
    {
        // Row start and end
        int row_s = indptr[y];
        int row_e = indptr[y+1];
        // Row pointer
        const __global double* row = M + row_s * 9;
        // Index of current element
        int j;

        // Each work-item accumulates as many products as necessary
        // into local variable "sum"
        double sum_a = 0.0;
        double sum_b = 0.0;
        double sum_c = 0.0;
        for (uint s = get_local_id(0) + smpOffset; s < nSmp; s += get_local_size(0))
        {
            sum_a = 0.0;
            sum_b = 0.0;
            sum_c = 0.0;

            for (uint x = 0; x < row_e - row_s; ++x)
            {
                j = indices[row_s + x];

                sum_a += row[s*9*m+x*9]*V[s*3*N+j*3] + \
                         row[s*9*m+x*9+1]*V[s*3*N+j*3+1] + \
                         row[s*9*m+x*9+2]*V[s*3*N+j*3+2];
                sum_b += row[s*9*m+x*9+3]*V[s*3*N+j*3] + \
                         row[s*9*m+x*9+4]*V[s*3*N+j*3+1] + \
                         row[s*9*m+x*9+5]*V[s*3*N+j*3+2];
                sum_c += row[s*9*m+x*9+6]*V[s*3*N+j*3] + \
                         row[s*9*m+x*9+7]*V[s*3*N+j*3+1] + \
                         row[s*9*m+x*9+8]*V[s*3*N+j*3+2];
            }

            result[s*3*N+y*3] = sum_a;
            result[s*3*N+y*3+1] = sum_b;
            result[s*3*N+y*3+2] = sum_c;

            /*result[y*3*nSmp+s] = row[s*9];
            result[(y*3+1)*nSmp+s] = row[s*9+3];
            result[(y*3+2)*nSmp+s] = row[s*9+6];*/

            /*if (offset != 0 && y == N-1)
            {
                result[11*nSmp+s] = sum_a;
            }*/

        }
    }
}
