
__kernel void matrix_dot_vector(
        const long N, const long nSmp,
        __global const long *indptr, __global const long *indices,
        __global const double *M, __global const double *V,
        __global double *result, __local double* partialDotProduct)
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
        for (uint s = 0; s < nSmp; ++s)
        {
            sum_a = 0.0;
            sum_b = 0.0;
            sum_c = 0.0;

            for (uint x = get_local_id(0); x < row_e - row_s; x += get_local_size(0))
            {
                j = indices[row_s + x];

                sum_a += row[s*9] * V[s*3*N+j*3] + row[s*9+1] * V[s*3*N+j*3+1] + row[s*9+2] * V[s*3*N+j*3+2];
                sum_b += row[s*9+3] * V[s*3*N+j*3] + row[s*9+4] * V[s*3*N+j*3+1] + row[s*9+5] * V[s*3*N+j*3+2];
                sum_c += row[s*9+6] * V[s*3*N+j*3] + row[s*9+7] * V[s*3*N+j*3+1] + row[s*9+8] * V[s*3*N+j*3+2];
            }

            // Each partial dot product is stored in shared memory
            // (get_local_size(0), nSmp, 3)
            partialDotProduct[get_local_id(0)*nSmp*3 + s*3] = sum_a;
            partialDotProduct[get_local_id(0)*nSmp*3 + s*3 + 1] = sum_b;
            partialDotProduct[get_local_id(0)*nSmp*3 + s*3 + 2] = sum_c;
        }

        // Perform parallel reduction to add each work-item's
        // partial dot product together
        for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2)
        {
            // Synchronize to make sure each work-item is done updating
            // shared memory; this is necessary because work-item read
            // results that have been written by other work-items
            barrier(CLK_LOCAL_MEM_FENCE);

            // Only the first work-items in the work-group add elements together
            if (get_local_id(0) < stride)
            {
                // Add two elements from the "partialDotProduct" array
                // and store the result in partialDotProduct[index]
                for (uint s = 0; s < nSmp; ++s)
                {
                    partialDotProduct[get_local_id(0)*nSmp*3 + s*3] += partialDotProduct[(get_local_id(0)+stride)*nSmp*3 + s*3];
                    partialDotProduct[get_local_id(0)*nSmp*3 + s*3 + 1] += partialDotProduct[(get_local_id(0)+stride)*nSmp*3 + s*3 + 1];
                    partialDotProduct[get_local_id(0)*nSmp*3 + s*3 + 2] += partialDotProduct[(get_local_id(0)+stride)*nSmp*3 + s*3 + 2];
                }
            }
        }

        // Write the result of the reduction to global memory
        if (get_local_id(0) == 0)
        {
            for (uint s = 0; s < nSmp; ++s)
            {
                result[s*3*N + y*3] = partialDotProduct[s*3];
                result[s*3*N + y*3 + 1] = partialDotProduct[s*3 + 1];
                result[s*3*N + y*3 + 2] = partialDotProduct[s*3 + 2];
            }
        }

        // Synchronize to make sure the first work-item is done with
        // reading partialDotProduct
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
