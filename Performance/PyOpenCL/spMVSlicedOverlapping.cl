// Each working_group deals with one row at a time,
// each working_item deals with one sample,
// that means if the nSmp is too small(<local_size) a lot of computing resources is wasted.
// So in theory, spMV2 should not be faster than CSR-Stream(spMV4) for small nSmp, but this is not true
// in reality, and why?

__kernel void matrix_dot_vector(
        long N, long offset,
        __global long *indptr, __global long *indices,
        __global double *M, __global double *V,
        __global double *result, __local double* partialDotProduct)
{
    for (uint y = get_group_id(0)+offset; y < N; y += get_num_groups(0))
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

        for (uint x = get_local_id(0); x < row_e - row_s; x += get_local_size(0))
        {
            j = indices[row_s + x];

            sum_a += row[x*9]*V[j*3] + row[x*9+1]*V[j*3+1] + row[x*9+2]*V[j*3+2];
            sum_b += row[x*9+3]*V[j*3]+row[x*9+4]*V[j*3+1] + row[x*9+5]*V[j*3+2];
            sum_c += row[x*9+6]*V[j*3]+row[x*9+7]*V[j*3+1] + row[x*9+8]*V[j*3+2];
        }

        // Each partial dot product is stored in shared memory
        // (get_local_size(0), 3)
        partialDotProduct[get_local_id(0)*3] = sum_a;
        partialDotProduct[get_local_id(0)*3 + 1] = sum_b;
        partialDotProduct[get_local_id(0)*3 + 2] = sum_c;

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
                partialDotProduct[get_local_id(0)*3] += partialDotProduct[(get_local_id(0)+stride)*3];
                partialDotProduct[get_local_id(0)*3 + 1] += partialDotProduct[(get_local_id(0)+stride)*3 + 1];
                partialDotProduct[get_local_id(0)*3 + 2] += partialDotProduct[(get_local_id(0)+stride)*3 + 2];
            }
        }

        // Write the result of the reduction to global memory
        if (get_local_id(0) == 0)
        {
            result[y*3] = partialDotProduct[0];
            result[y*3 + 1] = partialDotProduct[1];
            result[y*3 + 2] = partialDotProduct[2];
        }

        // Synchronize to make sure the first work-item is done with
        // reading partialDotProduct
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
