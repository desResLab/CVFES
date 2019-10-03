
__kernel void matrix_dot_vector(
        const long N, const long nSmp,
        __global const long *indptr, __global const long *indices,
        __global const double *matrix, __global const double *vector,
        __global double *result)
{
    int gid = get_global_id(0);
    long i, j, s;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    for (s = 0; s < nSmp; ++s) {
        for (i = indptr[gid]; i < indptr[gid+1]; ++i) {
            j = indices[i];
            a += matrix[i*9*nSmp+s*9] * vector[s*3*N + (j*3)] + matrix[i*9*nSmp+s*9+1] * vector[s*3*N + (j*3+1)] + matrix[i*9*nSmp+s*9+2] * vector[s*3*N + (j*3+2)];
            b += matrix[i*9*nSmp+s*9+3] * vector[s*3*N + (j*3)] + matrix[i*9*nSmp+s*9+4] * vector[s*3*N + (j*3+1)] + matrix[i*9*nSmp+s*9+5] * vector[s*3*N + (j*3+2)];
            c += matrix[i*9*nSmp+s*9+6] * vector[s*3*N + (j*3)] + matrix[i*9*nSmp+s*9+7] * vector[s*3*N + (j*3+1)] + matrix[i*9*nSmp+s*9+8] * vector[s*3*N + (j*3+2)];
        }
        result[s*3*N + (gid*3)] = a;
        result[s*3*N + (gid*3+1)] = b;
        result[s*3*N + (gid*3+2)] = c;
    }
}
