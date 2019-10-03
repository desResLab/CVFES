#include <stdio.h>
#include <stdlib.h>

#include "spmv.h"

int spmv(int m, const int* indptr, const int* indices, const double* a, const double* x, double* y)
{
    int i, j;

    for (i=0; i<m; i++)
    {
        for (j=indptr[i]; j<indptr[i+1]; j++)
        {
            y[i] += a[j]*x[indices[j]];
        }
    }

    return 0;
}
