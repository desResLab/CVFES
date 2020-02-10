// Testing MKL and C pure implementation.

#define LOOP_COUNT  1000

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "spmv.h"
#include "mmio.h"


int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J, *indptr;
    int curLine;
    double *val, *b, *y; // y = val*b

    int r;
    double s_initial, s_elapsed;
    int max_threads;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }
    else
    {
        if ((f = fopen(argv[1], "r")) == NULL)
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    /* reseve memory for matrices */
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    // Process vector I to generate indptr vector.
    indptr = (int *) malloc((M+1) * sizeof(int));
    curLine = 0;
    indptr[0] = 0;
    for (i=0; i<nz; i++)
    {
        if (I[i] != curLine)
        {
            indptr[++curLine] = i;
        }
    }
    indptr[M] = nz;

    // Start calling the mkl sparse matrix multiplication routines.
    // Set vector b.
    b = (double *) malloc(N * sizeof(double));
    for (i=0; i<N; i++)
    {
        b[i] = 1.0;
    }

    y = (double *) malloc(M * sizeof(double));
    for (i=0; i<M; i++)
    {
        y[i] = 0.0;
    }

    max_threads = mkl_get_max_threads();
    mkl_set_num_threads(max_threads);

    // Performance with MKL lib.
    s_initial = dsecnd();

    for (r=0; r<LOOP_COUNT; r++)
    {
        /*void mkl_cspblas_dcsrgemv (const char *transa , const MKL_INT *m , const double *a , const MKL_INT *ia , const MKL_INT *ja , const double *x , double *y );*/
        mkl_cspblas_dcsrgemv("N", &M, val, indptr, J, b, y);
    }

    s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;
    printf (" == Matrix multiplication using Intel(R) MKL dgemm completed ==\n"
            " == at %.5f milliseconds using %d thread(s) ==\n\n", (s_elapsed * 1000), max_threads);

    // Performance with my lib.
    for (i=0; i<M; i++)
    {
        y[i] = 0.0;
    }

    s_initial = dsecnd();

    for (r=0; r<LOOP_COUNT; r++)
    {
        /*void mkl_cspblas_dcsrgemv (const char *transa , const MKL_INT *m , const double *a , const MKL_INT *ia , const MKL_INT *ja , const double *x , double *y );*/
        spmv(M, indptr, J, val, b, y);
    }

    s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;
    printf (" == Matrix multiplication using my spmv completed ==\n"
            " == at %.5f milliseconds using %d thread(s) ==\n\n", (s_elapsed * 1000), 1);

    // for (i=0; i<M; i++)
    //     fprintf(stdout, "%20.19g\n", y[i]);

    return 0;
}
