#include <iostream>
#include "include/utils.h"
#include "include/sptrsv_syncfree_serialref.h"


int sptrsv_syncfree_analyser(uint64_t   *cscRowIdx,
                             const int    m,
                             const int    n,
                             const int    nnz,
                                   int   *csrRowHisto)
{
    memset(csrRowHisto, 0, m * sizeof(int));

    // generate row pointer by partial transposition
//#pragma omp parallel for
    for (int i = 0; i < nnz; i++)
    {
//#pragma omp atomic
        csrRowHisto[cscRowIdx[i]]++;
    }

    return 0;
}

int sptrsv_syncfree_executor(int           *cscColPtr,
                             uint64_t           *cscRowIdx,
                             const VALUE_TYPE    *cscVal,
                             const int           *graphInDegree,
                             const int            m,
                             const int            n,
                             const int            substitution,
                             const int            rhs,
                             const VALUE_TYPE    *b,
                                   VALUE_TYPE    *x)
{
    // malloc tmp memory to simulate atomic operations
    int *graphInDegree_atomic = (int *)malloc(sizeof(int) * m);
    memset(graphInDegree_atomic, 0, sizeof(int) * m);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *left_sum = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
    memset(left_sum, 0, sizeof(VALUE_TYPE) * m * rhs);

    int dia = 0;
    for (int i = n-1; i >= 0; i--)
    {
        dia = graphInDegree[i] - 1;

        // while loop, i.e., wait, until all nnzs are prepared
        do
        {
            // just wait
        }
        while (dia != graphInDegree_atomic[i]);

        for (int k = 0; k < rhs; k++)
        {
            VALUE_TYPE xi = (b[i * rhs + k] - left_sum[i * rhs + k]) / cscVal[cscColPtr[i+1]-1];
            x[i * rhs + k] = xi;
        }

        for (int j = cscColPtr[i]; j < cscColPtr[i+1]-1; j++)
        {
            int rowIdx = cscRowIdx[j];
            // atomic add
            for (int k = 0; k < rhs; k++)
                left_sum[rowIdx * rhs + k] += x[i * rhs + k] * cscVal[j];
            graphInDegree_atomic[rowIdx] += 1;
            //printf("node %i updated node %i\n", i, rowIdx);
        }
    }

    free(graphInDegree_atomic);
    free(left_sum);

    return 0;
}

double sptrsv_syncfree_serialref(int           *cscColPtrTR,
                              uint64_t           *cscRowIdxTR,
                              const VALUE_TYPE    *cscValTR,
                              const int            m,
                              const int            n,
                              const int            nnzTR,
                              const int            substitution,
                              const int            rhs,
                                    VALUE_TYPE    *x,
                              const VALUE_TYPE    *b) {
    int *graphInDegree = (int *)malloc(m * sizeof(int));
	DEBUG_INFO("SpTRSV Serial analyser started");
	double t_analyzer1 = omp_get_wtime();
    sptrsv_syncfree_analyser(cscRowIdxTR, m, n, nnzTR, graphInDegree);
	double t_analyzer2 = omp_get_wtime();
	DEBUG_INFO("SpTRSV Serial analyser on L used %f s", t_analyzer2 - t_analyzer1);

	DEBUG_INFO("SpTRSV Serial executor started");
	double t_executor1 = omp_get_wtime();
	sptrsv_syncfree_executor(cscColPtrTR, cscRowIdxTR, cscValTR, graphInDegree, m, n, substitution, rhs, b, x);
	double t_executor2 = omp_get_wtime();
	DEBUG_INFO("SpTRSV Serial executor used %f s", t_executor2 - t_executor1);

    free(graphInDegree);
    return t_analyzer2 - t_analyzer1 + t_executor2 - t_executor1;
}
