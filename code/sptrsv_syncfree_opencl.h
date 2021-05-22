//
// Created by aromanov on 3/7/21.
//

#ifndef SPARSEPROJECT_SPTRSV_SYNCFREE_OPENCL_H
#define SPARSEPROJECT_SPTRSV_SYNCFREE_OPENCL_H

#define VALUE_TYPE double
// #include "include/sptrsv_syncfree_serialref.h"

double sptrsv_syncfree_opencl (int           *cscColPtrTR,
                               int           *cscRowIdxTR,
                               VALUE_TYPE    *cscValTR,
                               int            m,
                               int            n,
                               int            nnzTR,
                               VALUE_TYPE    *x,
                               VALUE_TYPE    *b,
                               int rhs);

double sptrsv_syncfree2_opencl (int           *cscColPtrTR,
                               int           *cscRowIdxTR,
                               const VALUE_TYPE    *cscValTR,
                               const int            m,
                               const int            n,
                               const int            nnzTR,
                               VALUE_TYPE    *x,
                               const VALUE_TYPE    *b);

double sptrsv_syncfree3_opencl (int           *csrColIdx,
                               int           *csrRowPtr,
                               const VALUE_TYPE    *csrVal,
                               const int            m,
                               const int            n,
                               const int            nnz,
                               VALUE_TYPE    *x,
                               const VALUE_TYPE    *b);

#endif //SPARSEPROJECT_SPTRSV_SYNCFREE_OPENCL_H
