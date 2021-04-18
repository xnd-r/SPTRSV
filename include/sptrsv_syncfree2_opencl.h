//
// Created by aromanov on 4/6/21.
//

#ifndef SPARSEPROJECT_SPTRSV_SYNCFREE2_OPENCL_H
#define SPARSEPROJECT_SPTRSV_SYNCFREE2_OPENCL_H

#include "include/sptrsv_syncfree_serialref.h"

double sptrsv_syncfree2_opencl (int           *cscColPtrTR,
                               int           *cscRowIdxTR,
                               const VALUE_TYPE    *cscValTR,
                               const int            m,
                               const int            n,
                               const int            nnzTR,
                               VALUE_TYPE    *x,
                               const VALUE_TYPE    *b);

#endif //SPARSEPROJECT_SPTRSV_SYNCFREE2_OPENCL_H
