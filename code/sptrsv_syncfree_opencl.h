//
// Created by aromanov on 3/7/21.
//

#ifndef SPARSEPROJECT_SPTRSV_SYNCFREE_OPENCL_H
#define SPARSEPROJECT_SPTRSV_SYNCFREE_OPENCL_H

#define VALUE_TYPE double

double sptrsv_syncfree_opencl (int           *cscColPtrTR,
                               int           *cscRowIdxTR,
                               const VALUE_TYPE    *cscValTR,
                               const int            m,
                               const int            n,
                               const int            nnzTR,
                               VALUE_TYPE    *x,
                               const VALUE_TYPE    *b);

#endif //SPARSEPROJECT_SPTRSV_SYNCFREE_OPENCL_H
