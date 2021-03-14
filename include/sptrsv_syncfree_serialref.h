//
// Created by aromanov on 3/7/21.
//

#ifndef SPARSEPROJECT_SPTRSV_SYNCFREE_SERIALREF_H
#define SPARSEPROJECT_SPTRSV_SYNCFREE_SERIALREF_H

#define VALUE_TYPE double
#define SUBSTITUTION_BACKWARD 1

int sptrsv_syncfree_analyser(uint64_t   *cscRowIdx,
                             const int    m,
                             const int    n,
                             const int    nnz,
                             int   *csrRowHisto);

int sptrsv_syncfree_executor(int           *cscColPtr,
                             uint64_t           *cscRowIdx,
                             const VALUE_TYPE    *cscVal,
                             const int           *graphInDegree,
                             const int            m,
                             const int            n,
                             const int            substitution,
                             const int            rhs,
                             const VALUE_TYPE    *b,
                             VALUE_TYPE    *x);

double sptrsv_syncfree_serialref(int           *cscColPtrTR,
                                 uint64_t           *cscRowIdxTR,
                                 const VALUE_TYPE    *cscValTR,
                                 const int            m,
                                 const int            n,
                                 const int            nnzTR,
                                 const int            substitution,
                                 const int            rhs,
                                 VALUE_TYPE    *x,
                                 const VALUE_TYPE    *b);


#endif //SPARSEPROJECT_SPTRSV_SYNCFREE_SERIALREF_H
