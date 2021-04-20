#pragma once
#include "omp.h"
#include "mkl.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include "include/utils.h"

double base_gauss_lower(int n, double* val, uint64_t* row_index, int* col, double* x, double* b);
double base_gauss_upper(int n, double* val, uint64_t* row_index, int* col, double* x, double* b, int rhs);
double supernodal_upper(size_t sn, int* supernodes, double* x, double* val, int* row, uint64_t* col_index, int n, int rhs);
double supernodal_blas_upper(int n, int nz, size_t sn, int* supernodes, double* x, double* val, uint64_t* col_index, int* row, int rhs);
double ccs2ccs_pad(double* val, int* row, uint64_t* col_index, double* val_pad, uint64_t* col_index_pad, int* row_pad, int* nodes, int& sn, int& nnz);
int setLevelUpAndGetMaxLevel(int n, int* col, uint64_t* row_index, uint64_t* levelsUp);
double gaussBarrierUp(int n, double* x, double* b, double* val, int* col, uint64_t* row, int num_of_threads, int rhs);
