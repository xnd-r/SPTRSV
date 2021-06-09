#pragma once
#include <iostream>
#include <cstdint>
#include "mkl.h"
#include "omp.h"
#include <fstream>
#include <string.h>
#include <cmath>
#include "core.h"

#define DEBUG_INFO(f, args...) fprintf(stderr, "%s_%s\t%s:%d:%s():\t\t" f, __DATE__, __TIME__, __FILE__, __LINE__, __func__, ##args);


double read_csr(const char* filename, int* n, unsigned long long* nz, uint64_t** row, int** col, double** val);
void read_snodes(const char* filename, int* sn, int** snodes);
double fill_x_b(int n, double** x, double** b, int rhs);

void transpose(int n, unsigned long long nz, double*& val, uint64_t*& row, int*& col_index,
	double*& val_t, uint64_t*& row_t, int*& col_index_t);


void run(const char* task_type, const char* algo_type, const char* matrix_file, const char* snodes_file,
	int* n, unsigned long long* nz,
	uint64_t** row, int** col, double** val,
	uint64_t** row_pad, int** col_index_pad, double** val_pad,
	uint64_t** row_t, int** col_t, double** val_t,
	double** x, double** b, int* sn, int** snodes,
	int nthreads, int rhs, bool opt);

void check_result(int n, double* x1, double* x2, int rhs, bool col_major);
void compare(const char* task_type, int n, int* row, int* col, double* val, double* x, double* b, double* x_custom, int rhs, bool col_major);
