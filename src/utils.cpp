#include "include/utils.h"
#include <bits/stdint-uintn.h>
#include <c++/7/bits/c++config.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include "include/sptrsv_syncfree_opencl.h"

double read_csr(const char* filename, int* n, unsigned long long* nz, uint64_t** row_index, int** col, double** val) {
	DEBUG_INFO("Matrix: %s\n", filename);
	size_t pos = 0;
	std::string file_name(filename);
	std::string ext = file_name.substr(file_name.find_last_of(".") + 1);

	double t1 = omp_get_wtime();

	if (ext == "matrix") {
		FILE* f;
		f = fopen(filename, "rb");
		// add success open
		fread(n, sizeof(int), 1, f);
		fread(nz, sizeof(unsigned long long), 1, f);
		(*col) = (int*)malloc(*nz * sizeof(int));
		(*row_index) = (uint64_t*)malloc((*n + 1) * sizeof(uint64_t));
		(*val) = (double*)malloc(*nz * sizeof(double));
		fread(*col, sizeof(int), *nz, f);
		fread(*row_index, sizeof(uint64_t), *n + 1, f);
		fread(*val, sizeof(double), *nz, f);

		fclose(f);
	}
	else if (ext == "mtx"){
		std::ifstream file(filename);
		while (file.peek() == '%') {
			file.ignore(2048, '\n');
		}

		file >> *n >> *n >> *nz;
		(*col) = (int*)malloc(*nz * sizeof(int));
		(*row_index) = (uint64_t*)malloc((*n + 1) * sizeof(uint64_t));
		(*val) = (double*)malloc(*nz * sizeof(double));
		int col_i, row_i, cnt1 = 1;
		uint64_t cnt = 0;
		*(*row_index) = 0.;

		for (int i = 0; i < *nz; ++i) {
			file >> row_i >> col_i >> *(*val + i);
			*(*col + i) = row_i - 1;
			if (col_i == cnt1) {
				cnt++;
				continue;
			}
			else {
				*(*row_index + cnt1) = cnt;
				cnt++;
				cnt1++;
			}
		}
		*(*row_index + *n) = *nz;
	}
	else if (ext == "txt") {
		std::ifstream file(filename);
		file >> *n >> *nz;
		(*col) = (int*)malloc(*nz * sizeof(int));
		(*row_index) = (uint64_t*)malloc((*n + 1) * sizeof(uint64_t));
		(*val) = (double*)malloc(*nz * sizeof(double));
		for (int i = 0; i < *nz; ++i) {
			file >> *(*val + i);
		}
		for (int i = 0; i < *nz; ++i) {
			file >> *(*col + i);
		}
		for (int i = 0; i <= *n; ++i) {
			file >> *(*row_index + i);
		}
	}
	else {
		std::cout << "\nUnknown extension of matrix file " << ext.c_str() << " .Exit\n";
		return -1;
	}
	return omp_get_wtime() - t1;
}

void read_snodes(const char* filename, int* sn, int** snodes) {
	size_t pos = 0;
	std::string file_name(filename);
	std::string ext = file_name.substr(file_name.find_last_of(".") + 1);

	if (ext == "snodes") {
		FILE* f;
		f = fopen(filename, "rb");

		fread(sn, sizeof(int), 1, f);
		(*snodes) = (int*)malloc((*sn + 1) * sizeof(int));
		fread(*snodes, sizeof(int), *sn + 1, f);

		fclose(f);
	}
	else if (ext == "txt") {
		std::ifstream file(filename);
		file >> *sn;
		(*snodes) = (int*)malloc((*sn + 1) * sizeof(int));
		for (int i = 0; i <= *sn; ++i) {
			file >> *(*snodes + i);
		}
	}
	else {
		std::cout << "\nUnknown extension of file " << ext.c_str() << " .Exit\n";
		return;
	}
}

double fill_x_b(int n, double** x, double** b, int rhs) {
	double t1 = omp_get_wtime();
	(*x) = (double*)malloc(n * sizeof(double) * rhs);
	(*b) = (double*)malloc(n * sizeof(double) * rhs);
	srand(1984);

	// #pragma omp parallel for num_threads(omp_get_max_threads())
	for (int i = 0; i < n * rhs; ++i) {
		//(*b)[i] = (*x)[i] = (double)rand() / RAND_MAX;
		// (*b)[i] = (*x)[i] = (double)(rand() % 10 + 1);
		(*b)[i] = (*x)[i] = 1.;
	}
	return omp_get_wtime() - t1;
}

double transpose(int n, unsigned long long nz, double*& val, int*& row, uint64_t*& col_index,
	double*& val_t, uint64_t*& row_t, int*& col_index_t) {
	double t1 = omp_get_wtime();
	for (unsigned long long i = 0; i < nz; ++i)
	{
		col_index_t[row[i] + 1]++;
	}

	int S = 0, tmp;
	for (int i = 1; i <= n; ++i) {
		tmp = col_index_t[i];
		col_index_t[i] = S;
		S += tmp;
	}

	for (int i = 0; i < n; i++)
	{
		uint64_t j1 = col_index[i]; int j2 = col_index[i + 1];
		int Col = i;
		for (uint64_t j = j1; j < j2; j++)
		{
			double V = val[j];
			uint64_t RIndex = row[j];
			int IIndex = col_index_t[RIndex + 1];
			val_t[IIndex] = V;
			row_t[IIndex] = Col;
			col_index_t[RIndex + 1]++;
		}
	}
	double t2 = omp_get_wtime();
	return t2 - t1;
}

void run(const char* task_type, const char* algo_type, const char* matrix_file, const char* snodes_file,
	int* n, unsigned long long* nz,
	uint64_t** row, int** col, double** val,
	uint64_t** row_pad, int** col_index_pad, double** val_pad,
	uint64_t** row_t, int** col_t, double** val_t,
	double** x, double** b, int* sn, int** snodes,
	int nthreads, int rhs, bool opt) {
	srand(42);
	if (strcmp(task_type, "backward") == 0) { // m. b. need refactoring
		DEBUG_INFO("Task: Ux = B\n");
		DEBUG_INFO("Nuber of right sides: %d\n", rhs);

		double t_read_csr = read_csr(matrix_file, n, nz, row, col, val);
		DEBUG_INFO("Matrix read. Time: %f\n", t_read_csr);
		DEBUG_INFO("N = %d; NZ = %llu; Sparsity = %lf\n", *n, *nz, (2. * *nz) / (*n * *n + *n));
		double t_fill_x_b = fill_x_b(*n, x, b, rhs);
		DEBUG_INFO("Vectors x, b filled. Time: %f\n", t_fill_x_b);

		if ((strcmp(algo_type, "custom") == 0) || (strcmp(algo_type, "blas") == 0)) {

			read_snodes(snodes_file, sn, snodes);
			if (strcmp(algo_type, "custom") == 0) {
				if(opt){
				DEBUG_INFO("Algorithm: Supernodal custom optimized\n");
				double t_supernodal_upper = supernodal_upper_new(*sn, *snodes, *x, *val, *col, *row, *n, rhs);
				DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_supernodal_upper);
				}
				else{
				DEBUG_INFO("Algorithm: Supernodal custom\n");
				double t_supernodal_upper = supernodal_upper(*sn, *snodes, *x, *val, *col, *row, *n, rhs);
				DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_supernodal_upper);
				}
			}
			else {
				if(opt){
					DEBUG_INFO("Algorithm: Supernodal BLAS optimized\n");
				}
				else{
					DEBUG_INFO("Algorithm: Supernodal BLAS\n");
				}
				int extra_mem = 0;
				int node_size = 0;
				for (int si = 0; si < *sn; ++si) {
					node_size = *(*snodes + si + 1) - *(*snodes + si);
					extra_mem += (node_size * (node_size - 1)) / 2;
				}
				*val_pad = new double[*nz + extra_mem];
				*row_pad = new uint64_t[*nz + extra_mem];
				*col_index_pad = new int[*n + 1]{ 0 };
				DEBUG_INFO("Additional memory allocated\n");
				int nz_pad = 0;
				double t_ccs2ccs_pad = ccs2ccs_pad(*val, *col, *row, *val_pad, *row_pad, *col_index_pad, *snodes, *sn, nz_pad);
				DEBUG_INFO("Added %d elements in matrix. Padded-triangular format now. Time: %f\n", extra_mem, t_ccs2ccs_pad);

				if(opt){
					DEBUG_INFO("Algorithm is not implemented. Exit\n");
					exit(1);
					// double t_supernodal_blas_upper = supernodal_blas_upper_new(*n, nz_pad, *sn, *snodes, *x, *val_pad, *row_pad, *col_index_pad, rhs);
					// DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_supernodal_blas_upper);
				}
				else{
					double t_supernodal_blas_upper = supernodal_blas_upper(*n, nz_pad, *sn, *snodes, *x, *val_pad, *row_pad, *col_index_pad, rhs);
					DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_supernodal_blas_upper);
				}
			}
		}
		else if (strcmp(algo_type, "base") == 0) {
			if (opt){
				DEBUG_INFO("Algorithm: Base optimized\n");
				double t_base_gauss_upper = base_gauss_upper_new(*n, *val, *row, *col, *x, *b, rhs);
				DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_base_gauss_upper);
			}
			else{
				DEBUG_INFO("Algorithm: Base\n");
				double t_base_gauss_upper = base_gauss_upper(*n, *val, *row, *col, *x, *b, rhs);
				DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_base_gauss_upper);
			}
		}
		else if (strcmp(algo_type, "barrier") == 0) {
            DEBUG_INFO("Number of threads: %d\n", nthreads);
			if (opt){
				DEBUG_INFO("Algorithm: Barrier synchronization optimized\n");
				// Doesn't work
				double t_barrier_upper = barrier_upper_new(*n, *x, *b, *val, *col, *row, nthreads, rhs);
				DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_barrier_upper);
			}
			else{
				DEBUG_INFO("Algorithm: Barrier synchronization\n");
				double t_barrier_upper = barrier_upper(*n, *x, *b, *val, *col, *row, nthreads, rhs);
				DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_barrier_upper);
			}

		}
		else if (strcmp(algo_type, "write_first") == 0) {
			DEBUG_INFO("Algorithm: Write First\n");
			// for(int i = 0; i < *n; ++i){
			// 	std::cout << *(*x + i) << "\t" << *(*b + i) << "\n";
			// }
			std::cout << std::endl;
			int* row_t_int = new int[*n + 1];
			for (int i = 0; i <= *n; ++i) {
				row_t_int[i] = *(*row+i);
			}
			// rhs === 1 always
			double t_sync_free = sptrsv_syncfree3_opencl(
				*col, row_t_int, *val, *n, *n, *nz, *x, *b);
			DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_sync_free);
			delete[] row_t_int;
		}
		else if (strcmp(algo_type, "syncfree") == 0) {
			DEBUG_INFO("Algorithm: Syncfree\n");
			*val_t = new double[*nz]{ 0. };
			*col_t = new int[*n + 1]{ 0 };
			*row_t = new uint64_t[*nz]{ 0 };
			DEBUG_INFO("Matrix transposition\n");
			double t_trasnpose = transpose(*n, *nz, *val, *col, *row, *val_t, *row_t, *col_t);
			DEBUG_INFO("Transposition finished. Time: %f\n", t_trasnpose);

			// std::cout << "\n";
			// for (int i = 0; i < *nz; ++i) {
			// 	std::cout << *(*val + i) << " ";
			// }
			// std::cout << "\n";
			// for (int i = 0; i < *nz; ++i) {
			// 	std::cout << *(*col + i) << " ";
			// }
			// std::cout << "\n";
			// for (int i = 0; i < *n + 1; ++i) {
			// 	std::cout << *(*row + i) << " ";
			// }
			// std::cout << "\n";
			// exit(1);

			int* row_t_int = new int[*nz];
			for (int i = 0; i < *nz; ++i) {
				row_t_int[i] = *(*row_t+i);
			}

			double t_sync_free = sptrsv_syncfree_opencl(
				*col_t, row_t_int, *val_t, *n, *n, *nz, *x, *b, rhs);
			DEBUG_INFO("Algorithm finished. Time: %.3f\n", t_sync_free);
			delete[] row_t_int;
		}
		else if (strcmp(algo_type, "mkl") == 0) {
			int *int_row = new int[*n + 1] {0};
			for (int i = 0; i <= *n; ++i) {
				int_row[i] = (int)*(*row+i);
			}
			struct matrix_descr descrA;
			sparse_matrix_t csrA;
			sparse_status_t status_csr = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO,
				*n,  // number of rows
				*n,  // number of cols
				int_row,
				int_row + 1,
				*col,
				*val);
			descrA.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
			descrA.mode = SPARSE_FILL_MODE_UPPER;
			descrA.diag = SPARSE_DIAG_NON_UNIT;

			DEBUG_INFO("Algorithm: mkl_sparse_d_trsm\n");
			// DEBUG_INFO("mkl_get_max_threads: %d\n", mkl_get_max_threads());
			// mkl_set_dynamic(0);
			// mkl_set_num_threads(nthreads);
			DEBUG_INFO("Number of threads: %d\n", nthreads);

			double t1 = omp_get_wtime();
			sparse_status_t status = mkl_sparse_d_trsm(
				SPARSE_OPERATION_NON_TRANSPOSE,
				1.,
				csrA,
				descrA,
				SPARSE_LAYOUT_ROW_MAJOR,
				*b,
				rhs,
				rhs,
				*x,
				rhs);
			double t2 = omp_get_wtime();
			DEBUG_INFO("Algorithm finished. Time: %.3f\n", t2 - t1);
			delete[] int_row;
		}
		else {
			std::cout << "\nUnknown algorithm " << algo_type << ". Exit\n";
			return;
		}
	}
	else {
		std::cout << "\nUnknown task type " << task_type << ". Exit\n";
		return;
	}
}

void check_result(int n, double* x1, double* x2, const int rhs, bool col_major) {
	double sum = 0., norm = 0.;
	if (col_major){
		std::size_t x2_index = 0;
		for (std::size_t i = 0; i < n; ++i){
			for (std::size_t rh = 0; rh < rhs; ++rh){
				// std::cout << x2[rh * n + i] << "\t" << x1[x2_index] << "\n";
				sum += pow(x2[rh * n + i] - x1[x2_index], 2);
				norm += x2[rh * n + i] * x2[x2_index];
				++x2_index;
			}
		}
	}
	else{
		for (int i = 0; i < n * rhs; ++i) {
			sum += pow(x1[i] - x2[i], 2);
			norm += x1[i] * x1[i];
	//		if (abs(x1[i] - x2[i]) > 1e-4) {
			// DEBUG_INFO("Error: %d %f %f\n", i, x1[i], x2[i]);
	//		}
		}
	}
	DEBUG_INFO("Relative error: %.3e\n", sqrt(sum) / sqrt(norm));
	//std::cout.setf(std::ios::fixed);
	//std::cout.precision(32);
}

void compare(const char* task_type, int n, int* row, int* col, double* val, double* x_custom, double* b, double* x_check, int rhs, bool col_major) {
	if (strcmp(task_type, "backward") == 0) {
		struct matrix_descr descrA;
		sparse_matrix_t csrA;

		sparse_status_t status_csr = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO,
			n,  // number of rows
			n,  // number of cols
			row,
			row + 1,
			col,
			val);

		descrA.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
		descrA.mode = SPARSE_FILL_MODE_UPPER;
		descrA.diag = SPARSE_DIAG_NON_UNIT;

		DEBUG_INFO("Algorithm: mkl_sparse_d_trsm\n");
		double t1 = omp_get_wtime();
		sparse_status_t status = mkl_sparse_d_trsm(
			SPARSE_OPERATION_NON_TRANSPOSE,
			1.,
			csrA,
			descrA,
			SPARSE_LAYOUT_ROW_MAJOR,
			b,
			rhs,
			rhs,
			x_check,
			rhs);
		DEBUG_INFO("Algorithm finished. Time: %.3f\n", omp_get_wtime() - t1);
		std::cout << "Sparse_status_t: " << status << "\n";
		check_result(n, x_check, x_custom, rhs, col_major);
	}
	else {
		std::cout << "\nUnknown task type " << task_type << ". Exit";
	}
}
