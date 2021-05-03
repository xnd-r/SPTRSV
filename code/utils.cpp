#include "utils.h"
#include <iostream>


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
		(*b)[i] = (*x)[i] = (double)(rand() % 10 + 1);
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
	int nthreads, int rhs) {
	srand(42);
	if (strcmp(task_type, "forward") == 0) { // m. b. need refactoring
		DEBUG_INFO("Task: Ux = B\n");
		DEBUG_INFO("Nuber of right sides: %d\n", rhs);

		double t_read_csr = read_csr(matrix_file, n, nz, row, col, val);
		DEBUG_INFO("Matrix read. Time: %f\n", t_read_csr);
		double t_fill_x_b = fill_x_b(*n, x, b, rhs);
		DEBUG_INFO("Vectors x, b filled. Time: %f\n", t_fill_x_b);

		if ((strcmp(algo_type, "custom") == 0) || (strcmp(algo_type, "blas") == 0)) {

			read_snodes(snodes_file, sn, snodes);
			if (strcmp(algo_type, "custom") == 0) {
				DEBUG_INFO("Algorithm: Supernodal custom\n");
				double t_supernodal_upper = supernodal_upper(*sn, *snodes, *x, *val, *col, *row, *n, rhs);
				DEBUG_INFO("Algorithm finished. Time: %f\n", t_supernodal_upper);
			}
			else {
				DEBUG_INFO("Algorithm: Supernodal BLAS\n");
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

				double t_supernodal_blas_upper = supernodal_blas_upper(*n, nz_pad, *sn, *snodes, *x, *val_pad, *row_pad, *col_index_pad, rhs);
				DEBUG_INFO("Algorithm finished. Time: %f\n", t_supernodal_blas_upper);
			}
		}
		else if (strcmp(algo_type, "base") == 0) {
			DEBUG_INFO("Algorithm: Base\n");
			double t_base_gauss_upper = base_gauss_upper(*n, *val, *row, *col, *x, *b, rhs);
			DEBUG_INFO("Algorithm finished. Time: %f\n", t_base_gauss_upper);
		}
		else if (strcmp(algo_type, "barrier") == 0) {
            DEBUG_INFO("Algorithm: Barrier\n");
            DEBUG_INFO("Number of threads: %d\n", nthreads);
			double t_barrier_upper = gaussBarrierUp(*n, *x, *b, *val, *col, *row, nthreads, rhs);
			DEBUG_INFO("Algorithm finished. Time: %f\n", t_barrier_upper);
		}
		else if (strcmp(algo_type, "syncfree") == 0) {
			//(*col_t) = (int*)malloc(*nz * sizeof(int));
			//memset(col_t, 0, *nz * sizeof(int));
			//(*row_t) = (uint64_t*)malloc((*n + 1) * sizeof(uint64_t));
			//memset(row_t, 0, (*n + 1) * sizeof(uint64_t));
			//(*val_t) = (double*)malloc(*nz * sizeof(double));
			//memset(val_t, 0., *nz * sizeof(double));

			*val_t = new double[*nz]{ 0. };
			*col_t = new int[*n + 1]{ 0 };
			*row_t = new uint64_t[*nz]{ 0 };
			DEBUG_INFO("Matrix transposition\n");
			double t_trasnpose = transpose(*n, *nz, *val, *col, *row, *val_t, *row_t, *col_t);
			DEBUG_INFO("Transposition finished. Time: %f\n", t_trasnpose);

			//std::cout << "\n";
			//for (int i = 0; i < *nz; ++i) {
			//	std::cout << *(*val_t + i) << " ";
			//}
			//std::cout << "\n";
			//for (int i = 0; i < *nz; ++i) {
			//	std::cout << *(*row_t + i) << " ";
			//}
			//std::cout << "\n";
			//for (int i = 0; i < *n + 1; ++i) {
			//	std::cout << *(*col_t + i) << " ";
			//}
			//std::cout << "\n";
			//exit(1);

//			DEBUG_INFO("Algorithm: Synchronization free\n");
//			double t_sync_free = sptrsv_syncfree_serialref(
//				*col_t, *row_t, *val_t, *n, *n, *nz, SUBSTITUTION_BACKWARD, 1, *x, *b);
//			DEBUG_INFO("Algorithm finished. Time: %f\n", t_sync_free);

			int* row_t_int = new int[*nz];
			for (int i = 0; i < *nz; ++i) {
				row_t_int[i] = *(*row_t+i);
			}

			double t_sync_free = sptrsv_syncfree_opencl(
				*col_t, row_t_int, *val_t, *n, *n, *nz, *x, *b);
			DEBUG_INFO("Algorithm finished. Time: %f\n", t_sync_free);
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
			mkl_set_num_threads(nthreads);
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
			DEBUG_INFO("Algorithm finished. Time: %f\n", t2 - t1);
			// std::cout << "Sparse_status_t: " << status << "\n";
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

void check_result(int n, double* x1, double* x2) {
	double sum = 0., norm = 0.;
	for (int i = 0; i < n; ++i) {
		sum += pow(x1[i] - x2[i], 2);
		norm += x1[i] * x1[i];
//		if (abs(x1[i] - x2[i]) > 1e-4) {
//			DEBUG_INFO("Error: %d %f %f\n", i, x1[i], x2[i]);
//		}
	}
	DEBUG_INFO("Relative error: %f\n", sqrt(sum) / sqrt(norm));
	//std::cout.setf(std::ios::fixed);
	//std::cout.precision(32);
}

void compare(const char* task_type, int n, int* row, int* col, double* val, double* x_custom, double* b, double* x_check, int rhs) {
	if (strcmp(task_type, "forward") == 0) {
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
		DEBUG_INFO("Algorithm finished. Time: %f\n", omp_get_wtime() - t1);
		std::cout << "Sparse_status_t: " << status << "\n";
		check_result(n, x_check, x_custom);
	}
	else {
		std::cout << "\nUnknown task type " << task_type << ". Exit";
	}
}
