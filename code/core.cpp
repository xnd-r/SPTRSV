#include "core.h"

double base_gauss_lower(int n, double* val, uint64_t* row_index, int* col, double* x, double* b) {
	double sum;
	double t1 = omp_get_wtime();
	x[0] = b[0] / val[col[0]];
	for (int ir = 1; ir < n; ++ir) {
		sum = 0.;
		for (int j = row_index[ir]; j < row_index[ir + 1] - 1; ++j) {
			sum += val[j] * x[col[j]];
		}
		x[ir] = (b[ir] - sum) / val[row_index[ir + 1] - 1];
	}
	return omp_get_wtime() - t1;
}

double base_gauss_upper(int n, double* val, uint64_t* row_index, int* col, double* x, double* b) {
	double sum;
	double t1 = omp_get_wtime();
	x[n - 1] = b[n - 1] / val[row_index[n] - 1];
	for (int ir = n - 2; ir >= 0; ir--) {
		sum = 0.;
		for (int j = row_index[ir] + 1; j < row_index[ir + 1]; ++j) {
			sum += val[j] * x[col[j]];
		}
		x[ir] = (b[ir] - sum) / val[row_index[ir]];
	}
	return omp_get_wtime() - t1;
}

void get_tr_part_upper(int isn, int dim, double* x, double* val, int* row, uint64_t* col) {
	x[isn + dim - 1] = x[isn + dim - 1] / val[col[isn + dim - 1]];
	double sum;
	int cnt = 0;
	for (int j = isn + dim - 2; j >= isn; --j) {
		sum = 0.;
		for (int k = col[j] + 1; k <= col[j] + 1 + cnt; ++k) {
			sum += val[k] * x[row[k]];
		}
		x[j] = (x[j] - sum) / val[col[j]];
		cnt++;
	}
}

void get_rect_part_upper(int isn, int dim, double* x, double* val, int* row, uint64_t* col) {
	double sum;
	int cnt = 0;
	double t1 = omp_get_wtime();
	for (int i = isn; i < isn + dim; ++i) {
		sum = 0.;
		for (int j = col[i] + dim - cnt; j < col[i + 1]; ++j) {
			sum += val[j] * x[row[j]];
		}
		x[i] -= sum;
		cnt++;
	}
}

double supernodal_upper(size_t sn, int* supernodes, double* x, double* val, int* row, uint64_t* col_index) {
	double t1 = omp_get_wtime();
	for (int i = (int)sn - 1; i >= 0; --i) {
		get_rect_part_upper(supernodes[i], supernodes[i + 1] - supernodes[i], x, val, row, col_index);
		get_tr_part_upper(supernodes[i], supernodes[i + 1] - supernodes[i], x, val, row, col_index);

	}
	return omp_get_wtime() - t1;
}

double ccs2ccs_pad(double* val, int* row, uint64_t* col_index, double* val_pad, uint64_t* col_index_pad, int* row_pad, int* nodes, int& sn, int& nnz) {
	int pad = 0;
	int global_pad = 0;
	int col_ind_cnt = 0;
	double t1 = omp_get_wtime();
	for (int si = 0; si < (int)sn; ++si) {
		pad = nodes[si + 1] - nodes[si];

		for (int d = 0; d < pad; ++d) {
			for (int ci = 0; ci < d; ++ci, ++global_pad) {
				val_pad[global_pad] = 0.0;
				col_index_pad[global_pad] = row[col_index[nodes[si]]] + ci;
				col_ind_cnt++;
			}

			for (int ci = col_index[nodes[si] + d]; ci < col_index[nodes[si] + d + 1]; ++ci, ++global_pad) {
				val_pad[global_pad] = val[ci];
				col_index_pad[global_pad] = row[ci];
				col_ind_cnt++;
			}
			row_pad[nodes[si] + d + 1] = col_ind_cnt;
		}
	}
	nnz = col_ind_cnt;
	return omp_get_wtime() - t1;
}


double supernodal_blas_upper(int n, int nz, size_t sn, int* supernodes, double* x, double* val, uint64_t* row, int* col_index) {
	int tr_dim;
	int shift = 0;
	int rec_dim;
	int lda, ldb = 1;
	double* B = new double[n] {0.};
	double* c = new double[n] {0.};
	double t1 = omp_get_wtime();
	for (int i = (int)sn - 1; i >= 0; --i) {
		tr_dim = supernodes[i + 1] - supernodes[i];
		shift = col_index[supernodes[i]];
		lda = (col_index[supernodes[i + 1]] - col_index[supernodes[i]]) / tr_dim; // TODO: refactor
		rec_dim = col_index[supernodes[i] + 1] - col_index[supernodes[i]] - tr_dim;

		for (int i = 0; i < rec_dim; ++i) {
			B[i] = x[row[shift + tr_dim + i]];
		}

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			tr_dim, 1, rec_dim, 1., val + shift + tr_dim, lda, B, 1, 1., c, 1);

		for (int k = 0; k < tr_dim; ++k) {
			x[supernodes[i] + k] -= c[k];
			c[k] = 0.;
		}
		cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
			tr_dim, 1, 1., val + shift, lda, x + supernodes[i], ldb);
	}
	double t2 = omp_get_wtime() - t1;
	delete[] B;
	delete[] c;
	return t2;
}

int setLevelUpAndGetMaxLevel(int n, int* col, uint64_t* row_index, uint64_t* levelsUp)
{
	for (int i = 0; i < n; i++)
	{
		levelsUp[i] = 1;
	}
	uint64_t max_level_in_current_line = 1;
	int max = 1;
	for (int i = n - 1; i >= 0; i--)
	{
		max_level_in_current_line = 1;
		if ((row_index[i + 1] - row_index[i]) > 1)
		{
			for (int j = row_index[i] + 1; j < row_index[i + 1]; j++)
			{
				if (levelsUp[col[j]] > max_level_in_current_line)
					max_level_in_current_line = levelsUp[col[j]];
			}
			levelsUp[i] += max_level_in_current_line;
		}
		if (levelsUp[i] > max)
		{
			max = levelsUp[i];
		}
	}
	return max;
}

// Authors: Dmitriy Akhmedzhanov, Alexander Romanov
double gaussBarrierUp(int n, double* x, double* b, double* val, int* col, uint64_t* row, int num_of_threads)
{
	uint64_t index_top, index_low, vertex;
	double sum;
	double t1, t2;
	auto* levelsUp = (uint64_t*)malloc(n * sizeof(uint64_t));
	t1 = omp_get_wtime();
	uint64_t maxLevel = setLevelUpAndGetMaxLevel(n, col, row, levelsUp);

	auto* rowsByLevel = new std::vector<int>[maxLevel + 1];
	auto* tasksByLevel = new std::vector<int>[maxLevel + 1];
	for (int i = 0; i < n; i++)
	{
		rowsByLevel[levelsUp[i]].push_back(i);
	}

	for (int i = 0; i < maxLevel + 1; i++)
	{
		uint64_t levelNz = 0;
		for (int j : rowsByLevel[i])
		{
			vertex = j;
			index_top = row[vertex + 1];
			index_low = row[vertex];
			levelNz += index_top - index_low;
		}
		uint64_t limit = levelNz / num_of_threads;
		int current_task = 1;
		uint64_t current_nz = 0;
		tasksByLevel[i].resize(num_of_threads + 1, 0);

		for (int j = 0; j < rowsByLevel[i].size(); ++j)
		{
			vertex = rowsByLevel[i][j];
			index_top = row[vertex + 1];
			index_low = row[vertex];
			if (current_task == num_of_threads)
			{
				tasksByLevel[i][current_task] += 1;
				continue;
			}
			if (index_top - index_low + current_nz > limit)
			{
				current_nz = index_top - index_low;
				current_task++;
				tasksByLevel[i][current_task] += 1;
			}
			else
			{
				tasksByLevel[i][current_task] += 1;
				current_nz += index_top - index_low;
			}
		}
//		TODO: understand why
		for (int j = 1; j < tasksByLevel[i].size(); ++j)
		{
			tasksByLevel[i][j] = tasksByLevel[i][j] + tasksByLevel[i][j - 1];
		}
//		DEBUG_INFO("number_of_vertex: %zu\n", rowsByLevel[i].size());
	}
	t2 = omp_get_wtime() - t1;
	DEBUG_INFO("dag building elapsed time is %lf sec\n", t2);
	DEBUG_INFO("maxLevel: %lu \n", maxLevel);
	double t3 = omp_get_wtime();
	for (int bi = 0; bi < 1; bi++)
	{
#pragma omp parallel private(vertex, sum, index_low, index_top)
		for (int i = 1; i < maxLevel + 1; i++)
		{
			// #pragma omp parallel for private(vertex, sum, index_low, index_top)
			// for (int j = 0; j < rowsByLevel[i].size(); ++j)
			// {
			// 	vertex = rowsByLevel[i][j];
			// 	sum = 0;
			// 	index_top = row[vertex + 1];
			// 	index_low = row[vertex];
			// 	for (uint64_t k = index_low + 1; k < index_top; k++) {
			// 		sum += val[k] * x[col[k] + bi*n];
			// 	}
			// 	x[vertex + bi*n] = (b[vertex + bi*n] - sum) / val[index_low];
			// }
#pragma omp for schedule(static, 1)
			for (int t = 0; t < tasksByLevel[i].size() - 1; ++t)
			{
				for (int j = tasksByLevel[i][t]; j < tasksByLevel[i][t + 1]; ++j)
				{
					vertex = rowsByLevel[i][j];
					sum = 0;
					index_top = row[vertex + 1];
					index_low = row[vertex];
					for (uint64_t k = index_low + 1; k < index_top; k++) {
						sum += val[k] * x[col[k] + bi * n];
					}
					x[vertex + bi * n] = (b[vertex + bi * n] - sum) / val[index_low];
				}
			}
		}
	}
	double t4 = omp_get_wtime() - t3;
	DEBUG_INFO("elapsed time is %lf sec\n", t4);
	delete[] rowsByLevel;
	delete[] tasksByLevel;
	return t2 + t4;
}
