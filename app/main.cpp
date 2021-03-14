#include "include/utils.h"

int main(int argc, char** argv) {
	//if (argc < 4) {
	//	std::cout << "Usage:\n\ntask_type (can be \"forward\", \"full\")\n";
	//	std::cout << "algo_type (can be \"base\", \"custom\", \"blas\")\n";
	//	std::cout << "matrix (can be *.mtx, *.bin, *.txt)" << std::endl;
	//	return 1;
	//}
	double *x, *b, *val, *val_pad, *val_t;
	int n, *col, *col_pad, *col_t, sn, *snodes;
	unsigned long long nz;
	uint64_t *row, *row_pad, *row_t;

    const char* algo_type = argv[1];

	run("forward", algo_type,
     "/home/aromanov/devel/projects/sparse/matrices/bin/parabolic_fem.bin",
     "/home/aromanov/devel/projects/sparse/matrices/bin/parabolic_fem_snodes.bin",
		&n, &nz,
		&row, &col, &val,
		&row_pad, &col_pad, &val_pad,
		&row_t, &col_t, &val_t,
		&x, &b, &sn, &snodes);
	//run("forward", algo_type, "csr_6_upper.txt", "csr_6_upper_snodes.txt", &n, &nz, &row, &col, &val, &row_pad, &col_pad, &val_pad, &x, &b, &sn, &snodes);

	double* x_check = new double[n] { 0. };


	int* int_row = new int[n + 1];
	for (int i = 0; i <= n; ++i) {
		int_row[i] = (int)row[i];
	}

	compare("forward", n, int_row, col, val, x, b, x_check);
	
	free(x);
	free(b);
	free(val);
	free(col);
	free(row);
	if (strcmp(algo_type, "blas") == 0) {
		free(val_pad);
		free(col_pad);
		free(row_pad);
	}
	//if (strcmp(algo_type, "syncfree") == 0) {
	//	free(val_t);
	//	free(col_t);
	//	free(row_t);
	//}
	delete[] int_row;
	delete[] x_check;
	return 0;
}
