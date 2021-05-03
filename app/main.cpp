#include "include/utils.h"

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cout << "Usage:\n\nalgo_type (can be \"base\", \"custom\", \"blas\", "
                 "\"barrier\", \"syncfree\", \"write_first\", \"mkl\")\n";
    std::cout << "full path to matrix-file (can be *.mtx, *.matrix, *.txt)"
              << std::endl;
    std::cout << "full path to supernodes-file (can be *.snodes)" << std::endl;
    std::cout << "nthreads " << std::endl;
    std::cout << "n right sides " << std::endl;
    std::cout << "check" << std::endl;
    return 1;
  }
  double *x, *b, *val, *val_pad, *val_t;
  int n, *col, *col_pad, *col_t, sn, *snodes;
  unsigned long long nz;
  uint64_t *row, *row_pad, *row_t;

  const char *algo_type = argv[1];
  const char *mtx_path = argv[2];
  const char *snodes_path = argv[3];
  const int nthreads = std::atoi(argv[4]);
  const int nrhs = std::atoi(argv[5]);

  run("forward", algo_type, mtx_path, snodes_path, &n, &nz, &row, &col, &val,
      &row_pad, &col_pad, &val_pad, &row_t, &col_t, &val_t, &x, &b, &sn,
      &snodes, nthreads, nrhs);

  if (argc >= 7){
    const char* is_check = argv[6];
    if (strcmp(is_check, "check") == 0){
      double *x_check = new double[n * nrhs]{0.};
      int *int_row = new int[n + 1];
      for (int i = 0; i <= n; ++i) {
        int_row[i] = (int)row[i];
      }
      compare("forward", n, int_row, col, val, x, b, x_check, nrhs);
    if (argc == 8){
      const char* is_verbose = argv[7];
      if (strcmp(is_verbose, "verbose") == 0){
        std::cout << "x_custom x_check\n";
        for (int i = 0; i < n * nrhs; ++i){
          std::cout << x[i] << " " << x_check[i] << "\n";
        }
      }
    }
      delete[] x_check;
      delete[] int_row;
    }
  }

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
  // if (strcmp(algo_type, "syncfree") == 0) {
  //	free(val_t);
  //	free(col_t);
  //	free(row_t);
  //}
  return 0;
}
