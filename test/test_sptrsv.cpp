#include "include/utils.h"
#include <gtest/gtest.h>

class testFixture : public ::testing::Test {
public:
  int n;
  unsigned long long nz;
  uint64_t *row;
  int *col;
  double *val;
  double t_read_csr = -2.;
  int sn;
  int *snodes;
  testFixture() {
    const char *mtx_name = "./matrices/bin/parabolic_fem.matrix";
    const char *snode_name = "./matrices/bin/parabolic_fem.snodes";
    t_read_csr = read_csr(mtx_name, &n, &nz, &row, &col, &val);
    read_snodes(snode_name, &sn, &snodes);
  }

  void SetUp() {
    // code here will execute just before the test ensues
  }

  ~testFixture() {
    free(val);
    free(col);
    free(row);
    free(snodes);
  }
};

TEST_F(testFixture, canReadMatrix) {
  ASSERT_NE(t_read_csr, -1.);
  ASSERT_EQ(n, 525825);
  ASSERT_EQ(nz, 30626512);
  ASSERT_EQ(row[1], 7);
  ASSERT_EQ(col[1], 42);
  ASSERT_EQ(val[1], -0.15811300340759463);
  ASSERT_EQ(sn, 106842);
  ASSERT_EQ(snodes[1], 1);
}
