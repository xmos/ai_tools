


#include "Unity.h"


void test_nn_mat_vec_mul_s8();


int main(void)
{
  UnityBegin("src\\test_matrix_multiply.c");
  RUN_TEST(test_nn_mat_vec_mul_s8);

  return UnityEnd();
}
