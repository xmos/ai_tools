


#include "Unity.h"

#include <stdio.h>


void test_nn_mat_vec_mul_s8_case1();
void test_nn_mat_vec_mul_s8_case2();

void test_conv2d_deepin_deepout_relu_case1();

void test_conv2d_shallowin_deepout_relu_case1();

void test_fc_deepin_shallowout_lin_case1();

void test_maxpool2d_deep_case1();
void test_maxpool2d_deep_case2();
void test_maxpool2d_deep_case3();
void test_maxpool2d_deep_case4();


int main(void)
{
  int ret_val;

  UnityBegin("src\\test_nn_mat_vec_mul_s8.xc");
  RUN_TEST(test_nn_mat_vec_mul_s8_case1);
  RUN_TEST(test_nn_mat_vec_mul_s8_case2);
  ret_val = UnityEnd();
  printf("\n\n");


  UnityBegin("src\\test_conv2d_deepin_deepout_relu.xc");
  RUN_TEST(test_conv2d_deepin_deepout_relu_case1);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_conv2d_shallowin_deepout_relu.xc");
  RUN_TEST(test_conv2d_shallowin_deepout_relu_case1);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_fc_deepin_shallowout_lin.xc");
  RUN_TEST(test_fc_deepin_shallowout_lin_case1);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_maxpool2d_deep.xc");
  RUN_TEST(test_maxpool2d_deep_case1);
  RUN_TEST(test_maxpool2d_deep_case2);
  RUN_TEST(test_maxpool2d_deep_case3);
  RUN_TEST(test_maxpool2d_deep_case4);
  ret_val = UnityEnd();
  printf("\n\n");


  return ret_val;
}
