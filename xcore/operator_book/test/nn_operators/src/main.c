


#include "Unity.h"

#include <stdio.h>


void test_nn_mat_vec_mul_s8_case1();
void test_nn_mat_vec_mul_s8_case2();

void test_something();
void test_conv2d_deepin_deepout_relu_case1();
void test_conv2d_deepin_deepout_relu_case2();
void test_conv2d_deepin_deepout_relu_case3();
void test_conv2d_deepin_deepout_relu_case4();

void test_conv2d_shallowin_deepout_relu_case1();
void test_conv2d_shallowin_deepout_relu_case2();
void test_conv2d_shallowin_deepout_relu_case3();
void test_conv2d_shallowin_deepout_relu_case4();

void test_fc_deepin_shallowout_16_case1();
void test_fc_deepin_shallowout_16_case2();
void test_fc_deepin_shallowout_16_case3();

void test_fc_deepin_shallowout_8_case1();
void test_fc_deepin_shallowout_8_case1_5();
void test_fc_deepin_shallowout_8_case2();
void test_fc_deepin_shallowout_8_case3();

void test_maxpool2d_deep_case1();
void test_maxpool2d_deep_case2();
void test_maxpool2d_deep_case3();
void test_maxpool2d_deep_case4();

void test_avgpool2d_deep_case1();
void test_avgpool2d_deep_case2();


int main(void)
{
  int ret_val;
  printf("\n\n\n");

//   UnityBegin("src\\test_nn_mat_vec_mul_s8.xc");
//   RUN_TEST(test_nn_mat_vec_mul_s8_case1);
//   RUN_TEST(test_nn_mat_vec_mul_s8_case2);
//   ret_val = UnityEnd();
//   printf("\n\n");


  UnityBegin("src\\test_conv2d_deepin_deepout_relu.xc");
  RUN_TEST(test_something);
//   RUN_TEST(test_conv2d_deepin_deepout_relu_case1);
//   RUN_TEST(test_conv2d_deepin_deepout_relu_case2);
//   RUN_TEST(test_conv2d_deepin_deepout_relu_case3);
//   RUN_TEST(test_conv2d_deepin_deepout_relu_case4);
  ret_val = UnityEnd();
  printf("\n\n");

  
//   UnityBegin("src\\test_conv2d_shallowin_deepout_relu.xc");
//   RUN_TEST(test_conv2d_shallowin_deepout_relu_case1);
//   RUN_TEST(test_conv2d_shallowin_deepout_relu_case2);
//   RUN_TEST(test_conv2d_shallowin_deepout_relu_case3);
//   RUN_TEST(test_conv2d_shallowin_deepout_relu_case4);
//   ret_val = UnityEnd();
//   printf("\n\n");

  
//   UnityBegin("src\\test_maxpool2d_deep.xc");
//   RUN_TEST(test_maxpool2d_deep_case1);
//   RUN_TEST(test_maxpool2d_deep_case2);
//   RUN_TEST(test_maxpool2d_deep_case3);
//   RUN_TEST(test_maxpool2d_deep_case4);
//   ret_val = UnityEnd();
//   printf("\n\n");

  
//   UnityBegin("src\\test_avgpool2d_deep.xc");
//   RUN_TEST(test_avgpool2d_deep_case1);
//   RUN_TEST(test_avgpool2d_deep_case2);
//   ret_val = UnityEnd();
//   printf("\n\n");

  
//   UnityBegin("src\\test_fc_deepin_shallowout_16.xc");
//   RUN_TEST(test_fc_deepin_shallowout_16_case1);
//   RUN_TEST(test_fc_deepin_shallowout_16_case2);
//   RUN_TEST(test_fc_deepin_shallowout_16_case3);
//   ret_val = UnityEnd();
//   printf("\n\n");

  
//   UnityBegin("src\\test_fc_deepin_shallowout_8.xc");
//   RUN_TEST(test_fc_deepin_shallowout_8_case1);
//   RUN_TEST(test_fc_deepin_shallowout_8_case1_5);
//   RUN_TEST(test_fc_deepin_shallowout_8_case2);
//   RUN_TEST(test_fc_deepin_shallowout_8_case3);
//   ret_val = UnityEnd();
//   printf("\n\n");


  return ret_val;
}
