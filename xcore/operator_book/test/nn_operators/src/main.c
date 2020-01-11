


#include "Unity.h"

#include <stdio.h>


void test_vpu_memcpy_case0();
void test_vpu_memcpy_case1();

void test_conv2d_deepin_deepout_1x1();

void test_conv2d_deepin_deepout_case0();
void test_conv2d_deepin_deepout_case1();
void test_conv2d_deepin_deepout_case2();
void test_conv2d_deepin_deepout_case3();
void test_conv2d_deepin_deepout_case4();
void test_conv2d_deepin_deepout_case5();
void test_conv2d_deepin_deepout_case6();

void test_conv2d_shallowin_deepout_case1();
void test_conv2d_shallowin_deepout_case2();
void test_conv2d_shallowin_deepout_case3();
void test_conv2d_shallowin_deepout_case4();
void test_conv2d_shallowin_deepout_case5();
void test_conv2d_shallowin_deepout();

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
  
  UnityBegin("src\\test_vpu_memcpy.xc");
  RUN_TEST(test_vpu_memcpy_case0);
  RUN_TEST(test_vpu_memcpy_case1);
  ret_val = UnityEnd();
  printf("\n\n");

  UnityBegin("src\\test_conv2d_deepin_deepout_1x1.xc");
  RUN_TEST(test_conv2d_deepin_deepout_1x1);
  ret_val = UnityEnd();
  printf("\n\n");


  UnityBegin("src\\test_conv2d_deepin_deepout.xc");
  RUN_TEST(test_conv2d_deepin_deepout_case1);
  RUN_TEST(test_conv2d_deepin_deepout_case2);
  RUN_TEST(test_conv2d_deepin_deepout_case3);
  RUN_TEST(test_conv2d_deepin_deepout_case4);
  RUN_TEST(test_conv2d_deepin_deepout_case5);
  RUN_TEST(test_conv2d_deepin_deepout_case6);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_conv2d_shallowin_deepout.xc");
  RUN_TEST(test_conv2d_shallowin_deepout_case1);
  RUN_TEST(test_conv2d_shallowin_deepout_case2);
  RUN_TEST(test_conv2d_shallowin_deepout_case3);
  RUN_TEST(test_conv2d_shallowin_deepout_case4);
  RUN_TEST(test_conv2d_shallowin_deepout_case5);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_maxpool2d_deep.xc");
  RUN_TEST(test_maxpool2d_deep_case1);
  RUN_TEST(test_maxpool2d_deep_case2);
  RUN_TEST(test_maxpool2d_deep_case3);
  RUN_TEST(test_maxpool2d_deep_case4);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_avgpool2d_deep.xc");
  RUN_TEST(test_avgpool2d_deep_case1);
  RUN_TEST(test_avgpool2d_deep_case2);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_fc_deepin_shallowout_16.xc");
  RUN_TEST(test_fc_deepin_shallowout_16_case1);
  RUN_TEST(test_fc_deepin_shallowout_16_case2);
  RUN_TEST(test_fc_deepin_shallowout_16_case3);
  ret_val = UnityEnd();
  printf("\n\n");

  
  UnityBegin("src\\test_fc_deepin_shallowout_8.xc");
  RUN_TEST(test_fc_deepin_shallowout_8_case1);
  RUN_TEST(test_fc_deepin_shallowout_8_case1_5);
  RUN_TEST(test_fc_deepin_shallowout_8_case2);
  RUN_TEST(test_fc_deepin_shallowout_8_case3);
  ret_val = UnityEnd();
  printf("\n\n");


  return ret_val;
}
