


#include "unity.h"

#include <stdio.h>


void test_vpu_memcpy_case0();
void test_vpu_memcpy_case1();

void test_conv2d_deepin_deepout_1x1();
void test_conv2d_deepin_deepout_1x1_chans();
void test_conv2d_deepin_deepout_1x1_xsize();
void test_conv2d_deepin_deepout_3x3();
void test_conv2d_deepin_deepout_regions();

void test_conv2d_shallowin_deepout_1x1();
void test_conv2d_shallowin_deepout_1x1_chans();
void test_conv2d_shallowin_deepout_1x1_xsize();
void test_conv2d_shallowin_deepout_3x3();
void test_conv2d_shallowin_deepout_regions();

void test_conv2d_1x1_case0();
void test_conv2d_1x1_case1();
void test_conv2d_1x1_case2();

void test_conv2d_depthwise_case0();
void test_conv2d_depthwise_case1();
void test_conv2d_depthwise_case2();
void test_conv2d_depthwise_case3();
void test_conv2d_depthwise_case4();
void test_conv2d_depthwise_case5();

void test_fully_connected_16_case0();
void test_fully_connected_16_case1();
void test_fully_connected_16_case2();
void test_fully_connected_16_case3();
void test_fully_connected_16_case4();
void test_fully_connected_16_case5();
void test_fully_connected_16_case6();
void test_fully_connected_16_case7();
void test_fully_connected_16_case8();
void test_fully_connected_16_case9();
void test_fully_connected_16_case10();
void test_fully_connected_16_case11();

void test_fc_deepin_shallowout_16_case1();
void test_fc_deepin_shallowout_16_case2();
void test_fc_deepin_shallowout_16_case3();

void test_fc_deepin_shallowout_8_case1();
void test_fc_deepin_shallowout_8_case1_5();
void test_fc_deepin_shallowout_8_case2();
void test_fc_deepin_shallowout_8_case3();

void test_maxpool2d_case1();
void test_maxpool2d_case2();

void test_avgpool2d_2x2_case1();
void test_avgpool2d_2x2_case2();

void test_avgpool2d_case1();
void test_avgpool2d_case2();
void test_avgpool2d_case3();

void test_avgpool2d_global_case1();

void test_requantize_16_to_8_case0();
void test_requantize_16_to_8_case1();

void test_lookup8_case0();

void test_temp1();

void test_nn_compute_patch_v1_case0();

int main(void)
{
    int ret_val;
    printf("\n\n\n");

    UnityBegin("src\\test_vpu_memcpy.xc");
    RUN_TEST(test_vpu_memcpy_case0);
    RUN_TEST(test_vpu_memcpy_case1);
    ret_val = UnityEnd();
    printf("\n\n");

    UnityBegin("src\\test_conv2d_deepin_deepout.xc");
    RUN_TEST(test_conv2d_deepin_deepout_1x1);
    RUN_TEST(test_conv2d_deepin_deepout_1x1_chans);
    RUN_TEST(test_conv2d_deepin_deepout_1x1_xsize);
    RUN_TEST(test_conv2d_deepin_deepout_3x3);
    RUN_TEST(test_conv2d_deepin_deepout_regions);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_conv2d_shallowin_deepout.xc");
    RUN_TEST(test_conv2d_shallowin_deepout_1x1);
    RUN_TEST(test_conv2d_shallowin_deepout_1x1_chans);
    RUN_TEST(test_conv2d_shallowin_deepout_1x1_xsize);
    RUN_TEST(test_conv2d_shallowin_deepout_3x3);
    RUN_TEST(test_conv2d_shallowin_deepout_regions);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_conv2d_1x1.xc");
    RUN_TEST(test_conv2d_1x1_case0);
    RUN_TEST(test_conv2d_1x1_case1);
    RUN_TEST(test_conv2d_1x1_case2);
    ret_val = UnityEnd();
    printf("\n\n");

    UnityBegin("src\\test_conv2d_depthwise.xc");
    RUN_TEST(test_conv2d_depthwise_case0);
    RUN_TEST(test_conv2d_depthwise_case1);
    RUN_TEST(test_conv2d_depthwise_case2);
    RUN_TEST(test_conv2d_depthwise_case3);
    RUN_TEST(test_conv2d_depthwise_case4);
    RUN_TEST(test_conv2d_depthwise_case5);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_maxpool2d.xc");
    RUN_TEST(test_maxpool2d_case1);
    RUN_TEST(test_maxpool2d_case2);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_avgpool2d.xc");
    RUN_TEST(test_avgpool2d_case1);
    RUN_TEST(test_avgpool2d_case2);
    RUN_TEST(test_avgpool2d_case3);
    RUN_TEST(test_avgpool2d_2x2_case1);
    RUN_TEST(test_avgpool2d_2x2_case2);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_avgpool2d_global.xc");
    RUN_TEST(test_avgpool2d_global_case1);
    ret_val = UnityEnd();
    printf("\n\n");

    UnityBegin("src\\test_fc_deepin_shallowout_16.xc");
    RUN_TEST(test_fc_deepin_shallowout_16_case1);
    RUN_TEST(test_fc_deepin_shallowout_16_case2);
    RUN_TEST(test_fc_deepin_shallowout_16_case3);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_fully_/connected_16.xc");
    RUN_TEST(test_fully_connected_16_case0);
    RUN_TEST(test_fully_connected_16_case1);
    RUN_TEST(test_fully_connected_16_case2);
    RUN_TEST(test_fully_connected_16_case3);
    RUN_TEST(test_fully_connected_16_case4);
    RUN_TEST(test_fully_connected_16_case5);
    RUN_TEST(test_fully_connected_16_case6);
    RUN_TEST(test_fully_connected_16_case7);
    RUN_TEST(test_fully_connected_16_case8);
    RUN_TEST(test_fully_connected_16_case9);
    RUN_TEST(test_fully_connected_16_case10);
    RUN_TEST(test_fully_connected_16_case11);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_fc_deepin_shallowout_8.xc");
    RUN_TEST(test_fc_deepin_shallowout_8_case1);
    RUN_TEST(test_fc_deepin_shallowout_8_case1_5);
    RUN_TEST(test_fc_deepin_shallowout_8_case2);
    RUN_TEST(test_fc_deepin_shallowout_8_case3);
    ret_val = UnityEnd();
    printf("\n\n");


    UnityBegin("src\\test_requantize_16_to_8.xc");
    RUN_TEST(test_requantize_16_to_8_case0);
    RUN_TEST(test_requantize_16_to_8_case1);
    ret_val = UnityEnd();
    printf("\n\n");



    UnityBegin("src\\test_lookup8.xc");
    RUN_TEST(test_lookup8_case0);
    ret_val = UnityEnd();
    printf("\n\n");





    UnityBegin("src\\test_temp.local.xc");
    RUN_TEST(test_temp1);
    ret_val = UnityEnd();
    printf("\n\n");

    UnityBegin("src\\test_nn_compute_patch_v1.local.xc");
    RUN_TEST(test_nn_compute_patch_v1_case0);
    ret_val = UnityEnd();
    printf("\n\n");

    return ret_val;
}
