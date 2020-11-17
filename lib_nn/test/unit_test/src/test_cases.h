#pragma once

void test_vpu_memcpy();

void test_nn_conv2d_hstrip_deep_padded();
void test_nn_conv2d_hstrip_deep();
void test_nn_conv2d_hstrip_tail_deep_padded();
void test_nn_conv2d_hstrip_tail_deep();

void test_nn_conv2d_hstrip_shallowin_padded();
void test_nn_conv2d_hstrip_shallowin();
void test_nn_conv2d_hstrip_tail_shallowin_padded();
void test_nn_conv2d_hstrip_tail_shallowin();

void test_conv2d_deep();
void test_conv2d_shallowin();
void test_conv2d_im2col();
void test_conv2d_shallowin_deepout();
void test_conv2d_1x1();
void test_conv2d_depthwise();
void test_fully_connected_16();
void test_fully_connected_8();
void test_maxpool2d();
void test_avgpool2d_2x2();
void test_avgpool2d();
void test_avgpool2d_global();
void test_requantize_16_to_8();
void test_lookup8();
void test_add_elementwise();

void test_bnn_conv2d_bin();
void test_bnn_conv2d_bin_SISO();
void test_bnn_conv2d_int8();
void test_bnn_conv2d_int8_SISO();
void test_pad();
void test_bsign_8();

void test_bnn_conv2d_quant();
