

#include <stdio.h>

#include "unity.h"
#include "test_cases.h"

int main(void) {
    int ret_val;

    UNITY_BEGIN();

    test_vpu_memcpy();

    test_nn_conv2d_hstrip_deep_padded();
    test_nn_conv2d_hstrip_deep();
    test_nn_conv2d_hstrip_tail_deep_padded();
    test_nn_conv2d_hstrip_tail_deep();

    test_nn_conv2d_hstrip_shallowin_padded();
    test_nn_conv2d_hstrip_shallowin();
    test_nn_conv2d_hstrip_tail_shallowin_padded();
    test_nn_conv2d_hstrip_tail_shallowin();

    test_conv2d_deep();
    test_conv2d_shallowin();
    test_conv2d_im2col();
    test_conv2d_1x1();
    test_conv2d_depthwise();

    test_maxpool2d();
    test_avgpool2d();
    test_avgpool2d_global();

    test_fully_connected_16();
    test_fully_connected_8();
    test_requantize_16_to_8();
    test_lookup8();

    test_add_elementwise();

    test_bsign_8();
    test_pad();
    test_bnn_conv2d_bin();
    test_bnn_conv2d_bin_SISO(); //disabled until refactored to reduce memory usage
    test_bnn_conv2d_int8();

    test_bnn_conv2d_quant();

  return UNITY_END();
}
