

#include <stdio.h>

#include "unity.h"

#define CALL(F)     do { void F(); F(); } while(0)

int main(void) {
    int ret_val;

    UNITY_BEGIN();

    CALL(test_vpu_memcpy);

    CALL(test_nn_conv2d_hstrip_deep_padded);
    CALL(test_nn_conv2d_hstrip_deep);
    CALL(test_nn_conv2d_hstrip_tail_deep_padded);
    CALL(test_nn_conv2d_hstrip_tail_deep);

    CALL(test_nn_conv2d_hstrip_shallowin_padded);
    CALL(test_nn_conv2d_hstrip_shallowin);
    CALL(test_nn_conv2d_hstrip_tail_shallowin_padded);
    CALL(test_nn_conv2d_hstrip_tail_shallowin);

    CALL(test_conv2d_deep);
    CALL(test_conv2d_shallowin);
    CALL(test_conv2d_im2col);
    CALL(test_conv2d_1x1);
    CALL(test_conv2d_depthwise);

    CALL(test_maxpool2d);
    CALL(test_avgpool2d);
    CALL(test_avgpool2d_global);

    CALL(test_fully_connected_16);
    CALL(test_fully_connected_8);
    CALL(test_requantize_16_to_8);
    CALL(test_lookup8);

    CALL(test_add_elementwise);

    CALL(test_bsign_8);
    CALL(test_pad);
    CALL(test_bnn_conv2d_bin);
    CALL(test_bnn_conv2d_int8);
    CALL(test_bnn_conv2d_bin_SISO);

  return UNITY_END();
}
