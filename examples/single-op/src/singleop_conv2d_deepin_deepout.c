#include "nn_operator.h"
#include "singleop_conv2d_deepin_deepout.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

const int8_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_weights[1 * 3 * 3 * 1 * 16 * 32] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_WEIGHTS};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_biases_padded[3 * 3 * 2 * 16] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_BIASES_PADDED};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_shift_scale[2 * 16] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_SHIFT_SCALE};

void singleop_conv2d_deepin_deepout(const conv2d_deepin_deepout_input_t *conv2d_input_int8, conv2d_deepin_deepout_identity_t *Identity_int8)
{

     conv2d_deepin_deepout_relu(XC_conv2d_deepin_deepout_relu_weights, (data16_t *)XC_conv2d_deepin_deepout_relu_biases_padded, (int8_t *)conv2d_input_int8, Identity_int8, 5, 5, 3, 3, 16, 32, (int16_t*) &XC_conv2d_deepin_deepout_relu_shift_scale[0], (int16_t*) &XC_conv2d_deepin_deepout_relu_shift_scale[16]);
}
