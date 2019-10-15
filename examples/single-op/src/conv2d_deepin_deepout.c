#include "nn_operator.h"
#include "conv2d_deepin_deepout.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

const int8_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_weights[1 * 5 * 5 * 1 * 16 * 32] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_WEIGHTS};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_biases[2 * 16] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_BIASES};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_shift_scale[2 * 16] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_SHIFT_SCALE};

void conv2d_deepin_deepout(const conv2d_input_int8_t *conv2d_input_int8, identity_int8_t *Identity_int8)
{

     conv2d_deepin_deepout_relu(XC_conv2d_deepin_deepout_relu_weights, (data16_t *)XC_conv2d_deepin_deepout_relu_biases, (int8_t *)conv2d_input_int8, Identity_int8, 5, 5, 5, 5, 16, 32, (int16_t*) &XC_conv2d_deepin_deepout_relu_shift_scale[0], (int16_t*) &XC_conv2d_deepin_deepout_relu_shift_scale[16]);
}
