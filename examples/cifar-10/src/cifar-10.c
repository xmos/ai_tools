#include "nn_operator.h"
#include "cifar-10.h"
#include "argmax_16.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

const int8_t WORD_ALIGNED XC_conv2d_shallowin_deepout_relu_weights[32 * 5 * 8 * 4] = {XC_CONV2D_SHALLOWIN_DEEPOUT_RELU_WEIGHTS};
const int16_t WORD_ALIGNED XC_conv2d_shallowin_deepout_relu_biases[2 * 32] = {XC_CONV2D_SHALLOWIN_DEEPOUT_RELU_BIASES};
const int8_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_weights[32 * 5 * 5 * 32] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_WEIGHTS};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_biases[2 * 32] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_BIASES};
const int8_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_1_weights[64 * 5 * 5 * 32] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_1_WEIGHTS};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_1_biases[2 * 64] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_1_BIASES};
const int8_t WORD_ALIGNED XC_fc_deepin_shallowout_final_weights[10 * 1024] = {XC_FC_DEEPIN_SHALLOWOUT_FINAL_WEIGHTS};
const int32_t WORD_ALIGNED XC_fc_deepin_shallowout_final_biases[10] = {XC_FC_DEEPIN_SHALLOWOUT_FINAL_BIASES};
const int16_t WORD_ALIGNED XC_conv2d_shallowin_deepout_relu_shift_scale[2 * 32] = {XC_CONV2D_SHALLOWIN_DEEPOUT_RELU_SHIFT_SCALE};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_shift_scale[2 * 32] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_SHIFT_SCALE};
const int16_t WORD_ALIGNED XC_conv2d_deepin_deepout_relu_1_shift_scale[2 * 64] = {XC_CONV2D_DEEPIN_DEEPOUT_RELU_1_SHIFT_SCALE};
const int16_t WORD_ALIGNED XC_fc_deepin_shallowout_final_shift_scale[2 * 10] = {XC_FC_DEEPIN_SHALLOWOUT_FINAL_SHIFT_SCALE};

void xcore_model_quant(const xc_conv2d_shallowin_deepout_relu_input_t *XC_conv2d_shallowin_deepout_relu_input, xc_argmax_16_output_t *XC_argmax_16_output)
{
     int16_t WORD_ALIGNED XC_fc_deepin_shallowout_final_output[1 * 10];
     int8_t WORD_ALIGNED sequential_max_pooling2d_MaxPool[1 * 16 * 16 * 32];
     int8_t WORD_ALIGNED sequential_max_pooling2d_1_MaxPool[1 * 8 * 8 * 32];
     int8_t WORD_ALIGNED sequential_max_pooling2d_2_MaxPool[1 * 4 * 4 * 64];
     int8_t WORD_ALIGNED sequential_re_lu_Relu[1 * 32 * 32 * 32];
     int8_t WORD_ALIGNED sequential_re_lu_1_Relu[1 * 16 * 16 * 32];
     int8_t WORD_ALIGNED sequential_re_lu_2_Relu[1 * 8 * 8 * 64];

     //conv2d_shallowin_deepout_relu(XC_conv2d_shallowin_deepout_relu_weights, (data16_t *)XC_conv2d_shallowin_deepout_relu_biases, XC_conv2d_shallowin_deepout_relu_input, sequential_re_lu_Relu, 32, 32, 5, 8, 32, (int16_t*) &XC_conv2d_shallowin_deepout_relu_shift_scale[0], (int16_t*) &XC_conv2d_shallowin_deepout_relu_shift_scale[32]);
     maxpool2d_deep(sequential_re_lu_Relu, sequential_max_pooling2d_MaxPool, 32, 32, 32);
     conv2d_deepin_deepout_relu(XC_conv2d_deepin_deepout_relu_weights, (data16_t *)XC_conv2d_deepin_deepout_relu_biases, sequential_max_pooling2d_MaxPool, sequential_re_lu_1_Relu, 16, 16, 5, 5, 32, 32, (int16_t*) &XC_conv2d_deepin_deepout_relu_shift_scale[0], (int16_t*) &XC_conv2d_deepin_deepout_relu_shift_scale[32]);
     maxpool2d_deep(sequential_re_lu_1_Relu, sequential_max_pooling2d_1_MaxPool, 16, 16, 32);
     conv2d_deepin_deepout_relu(XC_conv2d_deepin_deepout_relu_1_weights, (data16_t *)XC_conv2d_deepin_deepout_relu_1_biases, sequential_max_pooling2d_1_MaxPool, sequential_re_lu_2_Relu, 8, 8, 5, 5, 64, 32, (int16_t*) &XC_conv2d_deepin_deepout_relu_1_shift_scale[0], (int16_t*) &XC_conv2d_deepin_deepout_relu_1_shift_scale[64]);
     maxpool2d_deep(sequential_re_lu_2_Relu, sequential_max_pooling2d_2_MaxPool, 8, 8, 64);
     fc_deepin_shallowout_lin(XC_fc_deepin_shallowout_final_weights, XC_fc_deepin_shallowout_final_biases, sequential_max_pooling2d_2_MaxPool, XC_fc_deepin_shallowout_final_output, 10, 1024, (uint16_t*) &XC_fc_deepin_shallowout_final_shift_scale[0], (int16_t*) &XC_fc_deepin_shallowout_final_shift_scale[10]);
     argmax_16(XC_fc_deepin_shallowout_final_output, (int32_t *) XC_argmax_16_output, 10);
}
