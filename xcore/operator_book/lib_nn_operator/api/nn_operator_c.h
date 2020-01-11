

#ifndef NN_OPERATOR_C_H_
#define NN_OPERATOR_C_H_

#include <stdint.h>

#include "nn_op_structs.h"

#ifdef __XC__
extern "C" {
#endif




void conv2d_deepin_deepout_block_c(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const nn_conv2d_dido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* shifts,
    const int16_t* scales);


void conv2d_shallowin_deepout_block_c(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* shifts,
    const int16_t* scales);



void maxpool2d_deep_c(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);




void avgpool2d_deep_c(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);




void fc_deepin_shallowout_16_c(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);




void fc_deepin_shallowout_8_c(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);



void argmax_16_c(
    const int16_t* A,
    int32_t* C,
    const int32_t N);

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_C_H_