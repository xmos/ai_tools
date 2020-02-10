

#ifndef NN_OPERATOR_ASM_H_
#define NN_OPERATOR_ASM_H_

#include <stdint.h>
#include "nn_types.h"

#include "nn_op_structs.h"

#ifdef __XC__
extern "C" {
#endif

#ifdef __XS3A__


#ifndef USE_ASM_conv2d_deepin_deepout_block
#define USE_ASM_conv2d_deepin_deepout_block     (1)
#endif
void conv2d_deepin_deepout_block_asm(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const nn_conv2d_dido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales);

 
#ifndef USE_ASM_conv2d_shallowin_deepout_block
#define USE_ASM_conv2d_shallowin_deepout_block     (1)
#endif
void conv2d_shallowin_deepout_block_asm(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales);   

#ifndef USE_ASM_fc_deepin_shallowout_16
#define USE_ASM_fc_deepin_shallowout_16    (1)
#endif
void fc_deepin_shallowout_16_asm(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);

#ifndef USE_ASM_fully_connected_16
#define USE_ASM_fully_connected_16      (1)
#endif
void fully_connected_16_asm(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const unsigned C_in, 
    const unsigned C_out);


#ifndef USE_ASM_fc_deepin_shallowout_8
#define USE_ASM_fc_deepin_shallowout_8    (1)
#endif
void fc_deepin_shallowout_8_asm(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);


// #ifndef USE_ASM_maxpool2d_deep
// #define USE_ASM_maxpool2d_deep    (1)
// #endif
// void maxpool2d_deep_asm(
//     const int8_t* X, 
//     int8_t* Y,
//     const int32_t height, 
//     const int32_t width,
//     const int32_t C_in);

#ifndef USE_ASM_maxpool2d
#define USE_ASM_maxpool2d       (1)
#endif
void maxpool2d_asm(
    int8_t* Y,
    const int8_t* X, 
    const nn_window_op_plan_t* plan);

#ifndef USE_ASM_avgpool2d
#define USE_ASM_avgpool2d      (1)
#endif
void avgpool2d_asm(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan);

void avgpool2d_2x2_asm(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan);


#ifndef USE_ASM_avgpool2d_global
#define USE_ASM_avgpool2d_global    (1)
#endif
void avgpool2d_global_asm(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans,
    const int32_t  bias,
    const uint32_t shift,
    const uint32_t scale);


#ifndef USE_ASM_requantize_16_to_8
#define USE_ASM_requantize_16_to_8    (1)
#endif
void requantize_16_to_8_asm(
    int8_t* y,
    const int16_t* x,
    const unsigned n);


#endif //__XS3A__

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_ASM_H_