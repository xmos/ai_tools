

#ifndef NN_OPERATOR_INLINE_H_
#define NN_OPERATOR_INLINE_H_

#include "nn_op_structs.h"
#include "nn_operator_c.h"
#include "nn_operator_asm.h"

#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif




static inline void conv2d_deepin_deepout_block(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const nn_conv2d_dido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_conv2d_deepin_deepout_block)

    conv2d_deepin_deepout_block_asm(Y, params, block, X, K, scales);

#else

    conv2d_deepin_deepout_block_c(Y, params, block, X, K, scales);

#endif
}

static inline void conv2d_deepin_deepout(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales)
{
    const unsigned block_count = params->block_count;
    for(int i = 0; i < block_count; i++){
        conv2d_deepin_deepout_block(
            Y, params, &params->blocks[i],
            X, K, scales
        );
    }
}


static inline void conv2d_shallowin_deepout_block(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_conv2d_shallowin_deepout_block)

    conv2d_shallowin_deepout_block_asm(Y, params, block, X, K, scales);

#else

    conv2d_shallowin_deepout_block_c(Y, params, block, X, K, scales);

#endif
}


static inline void conv2d_shallowin_deepout(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales)
{
    const unsigned block_count = params->block_count;
    for(int i = 0; i < block_count; i++){
        conv2d_shallowin_deepout_block(
            Y, params, &params->blocks[i],
            X, K, scales
        );
    }
}



static inline void conv2d_1x1(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const data16_t* BSS,
    const nn_conv2d_1x1_plan_t* plan)
{
#if defined(__XS3A__) && (USE_ASM_conv2d_1x1)
    conv2d_1x1_asm(Y, X, K, BSS, plan);
#else
    conv2d_1x1_c(Y, X, K, BSS, plan);
#endif
}




static inline void maxpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_window_op_plan_t* plan)
{
#if defined(__XS3A__) && (USE_ASM_maxpool2d)

    maxpool2d_asm(Y, X, plan);

#else

    maxpool2d_c(Y, X, plan);

#endif
}



static inline void avgpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan)
{
#if defined(__XS3A__) && (USE_ASM_avgpool2d)

    switch(plan->impl){
        case AVGPOOL2D_2X2:
            avgpool2d_2x2_asm(Y, X, plan);
            break;
        default:
            avgpool2d_asm(Y, X, plan);
            break;
    }

#else

    avgpool2d_c(Y, X, plan);

#endif
}

static inline void avgpool2d_global(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans,
    const int32_t  bias,
    const uint32_t shift,
    const uint32_t scale)
{
#if defined(__XS3A__) && (USE_ASM_avgpool2d)

    avgpool2d_global_asm(Y, X, x_height, x_width, x_chans, bias, shift, scale);

#else

    avgpool2d_global_c(Y, X, x_height, x_width, x_chans, bias, shift, scale);

#endif
}




static inline void fc_deepin_shallowout_16(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_fc_deepin_shallowout_16)

    fc_deepin_shallowout_16_asm(W, B, X, Y, C_out, C_in, shifts, scales);

#else

    fc_deepin_shallowout_16_c(W, B, X, Y, C_out, C_in, shifts, scales);

#endif
}


static inline void fully_connected_16(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const nn_fully_connected_plan_t* plan)
{
#if defined(__XS3A__) && (USE_ASM_fully_connected_16)

    fully_connected_16_asm(Y, W, X, BSS, plan);

#else

    fully_connected_16_c(Y, W, X, BSS, plan);

#endif
}



static inline void fc_deepin_shallowout_8(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_fc_deepin_shallowout_8)

    fc_deepin_shallowout_8_asm(W, B, X, Y, C_out, C_in, shifts, scales);

#else

    fc_deepin_shallowout_8_c(W, B, X, Y, C_out, C_in, shifts, scales);

#endif
}





static inline void argmax_16(
    const int16_t* A,
    int32_t* C,
    const int32_t N)
{
#if defined(__XS3A__) && (USE_ASM_argmax_16)

    argmax_16_asm(A, C, N);

#else

    argmax_16_c(A, C, N);

#endif
}


static inline void requantize_16_to_8(
    int8_t* y,
    const int16_t* x,
    const unsigned n)
{
#if defined(__XS3A__) && (USE_ASM_requantize_16_to_8)

    requantize_16_to_8_asm(y, x, n);

#else

    requantize_16_to_8_c(y, x, n);

#endif
}


static inline void lookup8(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned length)
{
#if defined(__XS3A__) && (USE_ASM_lookup8)

    lookup8_asm(Y, X, lut, length);

#else

    lookup8_c(Y, X, lut, length);

#endif
}





#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_INLINE_H_
