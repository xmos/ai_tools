

#ifndef NN_OPERATOR_INLINE_H_
#define NN_OPERATOR_INLINE_H_

#include "nn_operator_c.h"
#include "nn_operator_asm.h"

#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif


static inline void conv2d_deepin_deepout(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales)
{
    const unsigned block_count = params->block_count;
    for(int i = 0; i < block_count; i++){
        conv2d_deepin_deepout_block(
            Y, params, params->blocks[i],
            X, K, B, shifts, scales
        );
    }
}

static inline void conv2d_deepin_deepout_block(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const nn_conv2d_dido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_conv2d_deepin_deepout_block)

    conv2d_deepin_deepout_block_asm(Y, params, block, X, K, B, shifts, scales);

#else

    conv2d_deepin_deepout_block_c(Y, params, block, X, K, B, shifts, scales);

#endif
}


static inline void conv2d_shallowin_deepout(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales)
{
    const unsigned block_count = params->block_count;
    for(int i = 0; i < block_count; i++){
        conv2d_shallowin_deepout_block(
            Y, params, params->blocks[i],
            X, K, B, shifts, scales
        );
    }
}


static inline void conv2d_shallowin_deepout_block(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_conv2d_shallowin_deepout_block)

    conv2d_shallowin_deepout_block_asm(Y, params, block, X, K, B, shifts, scales);

#else

    conv2d_shallowin_deepout_block_c(Y, params, block, X, K, B, shifts, scales);

#endif
}



static inline void maxpool2d_deep(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in)
{
#if defined(__XS3A__) && (USE_ASM_maxpool2d_deep)

    maxpool2d_deep_asm(X, Y, height, width, C_in);

#else

    maxpool2d_deep_c(X, Y, height, width, C_in);

#endif
}



static inline void avgpool2d_deep(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in)
{
#if defined(__XS3A__) && (USE_ASM_avgpool2d_deep)

    avgpool2d_deep_asm(X, Y, height, width, C_in);

#else

    avgpool2d_deep_c(X, Y, height, width, C_in);

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


#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_INLINE_H_