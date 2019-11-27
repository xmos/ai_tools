

#ifndef NN_OPERATOR_INLINE_H_
#define NN_OPERATOR_INLINE_H_

#include "nn_operator_c.h"
#include "nn_operator_asm.h"

#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif



static inline void nn_mat_vec_mul_s8(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y)
{
#if defined(__XS3A__) && (USE_ASM_nn_mat_vec_mul_s8)

    nn_mat_vec_mul_s8_asm(W, x, N_bands, N_chunks, shr, y);

#else

    nn_mat_vec_mul_s8_c(W, x, N_bands, N_chunks, shr, y);

#endif
}





static inline void conv2d_deepin_deepout_relu(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out, 
    const int32_t C_in,
    const int16_t* shifts, 
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_conv2d_deepin_deepout_relu)

    conv2d_deepin_deepout_relu_asm(K, B, X, Y, height, width, K_h, K_w, C_out, C_in, shifts, scales);

#else

    conv2d_deepin_deepout_relu_c(K, B, X, Y, height, width, K_h, K_w, C_out, C_in, shifts, scales);

#endif
}



static inline void conv2d_shallowin_deepout_relu(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out,
    const int16_t* shifts, 
    const int16_t* scales)
{
#if defined(__XS3A__) && (USE_ASM_conv2d_shallowin_deepout_relu)

    conv2d_shallowin_deepout_relu_asm(K, B, X, Y, height, width, K_h, K_w, C_out, shifts, scales);

#else

    conv2d_shallowin_deepout_relu_c(K, B, X, Y, height, width, K_h, K_w, C_out, shifts, scales);

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