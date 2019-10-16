

#ifndef NN_OPERATOR_ASM_H_
#define NN_OPERATOR_ASM_H_

#include <stdint.h>
#include "nn_types.h"

#ifdef __XC__
extern "C" {
#endif

#ifdef __XS3A__


#ifndef USE_ASM_nn_mat_vec_mul_s8
#define USE_ASM_nn_mat_vec_mul_s8   (1)
#endif
void nn_mat_vec_mul_s8_asm(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y);



    
#ifndef USE_ASM_conv2d_deepin_deepout_relu
#define USE_ASM_conv2d_deepin_deepout_relu   (1)
#endif
void conv2d_deepin_deepout_relu_asm(
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
    const int16_t* scales);

int8_t* conv2d_deepin_deepout_relu_asm_patch(
    const int8_t* y,
    const int8_t* patch_k,
    const data16_t* biases_lo,
    const data16_t* biases_hi,
    const unsigned patch_row_incr,
    const unsigned kernel_row_incr,
    const int8_t* patch_x, 
    const unsigned rows,
    const unsigned row_maccs,          
    const unsigned chan_outs,          
    const unsigned kernel_advance,
    const int16_t* shifts,
    const int16_t* scales,
    const int16_t* add_vector);

#ifndef USE_ASM_conv2d_shallowin_deepout_relu
#define USE_ASM_conv2d_shallowin_deepout_relu   (1)
#endif
void conv2d_shallowin_deepout_relu_asm(
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
    const int16_t* scales);
    

int8_t* conv2d_shallowin_deepout_relu_asm_patch(
    int8_t* y,
    const int8_t* K,
    const unsigned c_out_groups,
    const uint32_t pad_mask,
    const int8_t* X,
    const unsigned rows,
    const data16_t* bias_lo,
    const data16_t* bias_hi,
    const int16_t* shifts,
    const int16_t* scales,
    const unsigned X_row_incr,
    const unsigned kernel_advance);


#ifndef USE_ASM_fc_deepin_shallowout_lin
#define USE_ASM_fc_deepin_shallowout_lin    (1)
#endif
void fc_deepin_shallowout_lin_asm(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);


#ifndef USE_ASM_maxpool2d_deep
#define USE_ASM_maxpool2d_deep    (1)
#endif
void maxpool2d_deep_asm(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);


#endif //__XS3A__

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_ASM_H_