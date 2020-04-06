
#ifndef NN_OP_ADVANCED_ASM_H_
#define NN_OP_ADVANCED_ASM_H_

#include <stdint.h>
#include "nn_types.h"

#include "nn_op_structs.h"

#ifdef __XC__
extern "C" {
#endif

#if defined(__XS3A__)

#ifndef DISABLE_ASM
#define DISABLE_ASM (0)
#endif

#if !defined(USE_ASM_nn_compute_hstrip_deep_padded) && !(DISABLE_ASM)
#define USE_ASM_nn_compute_hstrip_deep_padded     (1)
#endif
void nn_compute_hstrip_deep_padded_asm(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bss_block_t* BSS,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec);



#if !defined(USE_ASM_nn_compute_hstrip_tail_deep_padded) && !(DISABLE_ASM)
#define USE_ASM_nn_compute_hstrip_tail_deep_padded     (1)
#endif
void nn_compute_hstrip_tail_deep_padded_asm(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bss_block_t* BSS,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail);



#if !defined(USE_ASM_nn_compute_hstrip_deep) && !(DISABLE_ASM)
#define USE_ASM_nn_compute_hstrip_deep     (1)
#endif
void nn_compute_hstrip_deep_asm(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bss_block_t* BSS,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols);




#if !defined(USE_ASM_nn_compute_hstrip_tail_deep) && !(DISABLE_ASM)
#define USE_ASM_nn_compute_hstrip_tail_deep     (1)
#endif
void nn_compute_hstrip_tail_deep_asm(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bss_block_t* BSS,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail);

#endif //defined(__XS3A__)

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_ADVANCED_ASM_H_