
#ifndef NN_OP_ADVANCED_INLINE_H_
#define NN_OP_ADVANCED_INLINE_H_

#include <stdint.h>
#include "nn_types.h"

#include "nn_op_structs.h"

#ifdef __XC__
extern "C" {
#endif

static inline void nn_compute_hstrip_deep_padded(
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
        const int8_t* zero_point_vec)
{
#if defined(__XS3A__) && (USE_ASM_nn_compute_hstrip_deep_padded)

    nn_compute_hstrip_deep_padded_asm(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                pad_t, pad_b, pad_l_initial, pad_r_initial, x_v_stride,
                                k_cout_stride, y_h_stride, out_cols, zero_point_vec);

#else

    nn_compute_hstrip_deep_padded_c(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                pad_t, pad_b, pad_l_initial, pad_r_initial, x_v_stride,
                                k_cout_stride, y_h_stride, out_cols, zero_point_vec);

#endif
}




static inline void nn_compute_hstrip_tail_deep_padded(
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
        const channel_count_t C_out_tail)
{
#if defined(__XS3A__) && (USE_ASM_nn_compute_hstrip_tail_deep_padded)

    nn_compute_hstrip_tail_deep_padded_asm(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                            pad_t, pad_b, pad_l_initial, pad_r_initial, x_v_stride,
                                            k_cout_stride, y_h_stride, out_cols, zero_point_vec,
                                            C_out_tail);

#else

    nn_compute_hstrip_tail_deep_padded_c(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                        pad_t, pad_b, pad_l_initial, pad_r_initial, x_v_stride,
                                        k_cout_stride, y_h_stride, out_cols, zero_point_vec,
                                        C_out_tail);

#endif
}





static inline void nn_compute_hstrip_deep(
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
        const unsigned out_cols)
{
#if defined(__XS3A__) && (USE_ASM_nn_compute_hstrip_deep)

    nn_compute_hstrip_deep_asm(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                x_v_stride, k_cout_stride, y_h_stride, out_cols);

#else

    nn_compute_hstrip_deep_c(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                x_v_stride, k_cout_stride, y_h_stride, out_cols);

#endif
}





static inline void nn_compute_hstrip_tail_deep(
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
        const channel_count_t C_out_tail)
{
#if defined(__XS3A__) && (USE_ASM_nn_compute_hstrip_tail_deep)

    nn_compute_hstrip_tail_deep_asm(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                x_v_stride, k_cout_stride, y_h_stride, out_cols,
                                C_out_tail);

#else

    nn_compute_hstrip_tail_deep_c(Y, X, K, BSS, K_height, K_width, K_hori_stride, C_in,
                                x_v_stride, k_cout_stride, y_h_stride, out_cols,
                                C_out_tail);

#endif
}


#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_ADVANCED_INLINE_H_