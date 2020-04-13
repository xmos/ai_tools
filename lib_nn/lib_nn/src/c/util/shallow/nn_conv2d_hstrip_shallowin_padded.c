

#include "nn_operator.h"
#include "../../../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"
#include "../../vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define ADDR(V, INDEX)      &V[((int)(INDEX))]

WEAK_FUNC
void nn_conv2d_hstrip_shallowin_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec)
{
    xs3_vpu vpu;

    vpu_vector_t vec_vr;
    vpu_vector_t vec_tmp;
    vpu_vector_t adj_bias_hi;
    vpu_vector_t adj_bias_lo;

    const mem_stride_t window_h_stride = K_h_stride * C_in;
    const mem_stride_t k_cout_str = K_h * VPU_INT8_EPV;

    //Number of rows to actually be computed in a patch
    const unsigned patch_rows = K_h - pad_t - pad_b;

    VSETC(&vpu, MODE_S8);

    //Load Biases for current C_out group
    VLDD(&vpu, BSS->bias_hi);
    VLDR(&vpu, BSS->bias_lo);

    VLDC(&vpu, zero_point_vec);

    const nn_tensor_t* K_patch_start = ADDR(K, pad_t * VPU_INT8_EPV);
    X = ADDR(X, pad_t * x_v_stride);

    //Adjust for bias at top
    for(int row = pad_t; row; row--){
        const nn_tensor_t* K_tmp = K;
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            VLMACCR(&vpu, K_tmp);
            K_tmp = ADDR(K_tmp, -k_cout_str);
        }

        K = ADDR(K, VPU_INT8_EPV);
    }

    K = ADDR(K, VPU_INT8_EPV * patch_rows);

    for(int row = pad_b; row; row--){
        const nn_tensor_t* K_tmp = K;

        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            VLMACCR(&vpu, K_tmp);
            K_tmp = ADDR(K_tmp, -k_cout_str);
        }

        K = ADDR(K, VPU_INT8_EPV);
    }

    //Store adjusted accumulators
    VSTD(&vpu, &adj_bias_hi.u16[0]);
    VSTR(&vpu, &adj_bias_lo.u16[0]);

    int pad_l = pad_l_initial * C_in;
    int pad_r = pad_r_initial * C_in;

    int pad_l_relu = (pad_l > 0)? pad_l : 0;
    int pad_r_relu = (pad_r > 0)? pad_r : 0;

    uint32_t pad_mask = 32;

    pad_mask -= pad_l_relu;
    pad_mask -= pad_r_relu;

    pad_mask = ((1<<pad_mask)-1) << pad_l_relu;


    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K_patch_start;

        //Initialize accumulators
        VLDD(&vpu, &adj_bias_hi.u16[0]);
        VLDR(&vpu, &adj_bias_lo.u16[0]);

        VLDC(&vpu, zero_point_vec);
        VSTC(&vpu, vec_tmp.s8);

        // These rows are between top and bottom padding
        for(int pr = patch_rows; pr; pr--){

            VSTR(&vpu, vec_vr.s16);
            VLDR(&vpu, patch_X);
            VSTRPV(&vpu, vec_tmp.s8, pad_mask);
            VLDC(&vpu, vec_tmp.s8);
            VLDR(&vpu, vec_vr.s16);
            patch_X = ADDR(patch_X, x_v_stride);

            const nn_tensor_t* K_tmp = patch_K;

            for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
                VLMACCR(&vpu, K_tmp);
                K_tmp = ADDR(K_tmp, -k_cout_str);
            }
            
            patch_K = ADDR(patch_K, VPU_INT8_EPV);
        }
        
        //Done accumulating for the current patch

        //Set mode to 16-bit
        VSETC(&vpu, MODE_S16);

        //Saturate to 16-bit values
        VLSAT(&vpu, BSS->shift1);

        //Load scales into vC
        VLDC(&vpu, BSS->scale);
        VSTR(&vpu, vec_tmp.s16);
        VCLRDR(&vpu);
        VLMACC(&vpu, vec_tmp.s16);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSS->shift2);

        //Store result in Y
        const unsigned mask16 = 0xFFFF;
        VSTRPV(&vpu, Y, mask16);

        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);

        //Now make adjustments to pad_l and pad_r
        pad_l -= window_h_stride;
        pad_r += window_h_stride;

        int pad_l_relu = (pad_l > 0)? pad_l : 0;
        int pad_r_relu = (pad_r > 0)? pad_r : 0;

        pad_mask = 32;
        pad_mask -= pad_l_relu;
        pad_mask -= pad_r_relu;

        pad_mask = ((1<<pad_mask)-1) << pad_l_relu;
    }
}








