

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

void nn_conv2d_hstrip_shallowin_c(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols)
{
    xs3_vpu vpu;
    vpu_vector_t vec_tmp;

    const mem_stride_t window_h_stride = K_h_stride * C_in;
    const mem_stride_t k_cout_str = K_h * VPU_INT8_EPV;

    VSETC(&vpu, MODE_S8);

    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K;

        //Initialize accumulators
        VLDD(&vpu, BSS->bias_hi);
        VLDR(&vpu, BSS->bias_lo);

        // These rows are between top and bottom padding
        for(int pr = K_h; pr; pr--){

            VLDC(&vpu, patch_X);
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
    }
}








