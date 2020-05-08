

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



#define DO_VLMACCRS(K_addr, K_INCR)                                             \
    do {                                                                        \
        switch(tail_mod2){                                                      \
            case 0:                                                             \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
            case 4:                                                             \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
            case 8:                                                             \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp,-k_cout_str);          \
                VLMACCR(&vpu, K_tmp); K_addr = ADDR(K_addr, K_INCR);            \
                break;                                                          \
            default:                                                            \
                assert(0);                                                      \
        }                                                                       \
    } while(0)

WEAK_FUNC
void nn_conv2d_hstrip_tail_shallowin(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail)
{
    xs3_vpu vpu;

    vpu_vector_t vec_tmp1;

    const mem_stride_t window_h_stride = K_h_stride * C_in;
    const mem_stride_t k_cout_str = K_h * VPU_INT8_EPV;

    const unsigned tail_mod1  = 2*(16-C_out_tail);
    const unsigned tail_mod2  = 12-C_out_tail;
    const unsigned write_mask = (1<<C_out_tail)-1;

    VSETC(&vpu, MODE_S8);

    const nn_tensor_t* K_patch_start = ADDR(K, 0);

    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K_patch_start;

        //Initialize accumulators
        VLDD(&vpu, BSO->bias_hi);
        VLDR(&vpu, BSO->bias_lo);


        // These rows are between top and bottom padding
        for(int pr = K_h; pr; pr--){

            VLDC(&vpu, patch_X);
            patch_X = ADDR(patch_X, x_v_stride);
            
            VSTR(&vpu, vec_tmp1.s8);
            VLDR(&vpu, ADDR(vec_tmp1.s8, -tail_mod1));
            VSTD(&vpu, vec_tmp1.s8);
            VLDD(&vpu, ADDR(vec_tmp1.s8, -tail_mod1));

            const nn_tensor_t* K_tmp = ADDR(patch_K, 0);
            DO_VLMACCRS(patch_K, VPU_INT8_EPV);
        }
        
        //Done accumulating for the current patch

        //Set mode to 16-bit
        VSETC(&vpu, MODE_S16);

        //Saturate to 16-bit values
        VLSAT(&vpu, BSO->shift1);

        //Load scales into vC
        VLDC(&vpu, BSO->scale);
        VSTR(&vpu, vec_tmp1.s16);
        VCLRDR(&vpu);
        VLMACC(&vpu, vec_tmp1.s16);
        VLDC(&vpu, BSO->offset_scale);
        VLMACC(&vpu, BSO->offset);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        VSTRPV(&vpu, Y, write_mask);

        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}








