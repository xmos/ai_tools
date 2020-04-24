

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


#define DO_VLMACCRS(K_INCR)                                                 \
    do {                                                                    \
        switch(C_out_mod2){                                                 \
            case 0:                                                         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
            case 4:                                                         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
            case 8:                                                         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(&vpu, K_tmp); patch_K = ADDR(patch_K, K_INCR);            \
                break;                                                      \
            default:                                                        \
                assert(0);                                                  \
        }                                                                   \
    } while(0)

WEAK_FUNC
void nn_conv2d_hstrip_tail_deep(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_w,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail)
{
    xs3_vpu vpu;

    int8_t  vec_tmp1[2*XS3_VPU_VREG_WIDTH_BYTES];
    int8_t* vec_tmp2 = ADDR(vec_tmp1, XS3_VPU_VREG_WIDTH_BYTES);

    VSETC(&vpu, MODE_S8);
    VCLRDR(&vpu);
    VSTR(&vpu, vec_tmp1);

    
    const mem_stride_t win_h_stride = K_h_stride * C_in;
    const unsigned C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_in_tail = C_in % VPU_INT8_EPV;

    const unsigned C_out_mod1 = 2*(16-C_out_tail);
    const unsigned C_out_mod2 = (C_out_mod1>>1)-4;
    const unsigned write_mask = (1<<C_out_tail)-1;

    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K;

        VLDD(&vpu, BSO->bias_hi);
        VLDR(&vpu, BSO->bias_lo);

        for(int pr = K_h; pr; pr--){
            for(int col = K_w; col; col--){
                for(int cig = C_in_groups; cig; cig--){

                    VLDC(&vpu, patch_X);
                    patch_X = ADDR(patch_X, VPU_INT8_EPV);

                    VSTR(&vpu, vec_tmp2);
                    VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(&vpu, vec_tmp2);
                    VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                    const nn_image_t* K_tmp = ADDR(patch_K, 0);
                    DO_VLMACCRS(VPU_INT8_EPV);
                }

                if(C_in_tail){
                    VLDC(&vpu, patch_X);

                    VSTR(&vpu, vec_tmp2);
                    VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(&vpu, vec_tmp2);
                    VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                    VSTC(&vpu, vec_tmp2);
                    VLDC(&vpu, ADDR(vec_tmp2, C_in_tail-32));

                    const nn_image_t* K_tmp = ADDR(patch_K, (C_in_tail - 32));
                    DO_VLMACCRS(C_in_tail);

                    patch_X = ADDR(patch_X, C_in_tail);
                }
            }

            //patch_X currently pointing to pixel to right of patch in current row
            //patch_K should be pointing to the right place
            patch_X = ADDR(patch_X, x_v_stride);
        }
        
        //Done accumulating for the current patch

        //Set mode to 16-bit
        VSETC(&vpu, MODE_S16);


        //Saturate to 16-bit values
        VLSAT(&vpu, BSO->shift1);

        //Load scales into vC
        VLDC(&vpu, BSO->scale);
        VSTR(&vpu, vec_tmp2);
        VCLRDR(&vpu);
        VLMACC(&vpu, vec_tmp2);
        VLDC(&vpu, BSO->offset_scale);
        VLMACC(&vpu, BSO->offset);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        VSTRPV(&vpu, Y, write_mask);
        
        X = ADDR(X, win_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}








