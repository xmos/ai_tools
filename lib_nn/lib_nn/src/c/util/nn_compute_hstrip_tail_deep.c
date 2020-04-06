

#include "nn_operator.h"
#include "../../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"
#include "../vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define ADDR(V, INDEX)      &V[((int)(INDEX))]


#define DO_VLMACCRS(K_INCR)                                                 \
    do {                                                                    \
        switch(C_out_mod2){                                                 \
            case 0:                                                         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
            case 4:                                                         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
            case 8:                                                         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); K_tmp = ADDR(K_tmp, k_cout_stride);         \
                VLMACCR(K_tmp); patch_K = ADDR(patch_K, K_INCR);            \
                break;                                                      \
            default:                                                        \
                assert(0);                                                  \
        }                                                                   \
    } while(0)

void nn_compute_hstrip_tail_deep_c(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
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
    int8_t  vec_tmp1[2*XS3_VPU_VREG_WIDTH_BYTES];
    int8_t* vec_tmp2 = &vec_tmp1[XS3_VPU_VREG_WIDTH_BYTES];

    VSETC(MODE_S8);
    VCLRDR();
    VSTR(vec_tmp1);

    
    const mem_stride_t win_h_stride = K_h_stride * C_in;
    const unsigned C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_in_tail = C_in % VPU_INT8_EPV;

    const unsigned C_out_mod1 = 2*(16-C_out_tail);
    const unsigned C_out_mod2 = (C_out_mod1>>1)-4;
    const unsigned write_mask = (1<<C_out_tail)-1;

    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K;

        VLDD(BSS->bias_hi);
        VLDR(BSS->bias_lo);

        for(int pr = K_h; pr; pr--){
            for(int col = K_w; col; col--){
                for(int cig = C_in_groups; cig; cig--){

                    VLDC(patch_X);
                    patch_X = ADDR(patch_X, VPU_INT8_EPV);

                    VSTR(vec_tmp2);
                    VLDR(&vec_tmp2[((int) -C_out_mod1 )]);
                    VSTD(vec_tmp2);
                    VLDD(&vec_tmp2[((int) -C_out_mod1 )]);

                    const nn_image_t* K_tmp = patch_K;
                    DO_VLMACCRS(VPU_INT8_EPV);
                }

                if(C_in_tail){
                    VLDC(patch_X);

                    VSTR(vec_tmp2);
                    VLDR(&vec_tmp2[((int) -C_out_mod1 )]);
                    VSTD(vec_tmp2);
                    VLDD(&vec_tmp2[((int) -C_out_mod1 )]);

                    VSTC(vec_tmp2);
                    VLDC(&vec_tmp2[((int) (C_in_tail-32) )]);

                    const nn_image_t* K_tmp = patch_K + (C_in_tail - 32);
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
        VSETC(MODE_S16);


        //Saturate to 16-bit values
        VLSAT(BSS->shift1);

        //Load scales into vC
        VLDC(BSS->scale);
        VSTR(vec_tmp2);
        VCLRDR();
        VLMACC(vec_tmp2);

        //Set mode back to 8-bit
        VSETC(MODE_S8);

        //Saturate to 8-bit values
        VLSAT(BSS->shift2);

        //Store result in Y
        VSTRPV(Y, write_mask);
        
        X = ADDR(X, win_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}








