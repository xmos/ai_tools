

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

void nn_compute_hstrip_deep_c(
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
        const unsigned out_cols)
{
    const mem_stride_t window_h_stride = K_h_stride * C_in;

    //First half is for zeroing out tail elements. Second half is actually just
    //  tmp stuff
    uint8_t tmp_vec[2*XS3_VPU_VREG_WIDTH_BYTES] = { 0 };
    uint8_t* mask_vec = ADDR(tmp_vec, XS3_VPU_VREG_WIDTH_BYTES);

    int8_t zero_tail[XS3_VPU_VREG_WIDTH_BYTES] = { 0 };

    //Number of C_in_groups
    const unsigned C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_in_tail = C_in % VPU_INT8_EPV;

    VSETC(MODE_S8);

    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K;

        //Initialize accumulators
        VLDD(&BSS->bias_hi);
        VLDR(&BSS->bias_lo);

        // These rows are between top and bottom padding
        for(int pr = K_h; pr; pr--){

            for(int col = K_w; col; col--){

                for(int cig = C_in_groups; cig; cig--){

                    VLDC(patch_X);

                    const nn_image_t* K_tmp = patch_K;

                    for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                        VLMACCR(K_tmp);
                        K_tmp = ADDR(K_tmp, k_cout_stride);
                    }

                    patch_X = ADDR(patch_X, VPU_INT8_EPV);
                    patch_K = ADDR(patch_K, VPU_INT8_EPV);
                }


                if(C_in_tail){
                    //This sequence should load vC with the masked out X values
                    //  at the *END* of the vector. Means K needs to have the 
                    //  corresponding elements at the end, too.
                    const mem_stride_t tail_offset = C_in_tail - VPU_INT8_EPV;
                    VLDC(patch_X);
                    VSTC(mask_vec);
                    VLDC(mask_vec + tail_offset);
                    
                    const nn_image_t* K_tmp = ADDR(patch_K, tail_offset);

                    for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                        VLMACCR(K_tmp);
                        K_tmp = ADDR(K_tmp, k_cout_stride);
                    }

                    patch_X = ADDR(patch_X, C_in_tail);
                    patch_K = ADDR(patch_K, C_in_tail);
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

        VSTR(mask_vec);
        VCLRDR();
        VLMACC(mask_vec);

        //Set mode back to 8-bit
        VSETC(MODE_S8);

        //Saturate to 8-bit values
        VLSAT(BSS->shift2);

        //Store result in Y
        const unsigned mask16 = 0xFFFF;
        VSTRPV(Y, mask16);
        
        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}








