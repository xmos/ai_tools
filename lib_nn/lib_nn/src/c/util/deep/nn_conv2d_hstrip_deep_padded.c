

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


WEAK_FUNC
void nn_conv2d_hstrip_deep_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_w,
        const unsigned K_h_stride,
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
    xs3_vpu vpu;

    const mem_stride_t window_h_stride = K_h_stride * C_in;

    //First half is for zeroing out tail elements. Second half is actually just
    //  tmp stuff
    uint8_t tmp_vec[2*XS3_VPU_VREG_WIDTH_BYTES] = { 0 };
    uint8_t* mask_vec = ADDR(tmp_vec, XS3_VPU_VREG_WIDTH_BYTES);

    int8_t zero_tail[XS3_VPU_VREG_WIDTH_BYTES] = { 0 };

    //Number of C_in_groups
    const unsigned C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_in_tail = C_in % VPU_INT8_EPV;

    //Number of rows to actually be computed in a patch
    const unsigned patch_rows = K_h - pad_t - pad_b;

    VSETC(&vpu, MODE_S8);

    //Set the masked tail zero vector
    VLDR(&vpu, zero_point_vec);
    VSTRPV(&vpu, zero_tail, (1 << C_in_tail)-1);

    //Load Biases for current C_out group
    VLDD(&vpu, BSO->bias_hi);
    VLDR(&vpu, BSO->bias_lo);

    //Adjust for bias at top
    if(pad_t){

        for(int row = pad_t; row; row--){
            for(int col = K_w; col; col--){
                VLDC(&vpu, zero_point_vec);
                for(int cig = C_in_groups; cig; cig--){

                    const nn_image_t* K_tmp = K;
                    for(int c_out = VPU_INT8_ACC_PERIOD; c_out; c_out--){
                        VLMACCR(&vpu, K_tmp);
                        K_tmp = ADDR(K_tmp, k_cout_stride);
                    }

                    X = ADDR(X, VPU_INT8_EPV);
                    K = ADDR(K, VPU_INT8_EPV);
                }

                if(C_in_tail){
                    VLDC(&vpu, zero_tail);

                    const nn_image_t* K_tmp = K;
                    for(int c_out = VPU_INT8_ACC_PERIOD; c_out; c_out--){
                        VLMACCR(&vpu, K_tmp);
                        K_tmp = ADDR(K_tmp, k_cout_stride);
                    }

                    X = ADDR(X, C_in_tail);
                    K = ADDR(K, C_in_tail);
                }
            }

            X = ADDR(X, x_v_stride);
        }
    }

    if(pad_b){
        VLDC(&vpu, zero_point_vec);

        //Skip middle rows

        //Move K_tmp an additional patch_rows down to get to the part of the kernel
        //  in the bottom padding
        const nn_image_t* K_tmp = ADDR(K, patch_rows * K_w * (C_in_groups * VPU_INT8_EPV + C_in_tail));

        for(int row = pad_b; row; row--){
            for(int col = K_w; col; col--){
                VLDC(&vpu, zero_point_vec);
                for(int cig = C_in_groups; cig; cig--){

                    const nn_image_t* K_tmp2 = K_tmp;
                    for(int c_out = VPU_INT8_ACC_PERIOD; c_out; c_out--){
                        VLMACCR(&vpu, K_tmp2);
                        K_tmp2 = ADDR(K_tmp2, k_cout_stride);
                    }

                    K_tmp = ADDR(K_tmp, VPU_INT8_EPV);
                }

                if(C_in_tail){
                    VLDC(&vpu, zero_tail);

                    const nn_image_t* K_tmp2 = K_tmp;
                    for(int c_out = VPU_INT8_ACC_PERIOD; c_out; c_out--){
                        VLMACCR(&vpu, K_tmp2);
                        K_tmp2 = ADDR(K_tmp2, k_cout_stride);
                    }

                    K_tmp = ADDR(K_tmp, C_in_tail);
                }
            }
        }
    }

    //Finally, store the adjusted biases
    vpu_vector_t adj_bias_hi;
    vpu_vector_t adj_bias_lo;

    VSTD(&vpu, &adj_bias_hi.u16[0]);
    VSTR(&vpu, &adj_bias_lo.u16[0]);
    
    //Alright! Now we can do the actual patches.
    
    //At this point, 
    //  - X should be pointing at the top-left of the effective patch
    //  - K should be pointing at the first cell below the top padding
    //  - BSO_p should be pointing at the shift1's

    int pad_l = pad_l_initial;
    int pad_r = pad_r_initial;

    int center_cols = K_w;
    if(pad_l >= 0)  center_cols -= pad_l;
    if(pad_r >= 0)  center_cols -= pad_r;

    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K;

        const int cur_pad_l = (pad_l > 0)? pad_l : 0;
        const int cur_pad_r = (pad_r > 0)? pad_r : 0;

        //Initialize accumulators
        VLDD(&vpu, &adj_bias_hi.u16[0]);
        VLDR(&vpu, &adj_bias_lo.u16[0]);

        // These rows are between top and bottom padding
        for(int pr = patch_rows; pr; pr--){

            if(cur_pad_l){
                for(int col = cur_pad_l; col; col--){
                    VLDC(&vpu, zero_point_vec);

                    for(int cig = C_in_groups; cig; cig--){
                        const nn_image_t* K_tmp = patch_K;

                        for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                            VLMACCR(&vpu, K_tmp);
                            K_tmp = ADDR(K_tmp, k_cout_stride);
                        }

                        patch_X = ADDR(patch_X, VPU_INT8_EPV);
                        patch_K = ADDR(patch_K, VPU_INT8_EPV);
                    }

                    if(C_in_tail){
                        VLDC(&vpu, zero_tail);
                        const nn_image_t* K_tmp = patch_K;

                        for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                            VLMACCR(&vpu, K_tmp);
                            K_tmp = ADDR(K_tmp, k_cout_stride);
                        }

                        patch_X = ADDR(patch_X, C_in_tail);
                        patch_K = ADDR(patch_K, C_in_tail);
                    }
                }
            }

            if(center_cols){
                for(int col = center_cols; col; col--){

                    for(int cig = C_in_groups; cig; cig--){

                        VLDC(&vpu, patch_X);

                        const nn_image_t* K_tmp = patch_K;

                        for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                            VLMACCR(&vpu, K_tmp);
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
                        VLDC(&vpu, patch_X);
                        VSTC(&vpu, mask_vec);
                        VLDC(&vpu, mask_vec + tail_offset);
                        
                        const nn_image_t* K_tmp = ADDR(patch_K, tail_offset);

                        for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                            VLMACCR(&vpu, K_tmp);
                            K_tmp = ADDR(K_tmp, k_cout_stride);
                        }

                        patch_X = ADDR(patch_X, C_in_tail);
                        patch_K = ADDR(patch_K, C_in_tail);
                    }
                }
            }

            if(cur_pad_r){
                for(int col = cur_pad_r; col; col--){
                    VLDC(&vpu, zero_point_vec);
                    for(int cig = C_in_groups; cig; cig--){
                        const nn_image_t* K_tmp = patch_K;

                        for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                            VLMACCR(&vpu, K_tmp);
                            K_tmp = ADDR(K_tmp, k_cout_stride);
                        }

                        patch_X = ADDR(patch_X, VPU_INT8_EPV);
                        patch_K = ADDR(patch_K, VPU_INT8_EPV);
                    }

                    if(C_in_tail){
                        VLDC(&vpu, zero_tail);
                        const nn_image_t* K_tmp = patch_K;

                        for(int cout = VPU_INT8_ACC_PERIOD; cout; cout--){
                            VLMACCR(&vpu, K_tmp);
                            K_tmp = ADDR(K_tmp, k_cout_stride);
                        }

                        patch_X = ADDR(patch_X, C_in_tail);
                        patch_K = ADDR(patch_K, C_in_tail);
                    }
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
        VSTR(&vpu, mask_vec);
        VCLRDR(&vpu);
        VLMACC(&vpu, mask_vec);
        VLDC(&vpu, BSO->offset_scale);
        VLMACC(&vpu, BSO->offset);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        const unsigned mask16 = 0xFFFF;
        VSTRPV(&vpu, Y, mask16);

        //Now make adjustments to pad_l, pad_r and center_cols

        if(pad_l > 0){
            int tmp = (pad_l <= K_h_stride)? pad_l : K_h_stride;
            center_cols += tmp;
        }

        pad_l -= (int) K_h_stride;
        pad_r += (int) K_h_stride;

        if(pad_r > 0){
            int tmp = (pad_r <= K_h_stride)? pad_r : K_h_stride;
            center_cols -= tmp;
        }
        
        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}








