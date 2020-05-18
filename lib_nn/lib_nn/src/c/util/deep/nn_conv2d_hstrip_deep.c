

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

#if CONFIG_SYMMETRIC_SATURATION_conv2d_deep
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 


const extern int16_t vec_0x007F[VPU_INT8_ACC_PERIOD];
const extern int8_t vec_0x80[VPU_INT8_ACC_PERIOD];


WEAK_FUNC
void nn_conv2d_hstrip_deep(
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
        const unsigned out_cols)
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

    VSETC(&vpu, MODE_S8);

    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K;
        
#if !CONFIG_SYMMETRIC_SATURATION_conv2d_deep
        VLDR(&vpu, vec_0x80);
        VSTRPV(&vpu, Y, 0xFFFF);
#endif

        //Initialize accumulators
        VLDD(&vpu, &BSO->bias_hi);
        VLDR(&vpu, &BSO->bias_lo);

        // These rows are between top and bottom padding
        for(int pr = K_h; pr; pr--){

            for(int col = K_w; col; col--){

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

#if CONFIG_SYMMETRIC_SATURATION_conv2d_deep

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        const unsigned mask16 = 0xFFFF;
        VSTRPV(&vpu, Y, mask16);
        
#else

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        VSTR(&vpu, mask_vec);
        VLADD(&vpu, vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, mask_vec, -8);
        VDEPTH8(&vpu);

        //Store result in Y
        mask = mask & 0xFFFF;
        VSTRPV(&vpu, Y, mask);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

#endif

        
        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}









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

#if !CONFIG_SYMMETRIC_SATURATION_conv2d_deep
        VLDR(&vpu, vec_0x80);
        VSTRPV(&vpu, Y, 0xFFFF);
#endif

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


#if CONFIG_SYMMETRIC_SATURATION_conv2d_deep

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        const unsigned mask16 = 0xFFFF;
        VSTRPV(&vpu, Y, mask16);
        
#else

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        VSTR(&vpu, mask_vec);
        VLADD(&vpu, vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, mask_vec, -8);
        VDEPTH8(&vpu);
        
        //Store result in Y
        mask = mask & 0xFFFF;
        VSTRPV(&vpu, Y, mask);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

#endif


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



#define DO_VLMACCRS2(K_INCR)                                                 \
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

#if !CONFIG_SYMMETRIC_SATURATION_conv2d_deep
        VLDR(&vpu, vec_0x80);
        VSTRPV(&vpu, Y, write_mask);
#endif

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
                    DO_VLMACCRS2(VPU_INT8_EPV);
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
                    DO_VLMACCRS2(C_in_tail);

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

        
#if CONFIG_SYMMETRIC_SATURATION_conv2d_deep

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        VSTRPV(&vpu, Y, write_mask);
        
#else

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        VSTR(&vpu, vec_tmp2);
        VLADD(&vpu, vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, vec_tmp2, -8);
        VDEPTH8(&vpu);
        
        //Store result in Y
        mask = mask & write_mask;
        VSTRPV(&vpu, Y, mask);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

#endif
        
        X = ADDR(X, win_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}




#define DO_VLMACCRS(K_addr, K_INCR)                                         \
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
                VLMACCR(&vpu, K_tmp); K_addr = ADDR(K_addr, K_INCR);              \
                break;                                                      \
            default:                                                        \
                assert(0);                                                  \
        }                                                                   \
    } while(0)

WEAK_FUNC
void nn_conv2d_hstrip_tail_deep_padded(
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
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail)
{
    xs3_vpu vpu;

    vpu_vector_t vec_adj_b_hi;
    vpu_vector_t vec_adj_b_lo;

    int8_t  vec_tmp1[2*XS3_VPU_VREG_WIDTH_BYTES];
    int8_t* vec_tmp2 = ADDR(vec_tmp1, XS3_VPU_VREG_WIDTH_BYTES);

    VSETC(&vpu, MODE_S8);
    VCLRDR(&vpu);
    VSTR(&vpu, vec_tmp1);
    VLDD(&vpu, BSO->bias_hi);
    VLDR(&vpu, BSO->bias_lo);

    
    const unsigned patch_rows = K_h - pad_t - pad_b;
    const mem_stride_t win_h_stride = K_h_stride * C_in;
    const unsigned C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_in_tail = C_in % VPU_INT8_EPV;

    const unsigned C_out_mod1 = 2*(16-C_out_tail);
    const unsigned C_out_mod2 = (C_out_mod1>>1)-4;
    const unsigned write_mask = (1<<C_out_tail)-1;


    if(pad_t){

        for(int row = pad_t; row; row--){

            for(int col = K_w; col; col--){
                VLDC(&vpu, zero_point_vec);

                for(int cig = C_in_groups; cig; cig--){

                    VSTR(&vpu, vec_tmp2);
                    VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(&vpu, vec_tmp2);
                    VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                    const nn_image_t* K_tmp = ADDR(K, 0);
                    DO_VLMACCRS(K, VPU_INT8_EPV);
                    X = ADDR(X, VPU_INT8_EPV);
                }

                if(C_in_tail){

                    VSTR(&vpu, vec_tmp2);
                    VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(&vpu, vec_tmp2);
                    VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                    VSTC(&vpu, vec_tmp2);
                    VLDC(&vpu, ADDR(vec_tmp2, (C_in_tail-32) ));

                    const nn_image_t* K_tmp = ADDR(K, (C_in_tail - 32));
                    DO_VLMACCRS(K, C_in_tail);
                    X = ADDR(X, C_in_tail);
                }
            }

            X = ADDR(X, x_v_stride);
        }
    }

    if(pad_b){
        const nn_tensor_t* K_bot = ADDR(K, (patch_rows * K_w * C_in));
        for(int row = pad_b; row; row--){

            for(int col = K_w; col; col--){
                VLDC(&vpu, zero_point_vec);

                for(int cig = C_in_groups; cig; cig--){

                    VSTR(&vpu, vec_tmp2);
                    VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(&vpu, vec_tmp2);
                    VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                    const nn_image_t* K_tmp = ADDR(K_bot, 0);
                    DO_VLMACCRS(K_bot, VPU_INT8_EPV);
                }

                if(C_in_tail){

                    VSTR(&vpu, vec_tmp2);
                    VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(&vpu, vec_tmp2);
                    VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                    VSTC(&vpu, vec_tmp2);
                    VLDC(&vpu, ADDR(vec_tmp2, (C_in_tail-32)));

                    const nn_image_t* K_tmp = ADDR(K_bot, (C_in_tail - 32));
                    DO_VLMACCRS(K_bot, C_in_tail);
                }
            }
        }
    }

    VSTD(&vpu, vec_adj_b_hi.s16);
    VSTR(&vpu, vec_adj_b_lo.s16);

    int pad_l = pad_l_initial;
    int pad_r = pad_r_initial;

    int center_cols = K_w;
    if(pad_l >= 0)  center_cols -= pad_l;
    if(pad_r >= 0)  center_cols -= pad_r;

    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = ADDR(X, 0);
        const nn_image_t* patch_K = ADDR(K, 0);

#if !CONFIG_SYMMETRIC_SATURATION_conv2d_deep
        VLDR(&vpu, vec_0x80);
        VSTRPV(&vpu, Y, write_mask);
#endif

        VLDD(&vpu, vec_adj_b_hi.u16);
        VLDR(&vpu, vec_adj_b_lo.u16);

        for(int pr = patch_rows; pr; pr--){

            const int cur_pad_l = (pad_l > 0)? pad_l : 0;
            if(cur_pad_l){
                for(int col = cur_pad_l; col; col--){
                    VLDC(&vpu, zero_point_vec);

                    for(int cig = C_in_groups; cig; cig--){

                        VSTR(&vpu, vec_tmp2);
                        VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(&vpu, vec_tmp2);
                        VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                        const nn_image_t* K_tmp = ADDR(patch_K, 0);
                        DO_VLMACCRS(patch_K, VPU_INT8_EPV);
                        patch_X = ADDR(patch_X, VPU_INT8_EPV);
                    }

                    if(C_in_tail){

                        VSTR(&vpu, vec_tmp2);
                        VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(&vpu, vec_tmp2);
                        VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                        VSTC(&vpu, vec_tmp2);
                        VLDC(&vpu, ADDR(vec_tmp2, (C_in_tail-32)));

                        const nn_image_t* K_tmp = ADDR(patch_K, (C_in_tail - 32));
                        DO_VLMACCRS(patch_K, C_in_tail);
                        patch_X = ADDR(patch_X, C_in_tail);
                    }
                }
            }

            if(center_cols){
                for(int col = center_cols; col; col--){

                    for(int cig = C_in_groups; cig; cig--){

                        VLDC(&vpu, patch_X);
                        patch_X = ADDR(patch_X, VPU_INT8_EPV);

                        VSTR(&vpu, vec_tmp2);
                        VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(&vpu, vec_tmp2);
                        VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                        const nn_image_t* K_tmp = patch_K;
                        DO_VLMACCRS(patch_K, VPU_INT8_EPV);

                    }

                    if(C_in_tail){

                        VLDC(&vpu, patch_X);

                        VSTR(&vpu, vec_tmp2);
                        VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(&vpu, vec_tmp2);
                        VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                        VSTC(&vpu, vec_tmp2);
                        VLDC(&vpu, ADDR(vec_tmp2, (C_in_tail-32)));

                        const nn_image_t* K_tmp = ADDR(patch_K, (C_in_tail - 32));
                        DO_VLMACCRS(patch_K, C_in_tail);
                        patch_X = ADDR(patch_X, C_in_tail);
                    }
                }
            }

            const int cur_pad_r = (pad_r > 0)? pad_r : 0;

            if(cur_pad_r){
                for(int col = cur_pad_r; col; col--){
                    VLDC(&vpu, zero_point_vec);

                    for(int cig = C_in_groups; cig; cig--){

                        VSTR(&vpu, vec_tmp2);
                        VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(&vpu, vec_tmp2);
                        VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                        const nn_image_t* K_tmp = patch_K;
                        DO_VLMACCRS(patch_K, VPU_INT8_EPV);

                        patch_X = ADDR(patch_X, VPU_INT8_EPV);
                    }

                    if(C_in_tail){

                        VSTR(&vpu, vec_tmp2);
                        VLDR(&vpu, ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(&vpu, vec_tmp2);
                        VLDD(&vpu, ADDR(vec_tmp2, -C_out_mod1));

                        VSTC(&vpu, vec_tmp2);
                        VLDC(&vpu, ADDR(vec_tmp2, (C_in_tail-32)));

                        const nn_image_t* K_tmp = ADDR(patch_K, (C_in_tail - 32));
                        DO_VLMACCRS(patch_K, C_in_tail);
                        patch_X = ADDR(patch_X, C_in_tail);
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
        VSTR(&vpu, vec_tmp2);
        VCLRDR(&vpu);
        VLMACC(&vpu, vec_tmp2);
        VLDC(&vpu, BSO->offset_scale);
        VLMACC(&vpu, BSO->offset);

        
#if CONFIG_SYMMETRIC_SATURATION_conv2d_deep

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        VSTRPV(&vpu, Y, write_mask);
        
#else

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        VSTR(&vpu, vec_tmp2);
        VLADD(&vpu, vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, vec_tmp2, -8);
        VDEPTH8(&vpu);
        
        //Store result in Y
        mask = mask & write_mask;
        VSTRPV(&vpu, Y, write_mask);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

#endif

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
        
        X = ADDR(X, win_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}


