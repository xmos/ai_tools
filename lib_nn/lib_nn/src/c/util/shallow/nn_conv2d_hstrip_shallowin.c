

#include "nn_operator.h"
#include "../../../nn_op_helper.h"
// #include "nn_op_structs.h"

#include "xs3_vpu.h"
#include "../../../asm/asm_constants.h"
#include "../../vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



#if CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 



WEAK_FUNC
void nn_conv2d_hstrip_shallowin(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
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

#if !CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin
        VLDR(&vpu, vpu_vects.vec_0x80);
        VSTRPV(&vpu, Y, 0xFFFF);
#endif

        //Initialize accumulators
        VLDD(&vpu, BSO->bias_hi);
        VLDR(&vpu, BSO->bias_lo);

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
        VLSAT(&vpu, BSO->shift1);

        //Load scales into vC
        VLDC(&vpu, BSO->scale);
        VSTR(&vpu, vec_tmp.s16);
        VCLRDR(&vpu);
        VLMACC(&vpu, vec_tmp.s16);
        VLDC(&vpu, BSO->offset_scale);
        VLMACC(&vpu, BSO->offset);

#if CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin

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

        VSTR(&vpu, vec_tmp.s16);
        VLADD(&vpu, vpu_vects.vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, vec_tmp.s16, -8);
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
void nn_conv2d_hstrip_shallowin_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
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
    const int32_t patch_rows = K_h - pad_t - pad_b;

    VSETC(&vpu, MODE_S8);

    //Load Biases for current C_out group
    VLDD(&vpu, BSO->bias_hi);
    VLDR(&vpu, BSO->bias_lo);

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

    int32_t pad_l = pad_l_initial * C_in;
    int32_t pad_r = pad_r_initial * C_in;

    int32_t pad_l_relu = (pad_l > 0)? pad_l : 0;
    int32_t pad_r_relu = (pad_r > 0)? pad_r : 0;

    uint32_t pad_mask = 32;

    pad_mask -= pad_l_relu;
    pad_mask -= pad_r_relu;

    if(pad_mask == 32)
        pad_mask = 0xFFFFFFFF;
    else
        pad_mask = ((1<<pad_mask)-1) << pad_l_relu;


    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K_patch_start;

#if !CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin
        VLDR(&vpu, vpu_vects.vec_0x80);
        VSTRPV(&vpu, Y, 0xFFFF);
#endif

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
        VLSAT(&vpu, BSO->shift1);

        //Load scales into vC
        VLDC(&vpu, BSO->scale);
        VSTR(&vpu, vec_tmp.s16);
        VCLRDR(&vpu);
        VLMACC(&vpu, vec_tmp.s16);
        VLDC(&vpu, BSO->offset_scale);
        VLMACC(&vpu, BSO->offset);

#if CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin

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

        VSTR(&vpu, vec_tmp.s16);
        VLADD(&vpu, vpu_vects.vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, vec_tmp.s16, -8);
        VDEPTH8(&vpu);
        
        //Store result in Y
        mask = mask & 0xFFFF;
        VSTRPV(&vpu, Y, mask);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

#endif

        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);

        //Now make adjustments to pad_l and pad_r
        pad_l -= window_h_stride;
        pad_r += window_h_stride;

        int32_t pad_l_relu = (pad_l > 0)? pad_l : 0;
        int32_t pad_r_relu = (pad_r > 0)? pad_r : 0;

        pad_mask = 32;
        pad_mask -= pad_l_relu;
        pad_mask -= pad_r_relu;

        if(pad_mask == 32)
            pad_mask = 0xFFFFFFFF;
        else
            pad_mask = ((1<<pad_mask)-1) << pad_l_relu;
    }
}





#define DO_VLMACCRS2(K_addr, K_INCR)                                            \
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

    const int32_t tail_mod1  = 2*(16-C_out_tail);
    const int32_t tail_mod2  = 12-C_out_tail;
    const int32_t write_mask = (1<<C_out_tail)-1;

    VSETC(&vpu, MODE_S8);

    const nn_tensor_t* K_patch_start = ADDR(K, 0);

    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K_patch_start;
        
#if !CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin
        VLDR(&vpu, vpu_vects.vec_0x80);
        VSTRPV(&vpu, Y, write_mask);
#endif

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
            DO_VLMACCRS2(patch_K, VPU_INT8_EPV);
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

        
#if CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        VSTRPV(&vpu, Y, write_mask);
        
#else

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        VSTR(&vpu, vec_tmp1.s16);
        VLADD(&vpu, vpu_vects.vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, vec_tmp1.s16, -8);
        VDEPTH8(&vpu);
        
        //Store result in Y
        mask = mask & write_mask;
        VSTRPV(&vpu, Y, mask);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

#endif

        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);
    }
}








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
void nn_conv2d_hstrip_tail_shallowin_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
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
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail)
{
    xs3_vpu vpu;

    vpu_vector_t vec_tmp1;
    vpu_vector_t vec_tmp2;
    vpu_vector_t adj_bias_hi;
    vpu_vector_t adj_bias_lo;

    const mem_stride_t window_h_stride = K_h_stride * C_in;
    const mem_stride_t k_cout_str = K_h * VPU_INT8_EPV;

    //Number of rows to actually be computed in a patch
    const int32_t patch_rows = K_h - pad_t - pad_b;

    const int32_t tail_mod1  = 2*(16-C_out_tail);
    const int32_t tail_mod2  = 12-C_out_tail;
    const uint32_t write_mask = (1<<C_out_tail)-1;

    VSETC(&vpu, MODE_S8);

    //Load Biases for current C_out group
    VLDD(&vpu, BSO->bias_hi);
    VLDR(&vpu, BSO->bias_lo);

    VLDC(&vpu, zero_point_vec);

    const nn_tensor_t* K_patch_start = ADDR(K, pad_t * VPU_INT8_EPV);
    X = ADDR(X, pad_t * x_v_stride);

    //Adjust for bias at top
    for(int row = pad_t; row; row--){

        VSTR(&vpu, vec_tmp1.s8);
        VLDR(&vpu, ADDR(vec_tmp1.s8, -tail_mod1));
        VSTD(&vpu, vec_tmp1.s8);
        VLDD(&vpu, ADDR(vec_tmp1.s8, -tail_mod1));

        const nn_tensor_t* K_tmp = ADDR(K, 0);
        DO_VLMACCRS(K, VPU_INT8_EPV);
    }

    K = ADDR(K, VPU_INT8_EPV * patch_rows);

    for(int row = pad_b; row; row--){

        VSTR(&vpu, vec_tmp1.s8);
        VLDR(&vpu, ADDR(vec_tmp1.s8, -tail_mod1));
        VSTD(&vpu, vec_tmp1.s8);
        VLDD(&vpu, ADDR(vec_tmp1.s8, -tail_mod1));

        const nn_tensor_t* K_tmp = ADDR(K, 0);
        DO_VLMACCRS(K, VPU_INT8_EPV);
    }

    //Store adjusted accumulators
    VSTD(&vpu, &adj_bias_hi.u16[0]);
    VSTR(&vpu, &adj_bias_lo.u16[0]);

    int32_t pad_l = pad_l_initial * C_in;
    int32_t pad_r = pad_r_initial * C_in;

    int32_t pad_l_relu = (pad_l > 0)? pad_l : 0;
    int32_t pad_r_relu = (pad_r > 0)? pad_r : 0;

    uint32_t pad_mask = 32;

    pad_mask -= pad_l_relu;
    pad_mask -= pad_r_relu;


    if(pad_mask == 32)
        pad_mask = 0xFFFFFFFF;
    else
        pad_mask = ((1<<pad_mask)-1) << pad_l_relu;

    //Loop over the output pixels
    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = X;
        const nn_image_t* patch_K = K_patch_start;

#if !CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin
        VLDR(&vpu, vpu_vects.vec_0x80);
        VSTRPV(&vpu, Y, write_mask);
#endif

        //Initialize accumulators
        VLDD(&vpu, &adj_bias_hi.u16[0]);
        VLDR(&vpu, &adj_bias_lo.u16[0]);

        VLDC(&vpu, zero_point_vec);
        VSTC(&vpu, vec_tmp2.s8);

        // These rows are between top and bottom padding
        for(int pr = patch_rows; pr; pr--){

            VSTR(&vpu, vec_tmp1.s16);
            VLDR(&vpu, patch_X);
            VSTRPV(&vpu, vec_tmp2.s8, pad_mask);
            VLDC(&vpu, vec_tmp2.s8);
            VLDR(&vpu, vec_tmp1.s16);
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

#if CONFIG_SYMMETRIC_SATURATION_conv2d_shallowin

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

        //Saturate to 8-bit values
        VLSAT(&vpu, BSO->shift2);

        //Store result in Y
        VSTRPV(&vpu, Y, write_mask);
        
#else

        VLSAT(&vpu, BSO->shift2);

        VSTR(&vpu, vec_tmp1.s16);
        VLADD(&vpu, vpu_vects.vec_0x007F);
        VDEPTH1(&vpu);
        uint32_t mask = ~vpu.vR.s32[0];

        VLASHR(&vpu, vec_tmp1.s16, -8);
        VDEPTH8(&vpu);
        
        //Store result in Y
        mask = mask & write_mask;
        VSTRPV(&vpu, Y, mask);

        //Set mode back to 8-bit
        VSETC(&vpu, MODE_S8);

#endif

        X = ADDR(X, window_h_stride);
        Y = ADDR(Y, y_h_stride);

        //Now make adjustments to pad_l and pad_r
        pad_l -= window_h_stride;
        pad_r += window_h_stride;

        int32_t pad_l_relu = (pad_l > 0)? pad_l : 0;
        int32_t pad_r_relu = (pad_r > 0)? pad_r : 0;

        pad_mask = 32;
        pad_mask -= pad_l_relu;
        pad_mask -= pad_r_relu;

        if(pad_mask == 32)
            pad_mask = 0xFFFFFFFF;
        else
            pad_mask = ((1<<pad_mask)-1) << pad_l_relu;
    }
}


