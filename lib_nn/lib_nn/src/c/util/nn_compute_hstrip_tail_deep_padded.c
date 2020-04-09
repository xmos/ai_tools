

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


#define DO_VLMACCRS(K_addr, K_INCR)                                         \
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
                VLMACCR(K_tmp); K_addr = ADDR(K_addr, K_INCR);              \
                break;                                                      \
            default:                                                        \
                assert(0);                                                  \
        }                                                                   \
    } while(0)


void nn_compute_hstrip_tail_deep_padded_c(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
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
    vpu_vector_t vec_adj_b_hi;
    vpu_vector_t vec_adj_b_lo;

    int8_t  vec_tmp1[2*XS3_VPU_VREG_WIDTH_BYTES];
    int8_t* vec_tmp2 = ADDR(vec_tmp1, XS3_VPU_VREG_WIDTH_BYTES);

    VSETC(MODE_S8);
    VCLRDR();
    VSTR(vec_tmp1);
    VLDD(BSS->bias_hi);
    VLDR(BSS->bias_lo);

    
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
                VLDC(zero_point_vec);

                for(int cig = C_in_groups; cig; cig--){

                    VSTR(vec_tmp2);
                    VLDR(ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(vec_tmp2);
                    VLDD(ADDR(vec_tmp2, -C_out_mod1));

                    const nn_image_t* K_tmp = ADDR(K, 0);
                    DO_VLMACCRS(K, VPU_INT8_EPV);
                    X = ADDR(X, VPU_INT8_EPV);
                }

                if(C_in_tail){

                    VSTR(vec_tmp2);
                    VLDR(ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(vec_tmp2);
                    VLDD(ADDR(vec_tmp2, -C_out_mod1));

                    VSTC(vec_tmp2);
                    VLDC(ADDR(vec_tmp2, (C_in_tail-32) ));

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
                VLDC(zero_point_vec);

                for(int cig = C_in_groups; cig; cig--){

                    VSTR(vec_tmp2);
                    VLDR(ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(vec_tmp2);
                    VLDD(ADDR(vec_tmp2, -C_out_mod1));

                    const nn_image_t* K_tmp = ADDR(K_bot, 0);
                    DO_VLMACCRS(K_bot, VPU_INT8_EPV);
                }

                if(C_in_tail){

                    VSTR(vec_tmp2);
                    VLDR(ADDR(vec_tmp2, -C_out_mod1));
                    VSTD(vec_tmp2);
                    VLDD(ADDR(vec_tmp2, -C_out_mod1));

                    VSTC(vec_tmp2);
                    VLDC(ADDR(vec_tmp2, (C_in_tail-32)));

                    const nn_image_t* K_tmp = ADDR(K_bot, (C_in_tail - 32));
                    DO_VLMACCRS(K_bot, C_in_tail);
                }
            }
        }
    }

    VSTD(vec_adj_b_hi.s16);
    VSTR(vec_adj_b_lo.s16);

    int pad_l = pad_l_initial;
    int pad_r = pad_r_initial;

    int center_cols = K_w;
    if(pad_l >= 0)  center_cols -= pad_l;
    if(pad_r >= 0)  center_cols -= pad_r;

    for(int out_col = 0; out_col < out_cols; out_col++){

        const nn_image_t* patch_X = ADDR(X, 0);
        const nn_image_t* patch_K = ADDR(K, 0);

        VLDD(vec_adj_b_hi.u16);
        VLDR(vec_adj_b_lo.u16);

        for(int pr = patch_rows; pr; pr--){

            const int cur_pad_l = (pad_l > 0)? pad_l : 0;
            if(cur_pad_l){
                for(int col = cur_pad_l; col; col--){
                    VLDC(zero_point_vec);

                    for(int cig = C_in_groups; cig; cig--){

                        VSTR(vec_tmp2);
                        VLDR(ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(vec_tmp2);
                        VLDD(ADDR(vec_tmp2, -C_out_mod1));

                        const nn_image_t* K_tmp = ADDR(patch_K, 0);
                        DO_VLMACCRS(patch_K, VPU_INT8_EPV);
                        patch_X = ADDR(patch_X, VPU_INT8_EPV);
                    }

                    if(C_in_tail){

                        VSTR(vec_tmp2);
                        VLDR(ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(vec_tmp2);
                        VLDD(ADDR(vec_tmp2, -C_out_mod1));

                        VSTC(vec_tmp2);
                        VLDC(ADDR(vec_tmp2, (C_in_tail-32)));

                        const nn_image_t* K_tmp = ADDR(patch_K, (C_in_tail - 32));
                        DO_VLMACCRS(patch_K, C_in_tail);
                        patch_X = ADDR(patch_X, C_in_tail);
                    }
                }
            }

            if(center_cols){
                for(int col = center_cols; col; col--){

                    for(int cig = C_in_groups; cig; cig--){

                        VLDC(patch_X);
                        patch_X = ADDR(patch_X, VPU_INT8_EPV);

                        VSTR(vec_tmp2);
                        VLDR(ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(vec_tmp2);
                        VLDD(ADDR(vec_tmp2, -C_out_mod1));

                        const nn_image_t* K_tmp = patch_K;
                        DO_VLMACCRS(patch_K, VPU_INT8_EPV);

                    }

                    if(C_in_tail){

                        VLDC(patch_X);

                        VSTR(vec_tmp2);
                        VLDR(ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(vec_tmp2);
                        VLDD(ADDR(vec_tmp2, -C_out_mod1));

                        VSTC(vec_tmp2);
                        VLDC(ADDR(vec_tmp2, (C_in_tail-32)));

                        const nn_image_t* K_tmp = ADDR(patch_K, (C_in_tail - 32));
                        DO_VLMACCRS(patch_K, C_in_tail);
                        patch_X = ADDR(patch_X, C_in_tail);
                    }
                }
            }

            const int cur_pad_r = (pad_r > 0)? pad_r : 0;

            if(cur_pad_r){
                for(int col = cur_pad_r; col; col--){
                    VLDC(zero_point_vec);

                    for(int cig = C_in_groups; cig; cig--){

                        VSTR(vec_tmp2);
                        VLDR(ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(vec_tmp2);
                        VLDD(ADDR(vec_tmp2, -C_out_mod1));

                        const nn_image_t* K_tmp = patch_K;
                        DO_VLMACCRS(patch_K, VPU_INT8_EPV);

                        patch_X = ADDR(patch_X, VPU_INT8_EPV);
                    }

                    if(C_in_tail){

                        VSTR(vec_tmp2);
                        VLDR(ADDR(vec_tmp2, -C_out_mod1));
                        VSTD(vec_tmp2);
                        VLDD(ADDR(vec_tmp2, -C_out_mod1));

                        VSTC(vec_tmp2);
                        VLDC(ADDR(vec_tmp2, (C_in_tail-32)));

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








