

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



static void vlmacc8(
    int32_t* acc,
    const int8_t* X,
    const int8_t* W)
{
    for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++){
        // printf("!@ %d\t%d\t%d\n", k, X[k], W[k]);
        acc[k] += ((int32_t)X[k]) * W[k];
    }
}


WEAK_FUNC
void nn_conv2d_hstrip_depthwise_padded(
    int8_t* Y,
    const int8_t* X_in, 
    const int8_t* K_in,
    const nn_bso_block_t* BSO,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t pad_t,
    const int32_t pad_l_initial,
    const int32_t pad_b,
    const int32_t pad_r_initial,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t y_col_stride,
    const unsigned out_cols,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec)
{

    int pad_l = pad_l_initial * xk_col_stride;
    int pad_r = pad_r_initial * xk_col_stride;

    int center_cols = xk_col_stride * K_w;
    if(pad_l >= 0)  center_cols -= pad_l;
    if(pad_r >= 0)  center_cols -= pad_r;

    for(int out_col = 0; out_col < out_cols; out_col++){

        const int8_t* X = X_in;
        const int8_t* K = K_in;
        // ADDR(X, "out col start");
        // ADDR(Y, "out col start");
        // ADDR(K, "out col start");

        const int cur_pad_l = (pad_l > 0)? pad_l : 0;
        const int cur_pad_r = (pad_r > 0)? pad_r : 0;

        // printf("pads:  (%d, %d, %d, %d)\n", pad_t, pad_l, pad_l, pad_r);
        // printf("cur_pads:  (%d, %d, %d, %d)\n", pad_t, cur_pad_l, cur_pad_l, cur_pad_r);

        int32_t accs[VPU_INT8_VLMACC_ELMS];

        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] = ((int32_t)BSO->bias_hi[k]) << VPU_INT8_ACC_VR_BITS;
        
        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] |= BSO->bias_lo[k];
        
        //THIS LOOP IS IN PADDING (above image)
        for(int i = pad_t; i > 0; i--){
            // printf("PAD_T??\t%d\t%d\n", pad_t, i);
            for(int j = K_w; j > 0; j--){
                vlmacc8(accs, zero_point_vec, K);
                X = ADDR(X, xk_col_stride);
                K = ADDR(K, xk_col_stride);
            }
            X = ADDR(X, x_row_stride);
        }

        // These rows are inside image (vertically)
        for(int i = K_h - (pad_t + pad_b); i > 0; i--){

            //THIS LOOP IS IN PADDING (left of image)
            for(int j = cur_pad_l; j > 0; j -= xk_col_stride){
                // printf("PAD_L??\t%d\t%d\n", cur_pad_l, j);
                vlmacc8(accs, zero_point_vec, K);
                X = ADDR(X, xk_col_stride);
                K = ADDR(K, xk_col_stride);
            }

            for(int j = center_cols; j > 0; j-= xk_col_stride){
                vlmacc8(accs, X, K);
                X = ADDR(X, xk_col_stride);
                K = ADDR(K, xk_col_stride);
            }

            //THIS LOOP IS IN PADDING (right of image)
            for(int j = cur_pad_r; j > 0; j -= xk_col_stride){
                // printf("PAD_R??\t%d\t%d\n", cur_pad_r, j);
                vlmacc8(accs, zero_point_vec, K);
                X = ADDR(X, xk_col_stride);
                K = ADDR(K, xk_col_stride);
            }

            X = ADDR(X, x_row_stride);
        }
        
        //THIS LOOP IS IN PADDING (below image)
        for(int i = pad_b; i > 0; i--){
            // printf("PAD_B??\t%d\t%d\n", pad_b, i);
            for(int j = K_w; j > 0; j--){
                vlmacc8(accs, zero_point_vec, K);
                X = ADDR(X, xk_col_stride);
                K = ADDR(K, xk_col_stride);
            }
            X = ADDR(X, x_row_stride);
        }

        for(int k = 0; k < chans_to_write; k++){
            int16_t shift1  = BSO->shift1[k];
            int16_t scale   = BSO->scale[k];
            int16_t offset_scale = BSO->offset_scale[k];
            int16_t offset = BSO->offset[k];
            int16_t shift2  = BSO->shift2[k];
            accs[k] = vlsat_single_s16(accs[k], shift1);
            accs[k] = accs[k] * scale;
            accs[k] += ((int32_t)offset_scale) * offset;
            accs[k] = vlsat_single_s8(accs[k], shift2);
            Y[k] = (int8_t) accs[k];
        }

        if(pad_l > 0){
            int tmp = (pad_l <= window_hstride)? pad_l : window_hstride;
            center_cols += tmp;
        }

        pad_l -= (int) window_hstride;
        pad_r += (int) window_hstride;

        if(pad_r > 0){
            int tmp = (pad_r <= window_hstride)? pad_r : window_hstride;
            center_cols -= tmp;
        }
        
        X_in = ADDR(X_in,window_hstride);
        Y = ADDR(Y,y_col_stride);

    }
}







