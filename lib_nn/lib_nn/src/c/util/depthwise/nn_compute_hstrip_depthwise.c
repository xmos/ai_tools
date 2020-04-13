

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
#define INDEX_CAST(X)   ((int32_t)(X))

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
void nn_compute_hstrip_depthwise(
    int8_t* Y,
    const int8_t* X_in, 
    const int8_t* K_in,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t y_col_stride,
    const unsigned out_cols,
    const unsigned chans_to_write)
{

    for(int out_col = 0; out_col < out_cols; out_col++){

        const int8_t* X = X_in;
        const int8_t* K = K_in;

        int32_t accs[VPU_INT8_VLMACC_ELMS];

        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] = ((int32_t)BSS->bias_hi[k]) << VPU_INT8_ACC_VR_BITS;
        
        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] |= BSS->bias_lo[k];

        // These rows are inside image (vertically)
        for(int i = K_h; i > 0; i--){

            for(int j = xk_col_stride * K_w; j > 0; j-= xk_col_stride){
                vlmacc8(accs, X, K);
                X = &X[INDEX_CAST(xk_col_stride)];
                K = &K[INDEX_CAST(xk_col_stride)];
            }

            X = &X[INDEX_CAST(x_row_stride)];
        }
        
        for(int k = 0; k < chans_to_write; k++){
            int16_t shift1  = BSS->shift1[k];
            int16_t scale   = BSS->scale[k];
            int16_t shift2  = BSS->shift2[k];
            accs[k] = vlsat_single_s16(accs[k], shift1);
            accs[k] = accs[k] * scale;
            accs[k] = vlsat_single_s8(accs[k], shift2);
            Y[k] = (int8_t) accs[k];
        }
        
        X_in = &X_in[INDEX_CAST(window_hstride)];
        Y = &Y[INDEX_CAST(y_col_stride)];

    }
}






