
#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



WEAK_FUNC
void conv2d_1x1(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_1x1_plan_t* plan)
{
    X = ADDR(X, plan->start_stride.X);
    Y = ADDR(Y, plan->start_stride.Y);
    K = ADDR(K, plan->start_stride.K);

    const int8_t* X_start = X;

    const unsigned C_out_groups = plan->C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    const unsigned C_out_tail = plan->C_out % VPU_INT8_ACC_PERIOD;
    const unsigned C_in_groups = plan->C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_in_tail = plan->C_in % VPU_INT8_EPV;
    
    for(int cog = 0; cog <= C_out_groups; cog++){

        const int32_t cig_stride = (cog < C_out_groups)? plan->cig_stride.body : plan->cig_stride.tail;

        K = ADDR(K, cig_stride - VPU_INT8_EPV);

        const unsigned group_chans = (cog < C_out_groups)? VPU_INT8_ACC_PERIOD : C_out_tail;
        if(!group_chans) break;

        X = ADDR(X_start, 0);

        for(int pix = 0; pix < plan->pix_count; pix++){
            int64_t acc64[VPU_INT8_ACC_PERIOD];

            //Biases
            for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++)
                acc64[k] = (BSO->bias_hi[k] << VPU_INT8_ACC_VR_BITS)
                         | (BSO->bias_lo[k] << 0);

            for(unsigned cig = 0; cig <= C_in_groups; cig++){

                const unsigned chans_in  = (cig < C_in_groups)? VPU_INT8_EPV : plan->C_in % VPU_INT8_EPV;
                if(!chans_in) break;

                for(int k = group_chans-1; k >= 0; k--){

                    for(unsigned cin = 0; cin < chans_in; cin++){
                        acc64[k] += ((int32_t)X[cin]) * K[cin];
                        acc64[k] = sat_s32(acc64[k]);
                    }

                    if(k != 0)
                        K = ADDR(K, -plan->C_in);

                }
                if(cig == C_in_groups){
                    K = ADDR(K, C_in_tail - 32);
                }

                X = ADDR(X,chans_in);
                K = ADDR(K, cig_stride);
            }

            for(unsigned k = 0; k < group_chans; k++){


                int16_t shift1  = BSO->shift1[k];
                int16_t scale   = BSO->scale[k];
                int16_t shift2  = BSO->shift2[k];
                int16_t offset_scale = BSO->offset_scale[k];
                int16_t offset       = BSO->offset[k];
                
                int32_t res = vlsat_single_s16((int32_t)acc64[k], shift1);
                res = res * scale;
                res = res + ((int32_t)offset_scale) * offset;
                
                res = vlsat_single_s8(res, shift2);

                Y[k] = (int8_t) res;
            }

            K = ADDR(K,-plan->C_in);
            Y = ADDR(Y, plan->C_out);
        }

        Y = ADDR(Y, plan->cog_stride.Y);
        K = ADDR(K, plan->C_in);
        BSO = ADDR(BSO, 1);
    }
}
#undef ADDR









void conv2d_1x1_init(
    nn_conv2d_1x1_plan_t* plan,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const unsigned start_row,
    const unsigned start_col,
    const unsigned out_pixels)
{
    assert(x->height == y->height);
    assert(x->width == y->width);

    plan->start_stride.X = IMG_ADDRESS_VECT(x, start_row, start_col, 0);
    plan->start_stride.Y = IMG_ADDRESS_VECT(y, start_row, start_col, 0);
    plan->start_stride.K = 0;

    plan->pix_count = out_pixels;
    plan->C_in = x->channels;
    plan->C_out = y->channels;

    const unsigned C_out_tail = plan->C_out % VPU_INT8_ACC_PERIOD;

    plan->cig_stride.body = (VPU_INT8_ACC_PERIOD-1) * x->channels + VPU_INT8_EPV;
    plan->cig_stride.tail = (C_out_tail - 1) * x->channels + VPU_INT8_EPV;

    plan->cog_stride.Y = -(plan->pix_count * y->channels) + VPU_INT8_ACC_PERIOD;
    plan->cog_stride.K = x->channels * y->channels;

}
