
#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>




#define ADDR(VR, STR, PSTR)   printf("!\t%s = 0x%08X\t\t(%s)\n%s", (#VR), (unsigned) (VR), (STR), ("" PSTR))
void conv2d_1x1_c(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const data16_t* BSS,
    const nn_conv2d_1x1_plan_t* plan)
{
    X = &X[plan->start_stride.X];
    Y = &Y[plan->start_stride.Y];
    K = &K[plan->start_stride.K];
    // ADDR(X, "start", "");
    // ADDR(Y, "start", "");
    // ADDR(K, "start", "");

    const int8_t* X_start = X;

    const unsigned C_out_groups = plan->C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    const unsigned C_out_tail = plan->C_out % VPU_INT8_ACC_PERIOD;
    const unsigned C_in_groups = plan->C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_in_tail = plan->C_in % VPU_INT8_EPV;
    
    for(int cog = 0; cog <= C_out_groups; cog++){

        const int32_t cig_stride = (cog < C_out_groups)? plan->cig_stride.body : plan->cig_stride.tail;

        K = &K[(int)(cig_stride - VPU_INT8_EPV)];

        const unsigned group_chans = (cog < C_out_groups)? VPU_INT8_ACC_PERIOD : C_out_tail;
        if(!group_chans) break;

        X = X_start;
        // ADDR(X, "cog start", "");
        // ADDR(Y, "cog start", "");
        // ADDR(K, "cog start", "\n");

        for(int pix = 0; pix < plan->pix_count; pix++){
            int64_t acc64[VPU_INT8_ACC_PERIOD];
            // ADDR(X, "pixel start", "");
            // ADDR(Y, "pixel start", "");
            // ADDR(K, "pixel start", "");

            //Biases
            for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++)
                acc64[k] = (BSS[VPU_INT8_ACC_PERIOD * 0 + k] << VPU_INT8_ACC_VR_BITS)
                         | (BSS[VPU_INT8_ACC_PERIOD * 1 + k] <<  0);

            for(unsigned cig = 0; cig <= C_in_groups; cig++){

                const unsigned chans_in  = (cig < C_in_groups)? VPU_INT8_EPV : plan->C_in % VPU_INT8_EPV;
                if(!chans_in) break;

                for(int k = group_chans-1; k >= 0; k--){

                    for(unsigned cin = 0; cin < chans_in; cin++){
                        acc64[k] += ((int32_t)X[cin]) * K[cin];
                        acc64[k] = sat_s32(acc64[k]);
                    }

                    if(k != 0)
                        K = &K[(int)-plan->C_in];

                    // ADDR(X, "cout end", "");
                    // ADDR(Y, "cout end", "");
                    // ADDR(K, "cout end", "");
                }
                if(cig == C_in_groups){
                    K = &K[(int)(C_in_tail - 32)];
                }

                X = &X[chans_in];
                K = &K[cig_stride];
                // ADDR(X, "cig end", "");
                // ADDR(Y, "cig end", "");
                // ADDR(K, "cig end", "");
            }

            for(unsigned k = 0; k < group_chans; k++){


                int16_t shift1  = BSS[VPU_INT8_ACC_PERIOD * 2 + k];
                int16_t scale   = BSS[VPU_INT8_ACC_PERIOD * 3 + k];
                int16_t shift2  = BSS[VPU_INT8_ACC_PERIOD * 4 + k];
                
                int32_t res = vlsat_single_s16((int32_t)acc64[k], shift1);
                res = res * scale;
                res = vlsat_single_s8(res, shift2);

                Y[k] = (int8_t) res;
            }

            K = &K[(int)-plan->C_in];
            Y = &Y[plan->C_out];
            // ADDR(X, "pixel end", "");
            // ADDR(Y, "pixel end", "");
            // ADDR(K, "pixel end", "\n");
        }

        Y = &Y[plan->cog_stride.Y];
        K = &K[plan->C_in];
        // ADDR(Y, "cog end", "");
        // ADDR(K, "cog end", "\n");
        BSS = &BSS[5*VPU_INT8_ACC_PERIOD];
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
