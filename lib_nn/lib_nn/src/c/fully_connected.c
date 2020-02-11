

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>




#define BSS_INNER_SIZE  (5)
void fc_boggle_BSS(
    data16_t* bss_out,
    int32_t* bias,
    int16_t* shift1,
    int16_t* scale,
    int16_t* shift2,
    data16_t* scratch,
    const unsigned C_out)
{
    const unsigned ceil_C_out = (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2);

    data16_t* buff = NULL;

    if(((void*)bias) == ((void*)bss_out)){
        //bss_out is being updated in-place. We will need to use a scratch buffer

        if(scratch != NULL){
            //scratch buffer was provided by user
            buff = scratch;
        } else {
            //need to malloc a scratch buffer.
            buff = (data16_t*) malloc(C_out * BSS_INNER_SIZE *  sizeof(data16_t));

            if(buff == NULL){
                printf("Failed to allocate scratch buffer.");
                __builtin_trap();
            }
        }

    } else {
        //bss_out is not being updated in-place, just copy from the inputs to
        //  bss_out.
    }


    if(buff != NULL){
        memcpy(&buff[0], bias, C_out * sizeof(int32_t));
        memcpy(&buff[2*C_out], shift1, C_out*sizeof(data16_t));
        memcpy(&buff[3*C_out], scale, C_out*sizeof(data16_t));
        memcpy(&buff[4*C_out], shift2, C_out*sizeof(data16_t));

        bias = (int32_t*) &buff[0];
        shift1 = (int16_t*) &buff[2*C_out];
        scale = (int16_t*) &buff[3*C_out];
        shift2 = (int16_t*) &buff[4*C_out];
    }

    const unsigned C_out_groups = ceil_C_out >> VPU_INT8_ACC_PERIOD_LOG2;

    for(int cog = 0; cog < C_out_groups; cog++){

        const unsigned cog_offset = VPU_INT8_ACC_PERIOD * BSS_INNER_SIZE * cog;

        for(int coff = 0; coff < VPU_INT8_ACC_PERIOD; coff++){

            const unsigned cout = cog * VPU_INT8_ACC_PERIOD + coff;

            int32_t b      = bias[cout];
            data16_t shr1  = shift1[cout];
            data16_t scl   = scale[cout];
            data16_t shr2  = shift2[cout];

            data16_t b_lo = b & 0xFFFF;
            data16_t b_hi = (b & 0xFFFF0000) >> 16;

            bss_out[cog_offset + 0 * VPU_INT8_ACC_PERIOD + coff] = b_hi;
            bss_out[cog_offset + 1 * VPU_INT8_ACC_PERIOD + coff] = b_lo;
            bss_out[cog_offset + 2 * VPU_INT8_ACC_PERIOD + coff] = shr1;
            bss_out[cog_offset + 3 * VPU_INT8_ACC_PERIOD + coff] = scl;
            bss_out[cog_offset + 4 * VPU_INT8_ACC_PERIOD + coff] = shr2;
            
        }
    }


    if(buff != NULL && scratch == NULL){
        free(buff);
    }
}








void fc_deepin_shallowout_16_c(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales)
{
    assert(C_in % VPU_INT8_EPV == 0);
    assert(C_out <= 16);

    const int row_vlmaccrs = C_in / VPU_INT8_EPV;

    //Compute outputs one at a time
    for(unsigned k = 0; k < C_out; k++){

        //Compute pre-activation value
        int32_t acc32 = B[k];
        // printf("@@\t%ld\n", acc32);

        const int8_t* W_row = &W[k * C_in];

        for(unsigned v = 0; v < row_vlmaccrs; v++){

            int32_t vacc = 0;

            const int8_t* W_v = &W_row[v*VPU_INT8_EPV];
            const int8_t* X_v = &X[v*VPU_INT8_EPV];

            for(unsigned i = 0; i < VPU_INT8_EPV; i++)
                vacc += W_v[i] * X_v[i];

            int64_t acc64 = acc32 + vacc;

            if(acc64 > VPU_INT32_MAX)
                acc64 = VPU_INT32_MAX;
            if(acc64 < VPU_INT32_MIN)
                acc64 = VPU_INT32_MIN;

            acc32 = acc64;
        }

        //Compute shifts
        int16_t res = vlsat_single_s16(acc32, shifts[k]);

        //Compute scales
        res = vlmul_single_s16(res, scales[k]);

        Y[k] = res;
    }
}






void fully_connected_16_c(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const nn_fully_connected_plan_t* plan)
{
    const unsigned ACCS = VPU_INT8_ACC_PERIOD;

    const unsigned C_in = plan->c_in;
    const unsigned C_out = plan->c_out;

    for(unsigned cout = 0; cout < C_out; cout++){

        const unsigned cog = cout >> VPU_INT8_ACC_PERIOD_LOG2;
        const unsigned coff = cout & (ACCS - 1);

        const data16_t* BSS_cog = &BSS[5*ACCS * cog];
        const int8_t* W_row = &W[cout * C_in];
        const int32_t bias_hi = BSS_cog[coff + 0*ACCS];
        const int32_t bias_lo = BSS_cog[coff + 1*ACCS];
        const int16_t shift1  = BSS_cog[coff + 2*ACCS];
        const int16_t scale   = BSS_cog[coff + 3*ACCS];
        const int16_t shift2  = BSS_cog[coff + 4*ACCS];

        int64_t acc64 = (bias_hi << 16) | bias_lo;

        //For VERY deep inputs, it is possible that this doesn't match the assembly.
        for(unsigned cin = 0; cin < C_in; cin++){
            int32_t x = X[cin];
            int32_t w = W_row[cin];
            int32_t p = x * w;
            acc64 += p;
        }

        // printf("acc64 = %ld\n", acc64);

        acc64 =   (acc64 >= VPU_INT32_MAX)? VPU_INT32_MAX
                : (acc64 <= VPU_INT32_MIN)? VPU_INT32_MIN
                : acc64;

        int32_t res = vlsat_single_s16((int32_t)acc64, shift1);
        res = res * scale;
        res = vlsat_single_s16(res, shift2);

        Y[cout] = (int16_t) res;
    }
}







void fc_deepin_shallowout_8_c(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales)
{
    assert(C_in % VPU_INT8_EPV == 0);
    assert(C_out <= 16);

    const int row_vlmaccrs = C_in / VPU_INT8_EPV;

    //Compute outputs one at a time
    for(unsigned k = 0; k < C_out; k++){

        //Compute pre-activation value
        int32_t acc32 = B[k];
        // printf("@@\t%ld\n", acc32);

        const int8_t* W_row = &W[k * C_in];

        for(unsigned v = 0; v < row_vlmaccrs; v++){

            int32_t vacc = 0;

            const int8_t* W_v = &W_row[v*VPU_INT8_EPV];
            const int8_t* X_v = &X[v*VPU_INT8_EPV];

            for(unsigned i = 0; i < VPU_INT8_EPV; i++)
                vacc += W_v[i] * X_v[i];

            int64_t acc64 = acc32 + vacc;

            if(acc64 > VPU_INT32_MAX)
                acc64 = VPU_INT32_MAX;
            if(acc64 < VPU_INT32_MIN)
                acc64 = VPU_INT32_MIN;

            acc32 = acc64;
        }

        //Compute shifts
        int16_t res = vlsat_single_s16(acc32, shifts[k]);

        //Compute scales
        res = vlmul_single_s16(res, scales[k]);

        int8_t res8 = vdepth8_single_s16(res);

        Y[k] = res8;
    }
}






void fully_connected_init(
    nn_fully_connected_plan_t* plan,
    const unsigned C_in,
    const unsigned C_out)
{
    plan->c_in           = C_in;
    plan->c_out          = C_out;
    plan->cig_end_stride = (VPU_INT8_ACC_PERIOD-1) * C_in + 32;
    plan->tail_strat     = FC16_DEFAULT;
}