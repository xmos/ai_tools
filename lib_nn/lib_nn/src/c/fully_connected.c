

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"
#include "vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>







WEAK_FUNC
void fc_deepin_shallowout_16(
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






WEAK_FUNC
void fc_deepin_shallowout_8(
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







WEAK_FUNC
void fully_connected_16(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const nn_bso_block_t* BSO,
    const nn_fully_connected_plan_t* plan,
    const nn_fully_connected_job_t* job)
{
    xs3_vpu vpu;
    const unsigned ACCS = VPU_INT8_ACC_PERIOD;

    const unsigned C_in = plan->channels.X;

    Y = ADDR(Y, job->stride.start.Y);
    W = ADDR(W, job->stride.start.BSO);
    BSO = ADDR(BSO, (job->stride.start.BSO / sizeof(nn_bso_block_t)));

    const unsigned C_out = job->output.channels;

    for(unsigned cout = 0; cout < C_out; cout++){

        const unsigned cog = cout >> VPU_INT8_ACC_PERIOD_LOG2;
        const unsigned coff = cout & (ACCS - 1);

        const nn_bso_block_t* BSO_cog = &BSO[cog];

        const int8_t* W_row = &W[cout * C_in];
        const int32_t bias_hi      = BSO_cog->bias_hi[coff];
        const int32_t bias_lo      = BSO_cog->bias_lo[coff];
        const int16_t shift1       = BSO_cog->shift1[coff];
        const int16_t scale        = BSO_cog->scale[coff];
        const int16_t offset_scale = BSO_cog->offset_scale[coff];
        const int16_t offset       = BSO_cog->offset[coff];
        const int16_t shift2       = BSO_cog->shift2[coff];

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
        res = res + ((int32_t)offset_scale) * offset;
        res = vlsat_single_s16(res, shift2);

        Y[cout] = (int16_t) res;
    }
}





void fully_connected_init(
    nn_fully_connected_plan_t* plan,
    nn_fully_connected_job_t* jobs,
    const channel_count_t C_in,
    const channel_count_t C_out,
    const nn_fully_connected_job_params_t* job_params,
    const unsigned job_count)
{
    // Either a (non-NULL) job params pointer must be provided, or 
    // job_count must be 1.
    assert( job_count == 1 || job_params != NULL);

    plan->channels.X     = C_in;

    // Params for processing the entire output 
    const nn_fully_connected_job_params_t full_job = { 0, C_out };

    for(int i = 0; i < job_count; i++){

        // Job params to pull info from. If NULL was provided to the job_params argument,
        // we'll use the full job.
        const nn_fully_connected_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_fully_connected_job_t* job = &jobs[i];

        //Because of the BSO tensor format, start output channel must be a multiple of 16
        assert(params->start_channel % VPU_INT8_ACC_PERIOD == 0);

        job->output.channels = params->out_channels;

        job->stride.start.Y = sizeof(int16_t) * params->start_channel;
        job->stride.start.W = C_in * params->start_channel;
        job->stride.start.BSO = sizeof(nn_bso_block_t) * (params->start_channel >> VPU_INT8_ACC_PERIOD_LOG2);
    }

}