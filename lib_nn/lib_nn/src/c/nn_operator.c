

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



WEAK_FUNC
void argmax_16(
    const int16_t* A,
    int32_t* C,
    const int32_t N)
{
    if( N <= 0) return;

    *C = 0;

    for(int32_t i = 1; i < N; i++){
        if( A[i] > A[*C] ){
            *C = i;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

#if CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 

WEAK_FUNC
void requantize_16_to_8(
    int8_t* y,
    const int16_t* x,
    const nn_requantize_16_to_8_job_t* job)
{
    y = ADDR(y, job->start);
    x = ADDR(x, job->start);

    for(int i = 0; i < job->length; i++){
        y[i] = (x[i] < -0x7F80)? NEG_SAT_VAL : vdepth8_single_s16(x[i]);
    }
}

#undef NEG_SAT_VAL



void requantize_16_to_8_init(
    nn_requantize_16_to_8_job_t* jobs,
    const uint32_t length,
    unsigned job_count)
{
    // All of the jobs (except final one) must deal with a multiple
    // of 4 inputs so that the subsequent job is guaranteed to be
    // word-aligned.

    // ceil(length/4)
    const unsigned adj_count = (length + 3) >> 2;

    const unsigned count_per_job_a = (adj_count / job_count);
    const unsigned count_per_job_b = count_per_job_a + 1;
    const unsigned final = length % 4;
    const unsigned leftover = adj_count - (count_per_job_a * job_count);

    int32_t pos = 0;

    for(int k = 0; k < job_count; k++){
        jobs[k].start = pos;

        if(k == leftover)
            jobs[k].length = final;
        else if(k < leftover)
            jobs[k].length = 4*count_per_job_b;
        else
            jobs[k].length = 4*count_per_job_a;

        pos = pos + VPU_INT16_EPV * jobs[k].length;
    }

    assert(pos == length);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

WEAK_FUNC
void lookup8(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned length)
{
    for(int i = 0; i < length; i++){
        Y[i] = lut[X[i]];
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

WEAK_FUNC
void vpu_memcpy(
    void* dst,
    void* src,
    unsigned size)
{
    for(int i = 0; i < size; i++){
        ((int8_t*)dst)[i] = ((int8_t*)src)[i];
    }
}
