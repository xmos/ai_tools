

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
    int32_t* Y,
    const int16_t* X,
    const int32_t N)
{
    if( N <= 0) return;

    *Y = 0;

    for(int32_t i = 1; i < N; i++){
        if( X[i] > X[*Y] ){
            *Y = i;
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
 
    for(int k = 0; k < job_count; k++){
        jobs[k].length = 0;
    }

    int32_t left = (length >> 2) << 2;

    while(left){
        for(int k = 0; k < job_count; k++){
            if(left >= 4){
                jobs[k].length += 4;
                left -= 4;
            } else {
                jobs[k].length += left;
                left -= left;
            }
        }
        if(left == 0) break;
    }
    jobs[job_count-1].length += (length % 4);

    jobs[0].start = 0;

    int32_t pos = jobs[0].length;

    for(int k = 1; k < job_count; k++){
        jobs[k].start = jobs[k-1].start + jobs[k-1].length;
        pos += jobs[k].length;
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
void bsign_8(
    uint32_t* y,
    const int8_t* x,
    const nn_bsign_8_job_t* job)
{
    y = ADDR(y, job->start/32);
    x = ADDR(x, job->start);
   
    uint32_t j = 0;
    uint32_t shift = 0;

    for(int i = 0; i < job->length; i++)
    {
        if(x[i] < 0)
            y[j] |= (1 << shift);
        
        shift++;

        if(shift == 32) 
        {
            shift = 0; 
            j++;
        }
    }
}

void bsign_8_init(
    nn_bsign_8_job_t* jobs,
    const uint32_t length,
    unsigned job_count)
{
    for(int k = 0; k < job_count; k++){
        jobs[k].length = 0;
    }

    int32_t left = (length >> 5) << 5;

    while(left){
        for(int k = 0; k < job_count; k++){
            if(left >= 32){
                jobs[k].length += 32;
                left -= 32;
            } else {
                jobs[k].length += left;
                left -= left;
            }
        }
        if(left == 0) break;
    }
    jobs[job_count-1].length += (length % 32);

    jobs[0].start = 0;

    int32_t pos = jobs[0].length;

    for(int k = 1; k < job_count; k++){
        jobs[k].start = jobs[k-1].start + jobs[k-1].length;
        pos += jobs[k].length;
   
        assert(jobs[k].length % 8 == 0);
    }

    assert(pos == length);
}
