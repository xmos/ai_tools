
#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* XMOS reference bsign_8 implementation */
void bsign_8_ref(
    bnn_b32_t* y,
    const int8_t* x,
    const nn_bsign_8_plan_t * plan,
    const nn_bsign_8_job_t* job)
{
    y = ADDR(y, job->start/32);
    x = ADDR(x, job->start);
  
    uint32_t j = 0;
    uint32_t shift = 0;

    // Note, this matches Larq - where 0's are witten to the upper unused bytes of the tail word
    for(int i = 0; i < job->length; i++)
    {
        if(shift == 0)
            y[j] = 0;

        int32_t x_ = x[i] - plan->zero_point;

        if(x_ < 0)
            y[j] |= (1 << shift);

        shift++;

        if(shift == 32) 
        {
            ++j;
            shift = 0;
        }
    }
}

WEAK_FUNC
void bsign_8( 
    bnn_b32_t* y,
    const int8_t* x,
    const nn_bsign_8_plan_t * plan,
    const nn_bsign_8_job_t* job)
{
    /* Fall back to reference if no optimised version available */
    bsign_8_ref(y, x, plan, job);
}
