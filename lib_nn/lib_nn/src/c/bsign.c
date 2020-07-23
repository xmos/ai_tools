#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#if defined(__XS3A__)
/* Bsign_8 optimised for XS3A */
void bsign_8_( 
    uint32_t* y,
    const int8_t* x,
    const int8_t* zero_point_vect,
    const nn_bsign_8_job_t* job);

void bsign_8(
    uint32_t* y,
    const int8_t* x,
    const nn_bsign_8_plan_t * plan,
    const nn_bsign_8_job_t* job)
{
    /* This implementation follows the convetion of existing code - we are passing aroind a plan of contents 
     * 1 byte zero_point. This actually costs us more in terms of memory usage than simply passing around the value.
     * Worse than this we then generate a zero point vector PER_JOB effecting memory and runtime 
     * Therefore, TODO, put the zero-point vector in the plan and generate in init() 
     */
    int8_t zero_point_vect[VPU_INT8_EPV];
    memset(zero_point_vect, plan->zero_point, sizeof(zero_point_vect));

    /* Note, at this point we have no more use for the plan..*/
    bsign_8_(y, x, (const int8_t*)&zero_point_vect, job);
}
#endif

void bsign_8_init(
    nn_bsign_8_plan_t* plan,
    nn_bsign_8_job_t* jobs,
    const uint32_t length,
    const int8_t zero_point,
    unsigned job_count)
{
    plan->zero_point = zero_point;

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
