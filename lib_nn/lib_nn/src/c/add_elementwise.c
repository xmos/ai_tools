
#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


#ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #define CONFIG_SYMMETRIC_SATURATION_add_elementwise CONFIG_SYMMETRIC_SATURATION_GLOBAL
#else
  #ifndef CONFIG_SYMMETRIC_SATURATION_add_elementwise
    #define CONFIG_SYMMETRIC_SATURATION_add_elementwise (0)
  #endif 
#endif

#if CONFIG_SYMMETRIC_SATURATION_add_elementwise
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 


#define ASHR16(A,A_SHR)         (((A_SHR) >= 0)? ((A) >> (A_SHR)) : ((A) << -(A_SHR)))
#define ROUND_SHR(A,A_SHR)     (((A)+(1<<((A_SHR)-1))) >> (A_SHR))

#define MAX(A,B)        (((A) >= (B))? (A) : (B))
#define MIN(A,B)        (((A) <= (B))? (A) : (B))


WEAK_FUNC
void add_elementwise(
    int8_t Y[],
    const int8_t X0[],
    const int8_t X1[],
    const nn_add_params_t* params,
    const unsigned output_start,
    const unsigned output_count)
{
    for(int i = output_start; i < output_start+output_count; i++){

        // Change X1 and X2 so that they have the same quantization

        int64_t acc = params->output.bias;

        const int16_t tmp0 = ASHR16(X0[i], params->input[0].shr);
        acc += ((int32_t)ASHR16(X0[i], params->input[0].shr)) * params->input[0].multiplier;
        acc += ((int32_t)ASHR16(X1[i], params->input[1].shr)) * params->input[1].multiplier;

        acc = ROUND_SHR(acc, params->output.shr);

        acc = MIN(acc, VPU_INT8_MAX);
        acc = MAX(acc,  NEG_SAT_VAL);
        
        Y[i] = (int8_t) acc;
    }   
}