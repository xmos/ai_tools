
#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


// ASHR16(A,A_SHR) --> floor( A * 2**(-A_SHR) )
#define ASHR16(A,A_SHR)     (((A_SHR) >= 0)? ((A) >> (A_SHR)) : ((A) << -(A_SHR)))

//  MUL_Q14(A,B) -->  round((A * B)/(2.0**14))  //(provided A and B are int16)
#define MUL_Q14(A,B)   (((((int32_t)(A)) * (B)) + (1<<13)) >> 14)

// VDEPTH8(A) -->  round(A / (2**-8))
#define VDEPTH8(A)      ((int8_t)(((A) + (1<<7)) >> 8))

// "Requantize" an 8-bit value into a 16-bit value
// The assumption is that increasing the bit depth to 16 bits will avoid excessive information loss.
// (X1 and X2 start with different quantization parameters, but adding together values with different quantization
//  parameters doesn't make sense, so we need to requantize X1 and X2 to have the same quantization parameters
//  before adding them.)
#define REQUANT(A, A_SHR, OFFSET, MULT)     MUL_Q14((ASHR16(A, A_SHR) + (OFFSET)), (MULT))

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
        const int16_t tmp0 = REQUANT(X0[i], params->input[0].shr, params->input[0].offset, params->input[0].multiplier);
        const int16_t tmp1 = REQUANT(X1[i], params->input[1].shr, params->input[1].offset, params->input[1].multiplier);
        
        // Add them together
        int16_t out16 = tmp0 + tmp1;

        // Requantize the result with the output quantization
        out16 = MUL_Q14(out16, params->output.multiplier);
        out16 = out16 + params->output.offset;
        Y[i] = VDEPTH8(out16);

    }   

}