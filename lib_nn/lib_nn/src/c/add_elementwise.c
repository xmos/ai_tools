
#include <stdint.h>
#include "vpu_sim.h"

typedef struct {
    int16_t input1_shr;
    int16_t input2_shr;
    int16_t input1_offset; 
    int16_t input2_offset;
    int16_t input1_multiplier; /* Q1.14 */
    int16_t input2_multiplier; /* Q1.14 */
    int16_t output_multiplier; /* Q1.14 */
    int16_t output_offset;
    int32_t dummy; //for word-alignment. For now.
} add_params3_t;

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


void add_elementwise3(
    int8_t Y[],
    const int8_t X1[],
    const int8_t X2[],
    const add_params3_t* params, //per-channel? If so, need to add C_in and make this an array.
    const unsigned output_start,
    const unsigned output_count)
{

    for(int i = output_start; i < output_count; i++){

        // Change X1 and X2 so that they have the same quantization
        const int16_t tmp1 = REQUANT(X1[i], params->input1_shr, params->input1_offset, params->input1_multiplier);
        const int16_t tmp2 = REQUANT(X2[i], params->input2_shr, params->input2_offset, params->input2_multiplier);
        
        // Add them together
        int16_t out16 = tmp2 + tmp1;

        // Requantize the result with the output quantization
        out16 = MUL_Q14(out16, params->output_multiplier);
        out16 = out16 + params->output_offset;
        Y[i] = VDEPTH8(out16);

    }   

}