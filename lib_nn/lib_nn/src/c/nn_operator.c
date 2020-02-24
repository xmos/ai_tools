

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>




void argmax_16_c(
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

void requantize_16_to_8_c(
    int8_t* y,
    const int16_t* x,
    const unsigned n)
{
    for(int i = 0; i < n; i++){
        y[i] = vdepth8_single_s16(x[i]);
    }
}
