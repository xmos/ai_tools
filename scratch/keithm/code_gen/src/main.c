
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "cifar-10.h"

int main(void)
{
    int8_t XC_conv2d_shallowin_deepout_relu_input[1 * 32 * 32 * 4] = { 0 };
    int32_t XC_argmax_16_output[1 * 1] = { 0 };
    
    printf("starting: argmax_16_output=%ld\n", XC_argmax_16_output[0]);
    xcore_model_quant(XC_conv2d_shallowin_deepout_relu_input, XC_argmax_16_output);
    printf("finished: argmax_16_output=%ld\n", XC_argmax_16_output[0]);
    return 0;
}
