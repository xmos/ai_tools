
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "cifar-10.h"

int main(void)
{
    xc_conv2d_shallowin_deepout_relu_input_t input = { 0 };
    xc_argmax_16_output_t output = { 0 };
    
    printf("starting:\n");
    xcore_model_quant(&input, &output);
    printf("finished: argmax_16_output=%ld\n", output[0]);
    return 0;
}
