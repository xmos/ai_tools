
#include <stddef.h>
#include <stdio.h>
// #include <stdint.h>
// #include <string.h>
// #include <assert.h>

#include "fake_kernels.h"

int main(void)
{
    printf("starting\n");

    conv2d_deepin_deepout_relu();


    conv2d_shallowin_deepout_relu();

    maxpool2d_deep();

    fc_deepin_shallowout_lin();

    printf("finished\n");
    return 0;
}
