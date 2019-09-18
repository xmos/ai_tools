#include <stdint.h>
#include <stdio.h>

#include "fake_kernels.h"

void conv2d_deepin_deepout_relu()
{
    uint64_t sum = 0;
    for (uint64_t i=0; i < 100; i++)
    {
        sum += 1;
    }
    printf("%llu\n", sum);
}

void conv2d_shallowin_deepout_relu()
{
    uint64_t sum = 0;
    for (uint64_t i=0; i < 10; i++)
    {
        sum += 1;
    }
    printf("%llu\n", sum);
}

void maxpool2d_deep()
{
    uint64_t sum = 0;
    for (uint64_t i=0; i < 10; i++)
    {
        sum += 1;
    }
    printf("%llu\n", sum);
}

void fc_deepin_shallowout_lin()
{
    uint64_t sum = 0;
    for (uint64_t i=0; i < 1000; i++)
    {
        sum += 1;
    }
    printf("%llu\n", sum);
}
