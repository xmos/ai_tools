
#include <assert.h>
#include <stdio.h>
#include <string.h>

#define DECLARE(FUNC)      void benchmark_##FUNC(int argc, char** argv)

DECLARE(vpu_memcpy);
DECLARE(requantize_16_to_8);
DECLARE(lookup8);
DECLARE(conv2d_deep);
DECLARE(avgpool2d);
DECLARE(nn_conv2d_hstrip_deep);
DECLARE(bconv2d_bin_DIput);

#define elseif(FUNC)  else if(strcmp(#FUNC, argv[1])==0)    benchmark_##FUNC(argc-2, &(argv[2]))

int main(int argc, char** argv)
{
    assert(argc >= 2);
    
    if(strcmp("vpu_memcpy", argv[1])==0)
        benchmark_vpu_memcpy(argc-2, &(argv[2]));
    elseif(requantize_16_to_8);
    elseif(lookup8);
    elseif(avgpool2d);
    elseif(conv2d_deep);
    elseif(conv2d_deep);
    elseif(bconv2d_bin_DIput);
    else {
        printf("Function '%s' unknown.\n", argv[1]);
        assert(0);
    }

}