
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "xs3_vpu.h"
#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#define EACH_INT8(ITER)  size_t ITER = 0; ITER <   XS3_VPU_VREG_WIDTH_BYTES; ITER++
#define EACH_INT16(ITER) size_t ITER = 0; ITER < 2*XS3_VPU_VREG_WIDTH_WORDS; ITER++
#define EACH_INT32(ITER) size_t ITER = 0; ITER <   XS3_VPU_VREG_WIDTH_WORDS; ITER++

static void _test_vladd()
{
    vsetc_simple(VEC_INT_8, VEC_SH0, 32);
    vclrdr();
    
    int8_t WORD_ALIGNED data[XS3_VPU_VREG_WIDTH_BYTES] = {0};
    int8_t WORD_ALIGNED vR[XS3_VPU_VREG_WIDTH_BYTES] = {0};

    vldr(vR);

    for(EACH_INT8(i))
        TEST_ASSERT(vR[i] == 0);

    for(EACH_INT8(i)) 
        data[i] = i;

    vladd(data);
    vstr(vR);

    for(EACH_INT8(i))
        TEST_ASSERT(vR[i] == data[i]);
}

void test_vladd()
{
    _test_vladd();
}