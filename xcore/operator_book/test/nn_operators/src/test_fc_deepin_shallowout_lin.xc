
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

// static unsigned seed = 4434;


#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define C_out       (VPU_INT8_ACC_PERIOD)
#define C_in        (VPU_INT8_EPV)
void test_fc_deepin_shallowout_lin_case1()
{

    int8_t   WORD_ALIGNED  W[C_out][C_in]                = {{ 0 }};
    int32_t  WORD_ALIGNED  B[C_out]                      = { 0 };
    int8_t   WORD_ALIGNED  X[C_in];
    int8_t   WORD_ALIGNED  Y_c[C_out];
    uint16_t WORD_ALIGNED  shifts[C_out]                 = { 0 };
    int16_t  WORD_ALIGNED  scales[C_out]                 = { 0 };

    int8_t WORD_ALIGNED Y_expected[C_out] = { 0 };

    memset(Y_c, 0xCC, sizeof(Y_c));

    fc_deepin_shallowout_lin((int8_t*) W, (uint16_t*) B, (int8_t*) X, (int16_t*) Y_c, 
                             C_out, C_in, shifts, scales);

    for(unsigned c = 0; c < C_out; c++){
        char str_buff[100];
        sprintf(str_buff, "(c) = (%u)", c);
        TEST_ASSERT_EQUAL_MESSAGE(Y_expected[c], Y_c[c], str_buff);
    }

}
#undef C_in
#undef C_out
#undef DEBUG_ON


void test_fc_deepin_shallowout_lin()
{
    test_fc_deepin_shallowout_lin_case1();
}

