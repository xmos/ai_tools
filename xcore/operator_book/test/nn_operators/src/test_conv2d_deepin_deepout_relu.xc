
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

// static unsigned seed = 44334;


#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define C_out       (VPU_INT8_ACC_PERIOD)
#define C_in        (VPU_INT8_EPV)
#define K_h         (1)
#define K_w         (1)
#define height      (8)
#define width       (8)
void test_conv2d_deepin_deepout_relu_case1()
{

    int8_t   WORD_ALIGNED  K[C_out][K_h][K_w][C_in]      = {{{{ 0 }}}};
    uint16_t WORD_ALIGNED  B[C_out][2]                   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int8_t   WORD_ALIGNED  Y_c[height][width][C_out];
    uint16_t WORD_ALIGNED  shifts[C_out]                 = { 0 };
    int16_t  WORD_ALIGNED  scales[C_out]                 = { 0 };

    int8_t WORD_ALIGNED Y_expected[height][width][C_out] = {{{ 0 }}};

    memset(Y_c, 0xCC, sizeof(Y_c));

    conv2d_deepin_deepout_relu((int8_t*) K, (uint16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                height, width, K_h, K_w, C_out, C_in, shifts, scales);

    for(unsigned h = 0; h < height; h++){
        for(unsigned w = 0; w < width; w++){
            for(unsigned c = 0; c < C_out; c++){
                char str_buff[100];
                sprintf(str_buff, "(h,w,c) = (%u,%u,%u)", h,w,c);
                TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
            }
        }
    }

}
#undef width
#undef height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON


void test_conv2d_deepin_deepout_relu()
{
    test_conv2d_deepin_deepout_relu_case1();
}

