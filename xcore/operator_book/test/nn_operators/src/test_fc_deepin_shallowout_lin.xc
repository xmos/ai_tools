
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

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

#if (defined(__XS3A__) && USE_ASM_fc_deepin_shallowout_lin)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

// static unsigned seed = 4434;


#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define C_out       (VPU_INT8_ACC_PERIOD-1)
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

    fc_deepin_shallowout_lin_c((int8_t*) W, (int32_t*) B, (int8_t*) X, (int16_t*) Y_c, 
                               C_out, C_in, shifts, scales);
#if HAS_ASM
    int8_t   WORD_ALIGNED  Y_asm[C_out];
    fc_deepin_shallowout_lin_asm((int8_t*) W, (int32_t*) B, (int8_t*) X, (int16_t*) Y_asm, 
                               C_out, C_in, shifts, scales);
#endif

    for(unsigned c = 0; c < C_out; c++){
        char str_buff[100];
        sprintf(str_buff, "(c) = (%u)", c);
        TEST_ASSERT_EQUAL_MESSAGE(Y_expected[c], Y_c[c], str_buff);
#if HAS_ASM
        TEST_ASSERT_EQUAL_MESSAGE(Y_expected[c], Y_asm[c], str_buff);
#endif
    }
}
#undef C_in
#undef C_out
#undef DEBUG_ON


#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (4)
#define C_in            (VPU_INT8_EPV)
#define TEST_VECTORS    (10)
#define VECTOR_FMT      ("test_data/fc_deepin_shallowout_lin_case2.%u.dat")
#include "../test_data/fc_deepin_shallowout_lin_case2.h"
void test_fc_deepin_shallowout_lin_case2()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]                = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in];
    int32_t  WORD_ALIGNED  B[C_out]                      = { 0 };
    uint16_t WORD_ALIGNED  shifts[C_out]                 = { 0 };
    int16_t  WORD_ALIGNED  scales[C_out]                 = { 0 };
    int16_t  WORD_ALIGNED  Y_expected[C_out]              = { 0 };

    int16_t  WORD_ALIGNED  Y_c[C_out];

    int16_t Y_check[] = { Y_CHECK };

    for(int v = 0; v < TEST_VECTORS; v++){
        memset(Y_c, 0xCC, sizeof(Y_c));

        char filename[100];
        sprintf(filename, VECTOR_FMT, v);

        int input_file = _open(filename, O_RDONLY|O_BINARY, S_IREAD);
        assert(input_file != -1);
        _read(input_file, (char*) W, sizeof(W));
        _read(input_file, (char*) X, sizeof(X));
        _read(input_file, (char*) B, sizeof(B));
        _read(input_file, (char*) shifts, sizeof(shifts));
        _read(input_file, (char*) scales, sizeof(scales));
        _read(input_file, (char*) Y_expected, sizeof(Y_expected));
        _close(input_file);

        // printf("%d\t%d\n", Y_check[v], Y_expected[0]);
        assert(Y_check[v] == Y_expected[0]);

        fc_deepin_shallowout_lin((int8_t*) W, (int32_t*) B, (int8_t*) X, (int16_t*) Y_c, 
                                C_out, C_in, shifts, scales);
#if HAS_ASM
        int8_t   WORD_ALIGNED  Y_asm[C_out];
        fc_deepin_shallowout_lin_asm((int8_t*) W, (int32_t*) B, (int8_t*) X, (int16_t*) Y_asm, 
                               C_out, C_in, shifts, scales);
#endif

        for(unsigned c = 0; c < C_out; c++){
            char str_buff[100];
            sprintf(str_buff, "(v, c) = (%u, %u)", v, c);
            TEST_ASSERT_EQUAL_MESSAGE(Y_expected[c], Y_c[c], str_buff);
#if HAS_ASM
            TEST_ASSERT_EQUAL_MESSAGE(Y_expected[c], Y_asm[c], str_buff);
#endif

        }
    }


}
#undef VECTOR_FMT
#undef TEST_VECTORS
#undef C_in
#undef C_out
#undef DEBUG_ON


#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (15)
#define C_in            (3*VPU_INT8_EPV)
#define TEST_VECTORS    (10)
#define VECTOR_FMT      ("test_data/fc_deepin_shallowout_lin_case3.%u.dat")
#include "../test_data/fc_deepin_shallowout_lin_case3.h"
void test_fc_deepin_shallowout_lin_case3()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]                = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in];
    int32_t  WORD_ALIGNED  B[C_out]                      = { 0 };
    uint16_t WORD_ALIGNED  shifts[C_out]                 = { 0 };
    int16_t  WORD_ALIGNED  scales[C_out]                 = { 0 };
    int16_t  WORD_ALIGNED  Y_expected[C_out]              = { 0 };

    int16_t  WORD_ALIGNED  Y_c[C_out];

    int16_t Y_check[] = { Y_CHECK };

    for(int v = 0; v < TEST_VECTORS; v++){
        memset(Y_c, 0xCC, sizeof(Y_c));

        char filename[100];
        sprintf(filename, VECTOR_FMT, v);

        int input_file = _open(filename, O_RDONLY|O_BINARY, S_IREAD);
        assert(input_file != -1);
        _read(input_file, (char*) W, sizeof(W));
        _read(input_file, (char*) X, sizeof(X));
        _read(input_file, (char*) B, sizeof(B));
        _read(input_file, (char*) shifts, sizeof(shifts));
        _read(input_file, (char*) scales, sizeof(scales));
        _read(input_file, (char*) Y_expected, sizeof(Y_expected));
        _close(input_file);

        // printf("%d\t%d\n", Y_check[v], Y_expected[0]);
        assert(Y_check[v] == Y_expected[0]);

        fc_deepin_shallowout_lin((int8_t*) W, (int32_t*) B, (int8_t*) X, (int16_t*) Y_c, 
                                C_out, C_in, shifts, scales);
#if HAS_ASM
        int8_t   WORD_ALIGNED  Y_asm[C_out];
        fc_deepin_shallowout_lin_asm((int8_t*) W, (int32_t*) B, (int8_t*) X, (int16_t*) Y_asm, 
                               C_out, C_in, shifts, scales);
#endif

        for(unsigned c = 0; c < C_out; c++){
            char str_buff[100];
            sprintf(str_buff, "(v, c) = (%u, %u)", v, c);
            TEST_ASSERT_EQUAL_MESSAGE(Y_expected[c], Y_c[c], str_buff);
#if HAS_ASM
            TEST_ASSERT_EQUAL_MESSAGE(Y_expected[c], Y_asm[c], str_buff);
#endif
        }
    }


}
#undef VECTOR_FMT
#undef TEST_VECTORS
#undef C_in
#undef C_out
#undef DEBUG_ON


void test_fc_deepin_shallowout_lin()
{
    test_fc_deepin_shallowout_lin_case1();
    test_fc_deepin_shallowout_lin_case2();
    test_fc_deepin_shallowout_lin_case3();
}

