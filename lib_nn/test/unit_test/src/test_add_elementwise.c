
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "../src/nn_op_helper.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "unity.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

#ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #define CONFIG_SYMMETRIC_SATURATION_add_elementwise CONFIG_SYMMETRIC_SATURATION_GLOBAL
#else
  #ifndef CONFIG_SYMMETRIC_SATURATION_add_elementwise
    #define CONFIG_SYMMETRIC_SATURATION_add_elementwise (0)
  #endif 
#endif

#if CONFIG_SYMMETRIC_SATURATION_add_elementwise
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 



char msg_buff[200];

#define LENGTH     (16)

// Keep this real simple.
static void test_add_elementwise_case0()
{
    PRINTF("%s...\n", __func__);

    int8_t WORD_ALIGNED Y[LENGTH];
    int8_t WORD_ALIGNED X1[LENGTH];
    int8_t WORD_ALIGNED X2[LENGTH];
    
    int8_t Y_expected[LENGTH];

    for(int i = 0; i < LENGTH; i++){
        X1[i] = X2[i] = i;
    }

    nn_add_params_t params = {
        {   {   0, 0x0001 },
            {   0, 0x0001 } },
            {   0, 1}     };
            
    for(int i = 0; i < LENGTH; i++)
        Y_expected[i] = i;


    add_elementwise(Y, X1, X2, &params, 0, LENGTH);
    TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LENGTH);

}
#undef LENGTH



#define LENGTH     (128)

// Keep this real simple.
static void test_add_elementwise_case1()
{
    PRINTF("%s...\n", __func__);

    int8_t WORD_ALIGNED Y[LENGTH];
    int8_t WORD_ALIGNED X1[LENGTH];
    int8_t WORD_ALIGNED X2[LENGTH];
    
    int8_t Y_expected[LENGTH];

    for(int i = 0; i < LENGTH; i++){
        X1[i] = X2[i] = i;
    }

    nn_add_params_t params = {
        {   {   -8, 0x0001 },
            {   -7, 0x0002 } },
            {  -0x00008000, 8} };

    /* 
        y[i] = 1*(((x1[i] << 8) + 2*(x2[i] << 7)) + bias) >> 8

        y[i] = (i * 2^8 + i * 2^8 + bias) >> 8
             = (2 * i * 2^8 + bias) >> 8
             = i * 2^1 + (bias>>8)
             = 2*i + (bias>>8)

        bias =  -128 * 2^8
        y[i] = 2*i - 0x8000>>8 = 2*i - 128

        y[0] = 2*0 - 128 = -128
        y[127] = 2 * 127 - 128 = 126
    */
            
    for(int i = 0; i < LENGTH; i++)
        Y_expected[i] = 2*i - 128;

    unsigned start = 0;

    { // 0 <= i < 16
        unsigned count = 16;    // One full vector
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count], LENGTH - (start + count) );
        start += count;
    }

    { // 16 <= i < 20
        unsigned count = 4;     // Less than one vector
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count], LENGTH - (start + count) );
        start += count;
    }

    { // 20 <= i < 52
        unsigned count = 32;    // Two full vectors
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count], LENGTH - (start + count) );
        start += count;
    }

    { // 52 <= i < 128
        unsigned count = 76;    // 4 vectors and change.
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
    }

}
#undef LENGTH


#define LEN     (100)
#define REPS    (200)

static void test_add_elementwise_case2()
{
    PRINTF("%s...\n", __func__);

    int8_t WORD_ALIGNED Y[LEN];
    int8_t WORD_ALIGNED X0[LEN];
    int8_t WORD_ALIGNED X1[LEN];
    int8_t Y_expected[LEN];
    
    for(int v = 0; v < REPS; v++){

        unsigned elm_start = (pseudo_rand_uint32() % LEN) & 0xFFFFFFFC;
        unsigned elm_count = pseudo_rand_uint32() % (LEN - elm_start);

        PRINTF("  rep %u... (%u <= k < %u)\n", v, elm_start, elm_start+elm_count);

        nn_add_params_t params;

        int32_t min = 0;
        int32_t max = 0;

        for(int i = 0; i < 2; i++){
            params.input[i].shr = -8 + (pseudo_rand_uint16() % 2);
            params.input[i].multiplier = 1 + (pseudo_rand_uint16() >> 1);

            min += (((int32_t)-128)<<(-params.input[i].shr))*params.input[i].multiplier;
            max += (((int32_t) 127)<<(-params.input[i].shr))*params.input[i].multiplier;
        }

        uint32_t diff = max - min;

        unsigned scale = ceil_log2(diff);

        params.output.shr = scale - 8;

        params.output.bias = 0;

        pseudo_rand_bytes((char*)X0, LEN);
        pseudo_rand_bytes((char*)X1, LEN);

        memset(Y_expected, 0xCC, sizeof(Y_expected));

        for(int i = elm_start; i < elm_start+elm_count; i++){
            int32_t acc = params.output.bias;

            int32_t x0 = ((int32_t) X0[i]) << (-params.input[0].shr);
            acc += x0 * params.input[0].multiplier;

            int32_t x1 = ((int32_t) X1[i]) << (-params.input[1].shr);
            acc += x1 * params.input[1].multiplier;

            Y_expected[i] = vlsat_single_s8(acc, params.output.shr, NEG_SAT_VAL, VPU_INT8_MAX);
        }

        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X0, X1, &params, elm_start, elm_count);

        if(v == -1){
            PRINTF("    params.input[0].shr        = %d\n", params.input[0].shr);
            PRINTF("    params.input[0].multiplier = %d   (0x%04X)\n", params.input[0].multiplier, (unsigned) params.input[0].multiplier);
            PRINTF("    params.input[1].shr        = %d\n", params.input[1].shr);
            PRINTF("    params.input[1].multiplier = %d   (0x%04X)\n", params.input[1].multiplier, (unsigned) params.input[1].multiplier);

            PRINTF("    max = %ld\n", max);
            PRINTF("    min = %ld\n", min);
            PRINTF("    diff = %lu     (0x%08lX)\n", diff, diff);
            PRINTF("    scale = %u\n", scale);

            PRINTF("    params.output.bias = %ld    (0x%08lX)\n", params.output.bias, (uint32_t)params.output.bias);
            PRINTF("    params.output.shr = %d\n", params.output.shr);

            unsigned m = 13;
            PRINTF("      X0[%u] = %d\n", m, X0[m]);
            PRINTF("      X1[%u] = %d\n", m, X1[m]);
            PRINTF("      Y_expected[%u] = %d\n", m, Y_expected[m]);
            PRINTF("      Y[%u] = %d\n", m, Y[m]);
        }


        TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LEN);

    }
}
#undef LEN
#undef REPS

void test_add_elementwise()
{
    srand(563456);

    UNITY_SET_FILE();
    
    RUN_TEST(test_add_elementwise_case0);
    RUN_TEST(test_add_elementwise_case1);
    RUN_TEST(test_add_elementwise_case2);
}