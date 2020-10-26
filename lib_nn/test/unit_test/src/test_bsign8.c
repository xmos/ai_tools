
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
// #include "../src/nn_op_helper.h"
#include "xs3_vpu.h"

#include "unity.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

// for sprintf() calls
static char str_buff[200];

// Reference bsign implementation (currently copied from larq-compute-engine);
void larq_ref_bsign(int8_t *input, int32_t *output, size_t inputLength, int32_t zeroPoint);

void gen_expected(int8_t *input, uint32_t *output, size_t inputLength, int32_t zeroPoint)
{
    larq_ref_bsign(input, (int32_t*)output, inputLength, zeroPoint);
}

#define MAX_OUTPUT_WORDS (128)
#define MAX_JOBS (4)

void run_bsign_test(int8_t* x, size_t inputLength, int8_t zeroPoint, size_t jobCount)
{
    uint32_t WORD_ALIGNED y[MAX_OUTPUT_WORDS];
    uint32_t WORD_ALIGNED y_exp[MAX_OUTPUT_WORDS];

    memset(y, 0xAA, sizeof(y));
    memset(y_exp, 0xCC, sizeof(y_exp));
    
    size_t outputLength = (inputLength/32) + ((inputLength % 32) != 0);
   
    gen_expected(x, y_exp, inputLength, zeroPoint); 
  
    nn_bsign_8_job_t jobs[MAX_JOBS];
    nn_bsign_8_plan_t plan;

    bsign_8_prepare(&plan, jobs, inputLength, zeroPoint, jobCount);

    /* Compare our reference implementation against the (external) golden reference) */
    for(int i = 0; i < jobCount; i++)
    {
        bsign_8_ref(y, x, &plan, &jobs[i]);
    }

    for(int i = 0; i < outputLength; i++)
    {
        if(y[i] != y_exp[i])
                sprintf(str_buff, "(jobs: %zu) (inputLen: %zu) (index: %d)", jobCount, inputLength, i);

        TEST_ASSERT_EQUAL_MESSAGE(y_exp[i], y[i], str_buff);
    }

    /* Compare our optimised version for the platform under test */
    for(int i = 0; i < jobCount; i++)
    {
        bsign_8(y, x, &plan, &jobs[i]);
    }
    
    for(int i = 0; i < outputLength; i++)
    {
        if(y[i] != y_exp[i])
                sprintf(str_buff, "(jobs: %zu) (inputLen: %zu) (index: %d)", jobCount, inputLength, i);

        TEST_ASSERT_EQUAL_MESSAGE(y_exp[i], y[i], str_buff);
    }
}


/* A few basic short singular job rected tests and zero length corner */
void test_bsign_8_basic0()
{
    PRINTF("%s...\n", __func__);
   
    int8_t WORD_ALIGNED x[VPU_INT8_EPV+8]  = {
        0xFF, 0x01, 0x7F, 0x80, 0xFF, 0x7F, 0x7E, 0x00, 
        0x80, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0xFF,  
        0xFF, 0x01, 0x7F, 0x80, 0xFF, 0x7F, 0x7E, 0x00, 
        0x80, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0xFF,

        0x10, 0x01, 0xFF, 0x55, 0xAA, 0xCC, 0x7F, 0xF0 
    };

    run_bsign_test(x, VPU_INT8_EPV, 0, 1);
    run_bsign_test(x, VPU_INT8_EPV - 8, +1, 1);
    run_bsign_test(x, VPU_INT8_EPV + 8, -1, 1);
    run_bsign_test(x, 0, 0, 1);
}

/* A few longer basic directed tests */
void test_bsign_8_basic1()
{
    PRINTF("%s...\n", __func__);
  
    #define MULT 7 
    size_t jobLen = VPU_INT8_EPV * MULT;

    int8_t WORD_ALIGNED test_data[VPU_INT8_EPV]  = {
        0xFF, 0x01, 0x7F, 0x80, 0xFF, 0x7F, 0x7E, 0x00, 
        0x80, 0x80, 0x80, 0x7E, 0x00, 0x00, 0x00, 0xFF, 
        0xFE, 0x01, 0x7F, 0x80, 0xFF, 0x7F, 0x7E, 0x00, 
        0x80, 0x80, 0x80, 0x7E, 0x00, 0x00, 0x00, 0xFF
    };

    int8_t WORD_ALIGNED x[VPU_INT8_EPV * MULT];

    int index = 0;
    for (int i = 0; i < MULT; i++)
        for(int j = 0; j < VPU_INT8_EPV; j++)
        {
            x[index++] = test_data[j];
        }

    run_bsign_test(x,jobLen, 0, 1);
    run_bsign_test(x, jobLen-8, 1, 1);
    run_bsign_test(x, jobLen-16, -1, 1);
    run_bsign_test(x, jobLen-24, 0, 1);
}


#define MAX_INPUT_BYTES (512)
#define REPS (100)

/* Pseudo random test including multiple jobs */
void test_bsign_8_rand0()
{
    uint32_t WORD_ALIGNED y[MAX_OUTPUT_WORDS] = {0};
    uint32_t WORD_ALIGNED y_exp[MAX_OUTPUT_WORDS] = {0};
    int8_t WORD_ALIGNED x[MAX_INPUT_BYTES]  = {0};
    int8_t WORD_ALIGNED x_orig[MAX_INPUT_BYTES]  = {0};

    nn_bsign_8_job_t jobs[MAX_JOBS];
    nn_bsign_8_plan_t plan;

    for(int r = 0; r < REPS; r++)
    {
        PRINTF("%s...\n", __func__);
        unsigned inputLen = pseudo_rand_uint16() % (MAX_INPUT_BYTES+1);

        const int8_t zeroPoint = pseudo_rand_int8();

        // Only test with input lengths that produce byte aligned output sizes
        inputLen = (inputLen >> 3) << 3;

        pseudo_rand_bytes((char*)x_orig, sizeof(x_orig));
        vpu_memcpy(x, x_orig, sizeof(x));

        size_t jobCount = (pseudo_rand_uint16() % (MAX_JOBS-1))+1;
    
        run_bsign_test(x, inputLen, zeroPoint, jobCount);
    }
}

void test_bsign_8()
{
    srand(6654734);

    UNITY_SET_FILE();
    
    RUN_TEST(test_bsign_8_basic0);
    RUN_TEST(test_bsign_8_basic1);
    RUN_TEST(test_bsign_8_rand0);

}
