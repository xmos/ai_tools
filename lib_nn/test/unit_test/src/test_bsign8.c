
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "../src/nn_op_helper.h"
#include "xs3_vpu.h"

#include "unity.h"

#include <print.h>
#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

////////////////////////////////////////////////
// for sprintf() calls
static char str_buff[200];

// Reference bsign implementation (copied from larq-compute-engine);
void larq_ref_bsign(int8_t *input, uint32_t *output, size_t inputLength, int32_t zero_point);



void gen_expected(int8_t *input, uint32_t *output, size_t inputLength)
{

#if 0
    uint32_t j = 0;
    uint32_t shift = 0;
    uint32_t bits = 0;
    for(int i = 0; i < input_length; i++)
    {
        if(input[i] < 0)
        {
            bits |= (1 << shift);
        }
        
        shift++;
    
        if(shift == 32)
        {
            shift = 0;
            output[j++] = bits;
            bits = 0;
        }
    }

    // Tail tidy - don't write a full word.. i.e. same as VSTRPV
    if (shift != 0)
        output[j] = ((output[j] >> shift)<< shift) | bits;
#else

    larq_ref_bsign(input, output, inputLength, 0);

#endif

}

void test_bsign_8_case0()
{
    PRINTF("%s...\n", __func__);
    
    #define MAX_OUTPUT_WORDS 10
    #define MAX_INPUT_BYTES  32

    uint32_t WORD_ALIGNED y[MAX_OUTPUT_WORDS] = {0};
    uint32_t WORD_ALIGNED y_exp[MAX_OUTPUT_WORDS] = {0};
   
    int8_t WORD_ALIGNED x[MAX_INPUT_BYTES]  = {
        0xFF, 0x01, 0x7F, 0x80,
        0xFF, 0x7F, 0x7E, 0x00, 
        0x80, 0x80, 0x80, 0x80,
        0, 0, 0, 0xFF, 
        
        0xFF, 0x01, 0x7F, 0x80,
        0xFF, 0x7F, 0x7E, 0x00, 
        0x80, 0x80, 0x80, 0x80,
        0, 0, 0, 0xFF, 

    };

    gen_expected(x, y_exp, VPU_INT8_EPV); 

    nn_bsign_8_job_t job;

    bsign_8_init(&job, VPU_INT8_EPV, 1);
    bsign_8(y, x, &job);
   
    TEST_ASSERT_EQUAL(y_exp[0], y[0]);

    memset(y, 0, sizeof(y));

    y_exp[0] &= 0xFFFFFF;

    bsign_8_init(&job, 24, 1);
    bsign_8(y, x, &job);

    TEST_ASSERT_EQUAL(y_exp[0], y[0]);
}

void test_bsign_8_case1()
{
    PRINTF("%s...\n", __func__);
  
    #define MULT 7 
    size_t jobLen = VPU_INT8_ACC_PERIOD * MULT;
       
    uint32_t WORD_ALIGNED y[VPU_INT32_ACC_PERIOD] = {0};
    uint32_t WORD_ALIGNED y_exp[VPU_INT32_ACC_PERIOD] = {0};
   
    int8_t WORD_ALIGNED test_data[VPU_INT8_ACC_PERIOD ]  = {
        0xFF, 0x01, 0x7F, 0x80,
        0xFF, 0x7F, 0x7E, 0x00, 
        0x80, 0x80, 0x80, 0x7E,
        0, 0, 0, 0xFF
    };
    int8_t WORD_ALIGNED x[VPU_INT8_ACC_PERIOD * MULT]  = {0};

    int index = 0;
    for (int i = 0; i < MULT; i++)
        for(int j = 0; j < VPU_INT8_ACC_PERIOD; j++)
        {
            x[index++] = test_data[j] * i;
        }

    gen_expected(x, y_exp, jobLen); 

    nn_bsign_8_job_t job;

    bsign_8_init(&job, jobLen, 1);
    bsign_8(y, x, &job);

    for(int i = 0; i < (jobLen/32)+1; i++)
        TEST_ASSERT_EQUAL(y_exp[i], y[i]);
}

#define MAX_INPUT_LEN_BYTES (512)
#define MAX_OUTPUT_LEN_WORDS (MAX_INPUT_LEN_BYTES)
#define MAX_JOBS (4)
#define REPS (50)

void test_bsign_8_case2()
{
    uint32_t WORD_ALIGNED y[MAX_OUTPUT_LEN_WORDS] = {0};
    uint32_t WORD_ALIGNED y_exp[MAX_OUTPUT_LEN_WORDS] = {0};
    int8_t WORD_ALIGNED x[MAX_INPUT_LEN_BYTES]  = {0};
    int8_t WORD_ALIGNED x_orig[MAX_INPUT_LEN_BYTES]  = {0};

    nn_bsign_8_job_t jobs[MAX_JOBS];
   
    for(int r = 0; r < REPS; r++)
    {
        PRINTF("%s...\n", __func__);
        unsigned inputLen = pseudo_rand_uint16() % (MAX_INPUT_LEN_BYTES+1);

        // Only test with input lengths that produce byte aligned output sizes
        inputLen = (inputLen >> 3) << 3;


        pseudo_rand_bytes((char*)x_orig, sizeof(x_orig));
        vpu_memcpy(x, x_orig, sizeof(x));

        memset(y, 0xAA, sizeof(y));
        memset(y_exp, 0xAA, sizeof(y_exp));

        size_t jobCount = (pseudo_rand_uint16() % (MAX_JOBS-1))+1;

        bsign_8_init(jobs, inputLen, jobCount);

        gen_expected(x, y_exp, inputLen); 

        for(int i = 0; i < jobCount; i++)
        {
            bsign_8(y, x, &jobs[i]);
        }
        
        for(int i = 0; i < (inputLen/32)+1; i++)
        {
            if(y[i] != y_exp[i])
                sprintf(str_buff, "(rep: %d) (jobs: %d) (inputLen: %u) (index: %d) (x[%d] = %d)", r, jobCount, inputLen, i, i, x_orig[i]);

            TEST_ASSERT_EQUAL_MESSAGE(y_exp[i], y[i], str_buff);
        }
    }
}

void test_bsign_8()
{
    srand(6654734);

    UNITY_SET_FILE();
    
    RUN_TEST(test_bsign_8_case0);
    RUN_TEST(test_bsign_8_case1);
    RUN_TEST(test_bsign_8_case2);

}
