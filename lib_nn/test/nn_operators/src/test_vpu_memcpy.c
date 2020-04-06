
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
#include "unity.h"


#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)







#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define BUFF_SIZE       4096
void test_vpu_memcpy_case0()
{

    PRINTF("%s...\n", __func__);

    uint8_t WORD_ALIGNED src[BUFF_SIZE] = { 0 };
    uint8_t WORD_ALIGNED dst[BUFF_SIZE] = { 0 };

    memset(src, 1, sizeof(src));

    vpu_memcpy(dst, src, BUFF_SIZE-1);

    for(int i = 0; i < BUFF_SIZE-1; i++){
        TEST_ASSERT_EQUAL(1, dst[i]);
    }

    TEST_ASSERT_EQUAL(0, dst[BUFF_SIZE-1]);

}
#undef BUFF_SIZE
#undef DEBUG_ON





#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define BUFF_SIZE       4096
#define REPS            100
void test_vpu_memcpy_case1()
{
    unsigned seed = 452345;

    PRINTF("%s...\n", __func__);

    uint8_t WORD_ALIGNED src[BUFF_SIZE];
    uint8_t WORD_ALIGNED dst[BUFF_SIZE];

    pseudo_rand_bytes(&seed, (char*) src, BUFF_SIZE);

    for(int k = 0; k < REPS; k++){

        PRINTF("\trep %d...\n", k); 

        uint32_t size = pseudo_rand_uint32(&seed) % BUFF_SIZE;
        memset(dst, 0, sizeof(dst));

        vpu_memcpy(dst, src, size);

        for(int i = 0; i < size; i++){
            TEST_ASSERT_EQUAL(src[i], dst[i]);
        }

        for(int i = 0; i < BUFF_SIZE - size; i++){
            TEST_ASSERT_EQUAL(0, dst[size + i]);
        }

    }
}
#undef REPS
#undef BUFF_SIZE
#undef DEBUG_ON



void test_vpu_memcpy()
{
    UNITY_SET_FILE();

    RUN_TEST(test_vpu_memcpy_case0);
    RUN_TEST(test_vpu_memcpy_case1);
}