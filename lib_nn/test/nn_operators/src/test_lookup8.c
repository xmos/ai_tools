
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "unity.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)





#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define MAX_LEN         128
#define REPS            50
void test_lookup8_case0()
{
    unsigned seed = 12312314;

    PRINTF("%s...\n", __func__);

    uint8_t WORD_ALIGNED src[MAX_LEN] = { 0 };

    uint8_t WORD_ALIGNED lut[256];

    uint8_t WORD_ALIGNED dst[MAX_LEN] = { 0 };

    pseudo_rand_bytes(&seed, (char*) src, MAX_LEN);
    pseudo_rand_bytes(&seed, (char*) lut, 256);


    for(int k = 0; k < REPS; k++){

        PRINTF("\trep %d...\n", k); 

        uint32_t size = pseudo_rand_uint32(&seed) % MAX_LEN;

        memset(dst, 0xCC, sizeof(dst));
        
        lookup8((uint8_t*)dst, (uint8_t*) src, (uint8_t*) lut, size);
        for(int i = 0; i < MAX_LEN; i++)
            TEST_ASSERT_EQUAL( (i < size)? lut[src[i]] : 0xCC, dst[i]);

        

    }

}
#undef REPS
#undef MAX_LEN
#undef DEBUG_ON



void test_lookup8()
{
    UNITY_SET_FILE();

    RUN_TEST(test_lookup8_case0);
}