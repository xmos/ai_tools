
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

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)

static unsigned seed = 4412311;





unsafe {





#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define BUFF_SIZE       4096
void test_vpu_memcpy_case0()
{

    PRINTF("test_vpu_memcpy_case0()...\n");

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

    PRINTF("test_vpu_memcpy_case1()...\n");

    uint8_t WORD_ALIGNED src[BUFF_SIZE];
    uint8_t WORD_ALIGNED dst[BUFF_SIZE];

    pseudo_rand_bytes(&seed, src, BUFF_SIZE);

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





}