
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



////////////////////////////////////////////////
static void memset16(
    void* dst,
    const int16_t val,
    const unsigned len)
{
    for(int i = 0; i < len; i++){
        ((int16_t*)dst)[i] = val;
    }
}

////////////////////////////////////////////////
// for sprintf() calls
static char str_buff[200];

/****************************************************************************
 *
 * Case 0 - Checks several specific cases.
 *
 ****************************************************************************/
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define VEC_LEN         (VPU_INT16_EPV)
void test_requantize_16_to_8_case0()
{
    PRINTF("%s...\n", __func__);
    
    int8_t WORD_ALIGNED y[VEC_LEN] = {0};
    int16_t WORD_ALIGNED x[VEC_LEN]  = {0};

    typedef struct {
        int16_t x_val;
        unsigned N;
        int8_t exp_y;
    } test_case_t;

    test_case_t casses[] = {
            //X         //N     //Y
        {    0x0000,    16,      0x00    },
        {    0x0F00,    16,      0x0F    },
        {    0x027F,    16,      0x02    },
        {    0x0280,    16,      0x03    },
        {   -0x0180,    16,     -0x01    },
        {   -0x0181,    16,     -0x02    },
        {    0x0100,     1,      0x01    },
        {    0x0100,     8,      0x01    },
        {    0x0100,    15,      0x01    },
    };
    const unsigned first_case = 1;
    const unsigned last_case = -1;

    const unsigned N_casses = sizeof(casses) / sizeof(test_case_t);

    const int8_t XXX = 0xCC;
    
    for(int v = first_case; v < N_casses && v <= last_case; v++){

        test_case_t* casse = &casses[v];

        PRINTF("\t\tsub-case %d...\n", v);

        for(unsigned in_place = 0; in_place <= 1; in_place++){

#if DEBUG_ON
            PRINTF("\t\t\t%s...\n", in_place? "in-place" : "not in-place");
#endif

            int8_t* dest = in_place? (int8_t*) x : (int8_t*) y;

            memset16(x, casse->x_val, VEC_LEN);
            memset(y, XXX, VEC_LEN * sizeof(int8_t));

            requantize_16_to_8(dest, (int16_t*)x, casse->N);

            for(int k = 0; k < casse->N; k++){
                if(dest[k] != casse->exp_y)
                    sprintf(str_buff, "(vector: %d) (in-place: %u) (k: %d.)", v, in_place, k);
                TEST_ASSERT_EQUAL_MESSAGE(casse->exp_y, dest[k], str_buff);
            }

            if(!in_place){
                for(int k = casse->N; k < VEC_LEN; k++){
                    if(dest[k] != casse->exp_y)
                        sprintf(str_buff, "Operator didn't respect N. (Vector %d. Element index %d.)", v, k);
                    TEST_ASSERT_EQUAL_MESSAGE(XXX, dest[k], str_buff);
                }
            }
        }
    }
}
#undef VEC_LEN
#undef DEBUG_ON





/****************************************************************************
 *
 * Case 1 - Random data/length
 *
 ****************************************************************************/
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define MAX_LEN         (512)
#define REPS            50
void test_requantize_16_to_8_case1()
{
    unsigned seed = 6654734;

    PRINTF("%s...\n", __func__);

    int8_t  WORD_ALIGNED y[MAX_LEN];
    int16_t WORD_ALIGNED x[MAX_LEN];

    int16_t WORD_ALIGNED x_orig[MAX_LEN];
    
    const int8_t XXX = 0xCC;

    for(int v = 0; v < REPS; v++){

        PRINTF("\t\trep %d...\n", v); 

        const unsigned N = pseudo_rand_uint16(&seed) % (MAX_LEN+1);

        pseudo_rand_bytes(&seed, (char*)x_orig, sizeof(x_orig));
        vpu_memcpy(x, x_orig, sizeof(x));
        
        memset(y, XXX, sizeof(y));

        for(int in_place = 0; in_place < 2; in_place++){

#if DEBUG_ON
            PRINTF("\t\t\t%s...\n", in_place? "in-place" : "not in-place");
#endif

            int8_t* dest = in_place? (int8_t*) x : (int8_t*) y;

            requantize_16_to_8(dest, x, N);

            for(int i = 0; i < N; i++){

                int8_t exp_val = vdepth8_single_s16(x_orig[i]);

                if(dest[i] != exp_val)
                    sprintf(str_buff, "(rep: %d) (N: %u) (index: %d) (x[%d] = %d)", v, N, i, i, x_orig[i]);

                TEST_ASSERT_EQUAL_MESSAGE(exp_val, dest[i], str_buff);
            }

            if(!in_place){
                for(int i = N; i < MAX_LEN; i++)
                    TEST_ASSERT_EQUAL(XXX, dest[i]);
            }
        }
    }
}
#undef REPS
#undef MAX_LEN
#undef DEBUG_ON




void test_requantize_16_to_8()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_requantize_16_to_8_case0);
    RUN_TEST(test_requantize_16_to_8_case1);
}