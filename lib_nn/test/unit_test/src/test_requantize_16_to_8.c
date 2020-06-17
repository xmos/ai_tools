
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
  #define CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8 CONFIG_SYMMETRIC_SATURATION_GLOBAL
#else
  #ifndef CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
    #define CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8 (0)
  #endif 
#endif

#if CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 


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


void test_requantize_16_to_8_case0()
{
    PRINTF("%s...\n", __func__);
    
    int8_t WORD_ALIGNED y[VPU_INT8_ACC_PERIOD] = {0};
    int16_t WORD_ALIGNED x[VPU_INT8_ACC_PERIOD]  = {
        0x0000, 0x0100, 0x007F, 0x0080,
        0x7FFF, 0x7E80, 0x7E7F, 
        0x8000, 0x807F, 0x8080, 0x8034,
        0,0,0
    };

    int8_t y_exp[VPU_INT8_ACC_PERIOD] = {
        0x00, 0x01, 0x00, 0x01,
        0x7F, 0x7F, 0x7E,
        NEG_SAT_VAL, NEG_SAT_VAL, 0x81, NEG_SAT_VAL,
        0,0,0
    };

    nn_requantize_16_to_8_job_t job;

    requantize_16_to_8_init(&job, VPU_INT8_ACC_PERIOD, 1);
    requantize_16_to_8(y, x, &job);

    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i ++)
        TEST_ASSERT_EQUAL(y_exp[i], y[i]);


    memset(y, 0, sizeof(y));

    requantize_16_to_8_init(&job, VPU_INT8_ACC_PERIOD - 1, 1);
    requantize_16_to_8(y, x, &job);

    for(int i = 0; i < VPU_INT8_ACC_PERIOD-1; i ++)
        TEST_ASSERT_EQUAL(y_exp[i], y[i]);

}


/****************************************************************************
 *
 * Case 0 - Checks several specific cases.
 *
 ****************************************************************************/
#define VEC_LEN         (VPU_INT16_EPV)
void test_requantize_16_to_8_case1()
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
        {   -0x8000,    16,  NEG_SAT_VAL },
    };
    const unsigned first_case = 1;
    const unsigned last_case = -1;

    const unsigned N_casses = sizeof(casses) / sizeof(test_case_t);

    const int8_t XXX = 0xCC;
    
    for(int v = first_case; v < N_casses && v <= last_case; v++){

        test_case_t* casse = &casses[v];

        PRINTF("\t\tsub-case %d...\n", v);

        for(unsigned in_place = 0; in_place <= 1; in_place++){


            int8_t* dest = in_place? (int8_t*) x : (int8_t*) y;

            memset16(x, casse->x_val, VEC_LEN);
            memset(y, XXX, VEC_LEN * sizeof(int8_t));

            nn_requantize_16_to_8_job_t job;

            requantize_16_to_8_init(&job, casse->N, 1);
            requantize_16_to_8(dest, (int16_t*)x, &job);

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





/****************************************************************************
 *
 * Case 2 - Random data/length
 *
 ****************************************************************************/
#define MAX_LEN         (512)
#define REPS            50
void test_requantize_16_to_8_case2()
{
    PRINTF("%s...\n", __func__);

    int8_t  WORD_ALIGNED y[MAX_LEN];
    int16_t WORD_ALIGNED x[MAX_LEN];

    int16_t WORD_ALIGNED x_orig[MAX_LEN];
    
    const int8_t XXX = 0xCC;

    for(int v = 0; v < REPS; v++){

        PRINTF("\t\trep %d...\n", v); 

        const unsigned N = pseudo_rand_uint16() % (MAX_LEN+1);

        pseudo_rand_bytes((char*)x_orig, sizeof(x_orig));
        vpu_memcpy(x, x_orig, sizeof(x));
        
        memset(y, XXX, sizeof(y));

        for(int in_place = 0; in_place < 1; in_place++){


            int8_t* dest = in_place? (int8_t*) x : (int8_t*) y;

            nn_requantize_16_to_8_job_t job;

            requantize_16_to_8_init(&job, N, 1);
            requantize_16_to_8(dest, (int16_t*)x, &job);

            for(int i = 0; i < N; i++){

                int8_t exp_val = vdepth8_single_s16(x_orig[i]);
                if(x_orig[i] < -0x7F80)
                    exp_val = NEG_SAT_VAL;

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







/****************************************************************************
 *
 * Case 3 - Random data/length, multiple jobs
 *
 ****************************************************************************/
#define MAX_LEN         (512)
#define MAX_JOBS        (4)
#define REPS            50
void test_requantize_16_to_8_case3()
{
    PRINTF("%s...\n", __func__);

    int8_t  WORD_ALIGNED y[MAX_LEN];
    int16_t WORD_ALIGNED x[MAX_LEN];

    int16_t WORD_ALIGNED x_orig[MAX_LEN];
    
    const int8_t XXX = 0xCC;

    nn_requantize_16_to_8_job_t jobs[MAX_JOBS];

    for(int v = 0; v < REPS; v++){

        PRINTF("\t\trep %d...\n", v); 

        const unsigned N = pseudo_rand_uint16() % (MAX_LEN+1);

        pseudo_rand_bytes((char*)x_orig, sizeof(x_orig));
        vpu_memcpy(x, x_orig, sizeof(x));
        
        memset(y, XXX, sizeof(y));

        const unsigned job_count = (pseudo_rand_uint16() % (MAX_JOBS-1))+1;
        
        requantize_16_to_8_init(jobs, N, job_count);

        for(int in_place = 0; in_place < 1; in_place++){


            int8_t* dest = in_place? (int8_t*) x : (int8_t*) y;

            for(int j = 0; j < job_count; j++)
                requantize_16_to_8(dest, (int16_t*)x, &jobs[j]);
            

            for(int i = 0; i < N; i++){

                int8_t exp_val = vdepth8_single_s16(x_orig[i]);
                if(x_orig[i] < -0x7F80)
                    exp_val = NEG_SAT_VAL;

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



void test_requantize_16_to_8()
{
    srand(6654734);

    UNITY_SET_FILE();
    
    RUN_TEST(test_requantize_16_to_8_case0);
    RUN_TEST(test_requantize_16_to_8_case1);
    RUN_TEST(test_requantize_16_to_8_case2);
    RUN_TEST(test_requantize_16_to_8_case3);
}