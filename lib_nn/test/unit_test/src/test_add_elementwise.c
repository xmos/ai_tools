
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

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

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


typedef struct {
    int32_t left_shift;
    int32_t input1_offset;
    int32_t input2_offset;
    int32_t input1_multiplier;
    int32_t input2_multiplier;
    int32_t input1_shift;
    int32_t input2_shift;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t output_offset;
} add_params_t;

typedef struct {
    int16_t input1_shr;
    int16_t input2_shr;
    int16_t input1_offset; 
    int16_t input2_offset;
    int16_t input1_multiplier; /* Q1.14 */
    int16_t input2_multiplier; /* Q1.14 */
    int16_t output_multiplier; /* Q1.14 */
    int16_t output_offset;
    int32_t dummy; //for word-alignment. For now.
} add_params3_t;

void add_elementwise(
    int8_t Y[],
    const int8_t X1[],
    const int8_t X2[],
    const add_params_t* params, //per-channel? If so, need to add C_in and make this an array.
    const unsigned output_start,
    const unsigned output_count);

/*

        tmp1[k] <-- (((X1[k] >> params->input1_shr) + params->input1_offset) * params->input1_multiplier) >> 14
        tmp2[k] <-- (((X2[k] >> params->input2_shr) + params->input2_offset) * params->input2_multiplier) >> 14

        Y[k] = (((tmp1[k] + tmp2[k]) * params->output_multiplier) >> 14) + params->output_offset) >> 8

*/
void add_elementwise3(
    int8_t Y[],
    const int8_t X1[],
    const int8_t X2[],
    const add_params3_t* params, //per-channel? If so, need to add C_in and make this an array.
    const unsigned output_start,
    const unsigned output_count);

#define MAX_LEN     (32)

void test_add_elementwise_case0()
{
    PRINTF("%s...\n", __func__);

    int8_t WORD_ALIGNED Y[MAX_LEN];
    int8_t WORD_ALIGNED X1[MAX_LEN];
    int8_t WORD_ALIGNED X2[MAX_LEN];
    
    typedef struct {
        int8_t x1;
        int8_t x2;
        add_params_t params;
        int8_t exp;
    } test_case_t;


    //  v1[i] =  (((X1[i]+offset1)<<shl) * multiplier1) >> (31 + shr1);
    //  v2[i] =  (((X2[i]+offset2)<<shl) * multiplier2) >> (31 + shr2);
    //   Y[i] =  (((v1[i]+v2[i]) * out_mult) >> (31+out_shr)) + out_off;

    const test_case_t casses[] = {
    //       X1,    X2, { shl, offset1, offset2, multiplier1, multiplier2, shr1, shr2,    out_mult, out_shr, out_off },   exp }
        {  0x00,  0x00, {   0,  0x0000,  0x0000,  0x00000000,  0x00000000,    0,    0,  0x00000000,       0,  0x0000 },  0x00 },
        {  0x00,  0x00, {   0,  0x0000,  0x0000,  0x40000000,  0x40000000,    0,    0,  0x40000000,       0,  0x0000 },  0x00 },
        {  0x00,  0x00, {   0,  0x0000,  0x0000,  0x40000000,  0x40000000,    0,    0,  0x40000000,       0,  0x0001 },  0x01 },
        {  0x00,  0x00, {   0,  0x0000,  0x0000,  0x40000000,  0x40000000,    0,    0,  0x40000000,       0,  0x007F },  0x7F },
        {  0x00,  0x00, {   0,  0x0000,  0x0000,  0x40000000,  0x40000000,    0,    0,  0x40000000,       0, -0x0001 }, -0x01 },
        {  0x00,  0x00, {   0,  0x0000,  0x0000,  0x40000000,  0x40000000,    0,    0,  0x40000000,       0, -0x007F }, -0x7F },
        {  0x10,  0x00, {   0,  0x0000,  0x0000,  0x40000000,  0x40000000,   -1,    0,  0x40000000,      -1,  0x0000 },  0x10 },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];
        PRINTF("\ttest vector %u...\n", v);

        for(int len = 4; len <= MAX_LEN; len += 4){
            // PRINTF("\t\tlength %u...\n", len);
        
            memset(X1, casse->x1, len);
            memset(&X1[len], 0xAA, sizeof(X1)-len);

            memset(X2, casse->x2, len);
            memset(&X2[len], 0xBB, sizeof(X2)-len);

            memset(Y, 0xCC, sizeof(Y));

            add_elementwise(Y, X1, X2, &casse->params, 0, len);

            for(int i = 0; i < len; i++){
                // printf("%d\t%d\n", casse->exp, Y[i]);
                TEST_ASSERT_EQUAL(casse->exp, Y[i]);
            }
            for(int i = len; i < MAX_LEN; i++){
                TEST_ASSERT_EQUAL((int8_t)0xCC, Y[i]);
            }

        }
    }
}
#undef MAX_LEN



















#define MAX_LEN     (40)

void test_add_elementwise3_case0()
{
    PRINTF("%s...\n", __func__);

    int8_t WORD_ALIGNED Y[MAX_LEN];
    int8_t WORD_ALIGNED X1[MAX_LEN];
    int8_t WORD_ALIGNED X2[MAX_LEN];
    
    typedef struct {
        int8_t x1;
        int8_t x2;
        add_params3_t params;
        int8_t exp;
    } test_case_t;



    const test_case_t casses[] = {
    //       X1,    X2, { shr1, shr2, offset1, offset2,   mult1,   mult2, out_mul, out_off },   exp }
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x0000,  0x0000,  0x0000,  0x0000 },  0x00 }, //0
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x00 },
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0001 },  0x00 },
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0100 },  0x01 },

        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0200 },  0x02 }, //4
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0080 },  0x01 },
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x007F },  0x00 },
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000, -0x0001 },  0x00 },

        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000, -0x0100 }, -0x01 }, //8
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000, -0x0200 }, -0x02 },
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000, -0x0080 }, -0x00 },
        {  0x00,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000, -0x0081 }, -0x01 },

        {  0x01,  0x00, {    0,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x00 }, //12
        {  0x01,  0x00, {   -8,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x01 },
        {  0x02,  0x00, {   -8,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x02 },
        { -0x01,  0x00, {   -8,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 }, -0x01 },

        {  0x01,  0x00, {   -9,    0,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x02 }, //16
        {  0x01,  0x01, {   -8,   -8,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x02 },
        {  0x01,  0x02, {   -8,   -8,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x03 },
        {  0x01,  0x02, {   -8,   -9,  0x0000,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x05 },

        {  0x00,  0x00, {    0,    0,  0x0200,  0x0000,  0x4000,  0x4000,  0x4000,  0x0000 },  0x02 }, //20
        {  0x00,  0x00, {    0,    0,  0x0100,  0x0100,  0x4000,  0x4000,  0x4000,  0x0000 },  0x02 },
        {  0x00,  0x00, {    0,    0,  0x0100,  0x0200,  0x4000,  0x4000,  0x4000,  0x0000 },  0x03 },
        {  0x00,  0x00, {    0,    0,  0x0100,  0x0400,  0x4000,  0x4000,  0x4000,  0x0000 },  0x05 },

        {  0x10,  0x00, {   -8,    0,  0x0000,  0x0000,  0x2000,  0x4000,  0x4000,  0x0000 },  0x08 }, //24
        {  0x10,  0x10, {   -8,   -8,  0x0000,  0x0000,  0x2000,  0x2000,  0x4000,  0x0000 },  0x10 },
        {  0x10,  0x20, {   -8,   -8,  0x0000,  0x0000,  0x6000,  0x1000,  0x4000,  0x0000 },  0x20 },
        {  0x10,  0x20, {   -8,   -9,  0x0000,  0x0000,  0x6000,  0x1000,  0x4000,  0x0000 },  0x28 },

        {  0x10,  0x00, {   -8,    0,  0x0000,  0x0000,  0x2000,  0x4000,  0x2000,  0x0000 },  0x04 }, //28
        {  0x10,  0x10, {   -8,   -8,  0x0000,  0x0000,  0x2000,  0x2000, -0x8000,  0x0000 }, -0x20 },
        {  0x10,  0x20, {   -8,   -8,  0x0000,  0x0000,  0x6000,  0x1000,  0x1000,  0x0000 },  0x08 },
        {  0x10,  0x20, {   -8,   -9,  0x0000,  0x0000,  0x6000,  0x1000,  0x6000,  0x0000 },  0x3C },

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];
        PRINTF("\ttest vector %u...\n", v);

        for(int len = 4; len <= MAX_LEN; len += 4){
            // PRINTF("\t\tlength %u...\n", len);
        
            memset(X1, casse->x1, len);
            memset(&X1[len], 0xAA, sizeof(X1)-len);

            memset(X2, casse->x2, len);
            memset(&X2[len], 0xBB, sizeof(X2)-len);

            memset(Y, 0xCC, sizeof(Y));

            add_elementwise3(Y, X1, X2, &casse->params, 0, len);

            for(int i = 0; i < len; i++){
                // printf("%d\t%d\n", casse->exp, Y[i]);
                TEST_ASSERT_EQUAL(casse->exp, Y[i]);
            }
            for(int i = len; i < MAX_LEN; i++){
                TEST_ASSERT_EQUAL((int8_t)0xCC, Y[i]);
            }

        }
    }
}
#undef MAX_LEN

void test_add_elementwise()
{
    srand(6654734);

    UNITY_SET_FILE();
    
    RUN_TEST(test_add_elementwise_case0);
    RUN_TEST(test_add_elementwise3_case0);
}