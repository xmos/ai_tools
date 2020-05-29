
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






///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 0 - Basic functionality w/  C_out = 16  and C_in = 32. No input or output tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case0()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0020    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x0100    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x0100    },
        
        //  ((X * W * 32 + B) >> shift) * (scale / 2^14) 
        //  = ((2^4 * 2^3 * 2^5  + 0xE0) >> 4) * -(2^-1)
        //  = - ((2^12 + 0xE0) >> 4) / 2    =   - (2^8 + 0xE) / 2
        //  = - (2^7 + 0x7)     = - 0x87
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x0087    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);
    
    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSO.B[k] = casse->bias;
            BSO.shift1[k] = casse->shift;
            BSO.scale[k] = casse->scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 14;
        }

        nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                        (int32_t*) &BSO.B, 
                        (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                        (int16_t*) &BSO.shift2, 
                        NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                     (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            if(Y[c] != casse->y)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y[c], str_buff);

        }
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 1 - Functionality w/ > 1 C_in group. No tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (4 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case1()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0080    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x0400    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x0400    },
        
        //  ((X * W * 128 + B) >> shift) * (scale / 2^14) 
        //  = ((2^4 * 2^3 * 2^7  + 0xE0) >> 4) * -(2^-1)
        //  = - ((2^14 + 0xE0) >> 4) / 2    =   - (2^10 + 0xE) / 2
        //  = - (2^9 + 0x7)     = - 0x207
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x0207    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSO.B[k] = casse->bias;
            BSO.shift1[k] = casse->shift;
            BSO.scale[k] = casse->scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 14;
        }

        
        nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                        (int32_t*) &BSO.B, 
                        (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                        (int16_t*) &BSO.shift2, 
                        NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                     (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            if(Y[c] != casse->y)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y[c], str_buff);

        }
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 2 - Functionality w/ > 1 C_out group. No tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (3 * VPU_INT8_ACC_PERIOD)
#define C_in            (VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case2()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0020    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x0100    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x0100    },
        
        //  ((X * W * 32 + B) >> shift) * (scale / 2^14) 
        //  = ((2^4 * 2^3 * 2^5  + 0xE0) >> 4) * -(2^-1)
        //  = - ((2^12 + 0xE0) >> 4) / 2    =   - (2^8 + 0xE) / 2
        //  = - (2^7 + 0x7)     = - 0x87
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x0087    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSO.B[k] = casse->bias;
            BSO.shift1[k] = casse->shift;
            BSO.scale[k] = casse->scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 14;
        }


        nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                        (int32_t*) &BSO.B, 
                        (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                        (int16_t*) &BSO.shift2, 
                        NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                     (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){
            if(Y[c] != casse->y)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y[c], str_buff);
        }
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 3 - More than 1 C_out group and more than one C_in group. No Tails
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (3 * VPU_INT8_ACC_PERIOD)
#define C_in            (4 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case3()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0080    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x0400    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x0400    },
        
        //  ((X * W * 32 + B) >> shift) * (scale / 2^14) 
        //  = ((2^4 * 2^3 * 2^5  + 0xE0) >> 4) * -(2^-1)
        //  = - ((2^12 + 0xE0) >> 4) / 2    =   - (2^8 + 0xE) / 2
        //  = - (2^7 + 0x7)     = - 0x87
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x0207    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSO.B[k] = casse->bias;
            BSO.shift1[k] = casse->shift;
            BSO.scale[k] = casse->scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 14;
        }


        nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                        (int32_t*) &BSO.B, 
                        (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                        (int16_t*) &BSO.shift2, 
                        NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                     (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            if(Y[c] != casse->y)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y[c], str_buff);
        }
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 4 - C_in < VPU_INT8_EPV
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (12)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case4()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },    // 0
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },    // 7
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         C_in      },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         8*C_in    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         8*C_in    },
        
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -(4*C_in+0x7)    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSO.B[k] = casse->bias;
            BSO.shift1[k] = casse->shift;
            BSO.scale[k] = casse->scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 14;
        }


        nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                        (int32_t*) &BSO.B, 
                        (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                        (int16_t*) &BSO.shift2, 
                        NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                     (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            if(Y[c] != casse->y)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y[c], str_buff);
        }
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 5 - Multiple C_in groups with a tail
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (2 * VPU_INT8_EPV + 4)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case5()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },

        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0044    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x0220    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x0220    },
        
        //  ((X * W * C_in + B) >> shift) * (scale / 2^14) 
        //  = ((2^4 * 2^3 * 68  + 0xE0) >> 1) * -(2^-1)
        //  = ((2^4 * 2^3 * 2^2 * 17 + 0xE0) >> 1) * -(2^-1)
        //  = - ((2^9 * 17 + 0xE0) >> 2)       =   - (2^7 * 17 + 0x38)
        //  = - 0x880 - 0x38 = - 8B8 
        {   0x10,       0x08,       0x000000E0,     1,             -0x2000,        -0x08B8    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSO.B[k] = casse->bias;
            BSO.shift1[k] = casse->shift;
            BSO.scale[k] = casse->scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 14;
        }


        nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                        (int32_t*) &BSO.B, 
                        (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                        (int16_t*) &BSO.shift2, 
                        NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                     (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            if(Y[c] != casse->y)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y[c], str_buff);

        }

    }

}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 6 -  C_out < 16  (tests both even and odd C_out). No input tail.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (12)
#define C_in            (2 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case6()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0040    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x0200    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x0200    },
        
        //  ((X * W * 64 + B) >> shift) * (scale / 2^14) 
        //  = ((2^4 * 2^3 * 2^6  + 0xE0) >> 4) * -(2^-1)
        //  = - ((2^13 + 0xE0) >> 4) / 2    =   - (2^9 + 0xE) / 2
        //  = - (2^7 + 0x7)     = - 0x87
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x0107    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        //Test both even and odd values for C_out, since they
        //  follow slightly different paths internally
        for(int oddness = 0; oddness < 2; oddness++) {

            const unsigned C_out_tmp = C_out - oddness;

            PRINTF("\t\tC_out = %u...\n", C_out_tmp);

            memset(X, casse->x, sizeof(X));
            memset(W, casse->w, sizeof(W));

            for(int k = 0; k < C_out_tmp; k++){
                BSO.B[k] = casse->bias;
                BSO.shift1[k] = casse->shift;
                BSO.scale[k] = casse->scale;
                BSO.offset_scale[k] = 0;
                BSO.offset[k]       = 0;
                BSO.shift2[k] = 14;
            }


            nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                            (int32_t*) &BSO.B, 
                            (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                            (int16_t*) &BSO.shift2, 
                            NULL, C_out  );

            nn_fully_connected_plan_t plan;
            nn_fully_connected_job_t job;
            fully_connected_init(&plan, &job, C_in, C_out_tmp, NULL, 1);

            PRINTF("\t\t\tC...\n");
            memset(Y, 0xCC, sizeof(Y));
            fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                        (nn_bso_block_t*) &BSO, &plan, &job);

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int16_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int16_t) 0xCCCC;

                if(Y[c] != exp_val)
                    sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);

            }

        }

    }

}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 7 -  C_out > 16 but not a multiple of 16. No Input tail.  (Tests even AND odd C_out)
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (3 * VPU_INT8_ACC_PERIOD + 6)
#define C_in            (2 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case7()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0040    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x0200    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x0200    },
        
        //  ((X * W * 64 + B) >> shift) * (scale / 2^14) 
        //  = ((2^4 * 2^3 * 2^6  + 0xE0) >> 4) * -(2^-1)
        //  = - ((2^13 + 0xE0) >> 4) / 2    =   - (2^9 + 0xE) / 2
        //  = - (2^7 + 0x7)     = - 0x87
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x0107    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        //Test both even and odd values for C_out, since they
        //  follow slightly different paths internally
        for(int oddness = 0; oddness < 2; oddness++) {

            const unsigned C_out_tmp = C_out - oddness;

            PRINTF("\t\tC_out = %u...\n", C_out_tmp);

            memset(X, casse->x, sizeof(X));
            memset(W, casse->w, sizeof(W));

            for(int k = 0; k < C_out_tmp; k++){
                BSO.B[k] = casse->bias;
                BSO.shift1[k] = casse->shift;
                BSO.scale[k] = casse->scale;
                BSO.offset_scale[k] = 0;
                BSO.offset[k]       = 0;
                BSO.shift2[k] = 14;
            }


            nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                            (int32_t*) &BSO.B, 
                            (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, 
                            (int16_t*) &BSO.offset_scale,
                            (int16_t*) &BSO.offset,
                            (int16_t*) &BSO.shift2, 
                            NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out_tmp, NULL, 1);

            PRINTF("\t\t\tC...\n");
            memset(Y, 0xCC, sizeof(Y));
            fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                        (nn_bso_block_t*) &BSO, &plan, &job);

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int16_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int16_t) 0xCCCC;

                if(Y[c] != exp_val)
                    sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);

            }

        }

    }

}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON












///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 8 -  C_out < 16 (Even and Odd) with input tail.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (12)
#define C_in            (24)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case8()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0018    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x00C0    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x00C0    },
        
        //  (2^4 * 2^3 * 24 + 0xE0) >> 5
        //  (2^7 * 2^3 * 3 + 0xE0) >> 5
        //  (2^10 * 3 + 0xE0) >> 5
        //  ( 2^5 * 3 + 0x7)
        //  0x60 + 0x7
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x0067    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        //Test both even and odd values for C_out, since they
        //  follow slightly different paths internally
        for(int oddness = 0; oddness < 2; oddness++) {

            const unsigned C_out_tmp = C_out - oddness;

            PRINTF("\t\tC_out = %u...\n", C_out_tmp);

            memset(X, casse->x, sizeof(X));
            memset(W, casse->w, sizeof(W));

            for(int k = 0; k < C_out_tmp; k++){
                BSO.B[k] = casse->bias;
                BSO.shift1[k] = casse->shift;
                BSO.scale[k] = casse->scale;
                BSO.offset_scale[k] = 0;
                BSO.offset[k]       = 0;
                BSO.shift2[k] = 14;
            }


            nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                            (int32_t*) &BSO.B, 
                            (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, 
                            (int16_t*) &BSO.offset_scale,
                            (int16_t*) &BSO.offset,
                            (int16_t*) &BSO.shift2, 
                            NULL, C_out  );

            nn_fully_connected_plan_t plan;
            nn_fully_connected_job_t job;
            fully_connected_init(&plan, &job, C_in, C_out_tmp, NULL, 1);

            PRINTF("\t\t\tC...\n");
            memset(Y, 0xCC, sizeof(Y));
            fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                        (nn_bso_block_t*) &BSO, &plan, &job);

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int16_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int16_t) 0xCCCC;

                if(Y[c] != exp_val)
                    sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);

            }

        }

    }

}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON












///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 9 -  Multiple input and output groups, with both input and output tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)
#define C_in            (3 * VPU_INT8_EPV + 24)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case9()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out]        = { 0 };

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int16_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x0000    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x0001    },
        {   0x00,       0x00,       0x00000100,     0,              0x4000,         0x0100    },
        {   0x00,       0x00,       0x00000100,     0,             -0x4000,        -0x0100    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x0010    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x0008    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x0000    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x0078    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         0x03C0    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         0x03C0    },
        
        //  (2^4 * 2^3 * 120 + 0xE0) >> 5
        //  (2^7 * 2^3 * 15 + 0xE0) >> 5
        //  (2^10 * 15 + 0xE0) >> 5
        //  ( 2^5 * 15 + 0x7)
        //  0x1E0 + 0x7
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x01E7    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSO.B[k] = casse->bias;
            BSO.shift1[k] = casse->shift;
            BSO.scale[k] = casse->scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 14;
        }

        nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                        (int32_t*) &BSO.B, 
                        (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, 
                        (int16_t*) &BSO.offset_scale,
                        (int16_t*) &BSO.offset,
                        (int16_t*) &BSO.shift2, 
                        NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        PRINTF("\t\t\tC...\n");
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                    (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            int16_t exp_val = casse->y;

            if(Y[c] != exp_val)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);

        }


    }

}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON












///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 10 -  Weights aren't constant. (Inputs are)
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)
#define C_in            (3 * VPU_INT8_EPV + 24)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case10()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out];

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    for(int k = 0; k < C_in; k++){
        X[k] = 1;
    }

    for(int k = 0; k < C_out; k++){
        for(int j = 0; j < C_in; j++){
            W[k][j] = k + j - 64;
        }
    }

    for(int k = 0; k < C_out; k++){
        BSO.B[k] = 0x0000;
        BSO.shift1[k] = 1;
        BSO.scale[k] = -0x2000;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 14;
    }
    

    nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                    (int32_t*) &BSO.B, 
                    (int16_t*) &BSO.shift1, 
                    (int16_t*) &BSO.scale, 
                    (int16_t*) &BSO.offset_scale,
                    (int16_t*) &BSO.offset,
                    (int16_t*) &BSO.shift2, 
                    NULL, C_out  );

        nn_fully_connected_plan_t plan;
        nn_fully_connected_job_t job;
        fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

        PRINTF("\t\t\tC...\n");
        memset(Y, 0xCC, sizeof(Y));
        fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                    (nn_bso_block_t*) &BSO, &plan, &job);

        PRINTF("\t\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

        int16_t exp_val = -(C_in*(c-64) + (C_in/2)*(C_in-1))/4;

        if(Y[c] != exp_val)
            sprintf(str_buff, "C failed. (index: %u)", c);

        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON












///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 11 -  Weights differ for each output channel. Inputs aren't constant.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)
#define C_in            (3 * VPU_INT8_EPV + 24)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_16_case11()
{
    int8_t   WORD_ALIGNED  W[C_out][C_in]   = {{ 0 }};
    int8_t   WORD_ALIGNED  X[C_in]          = { 0 };
    
    struct {
        int32_t B[ceil_C_out];
        int16_t shift1[ceil_C_out];
        int16_t scale[ceil_C_out];
        int16_t offset_scale[ceil_C_out];
        int16_t offset[ceil_C_out];
        int16_t shift2[ceil_C_out];
    } BSO;

    int16_t WORD_ALIGNED  Y[C_out];

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("%s...\n", __func__);

    for(int k = 0; k < C_in; k++){
        X[k] = k-64 ;
    }

    for(int k = 0; k < C_out; k++){
        for(int j = 0; j < C_in; j++){
            W[k][j] = k - 24;
        }
    }

    for(int k = 0; k < C_out; k++){
        BSO.B[k] = 0x0000;
        BSO.shift1[k] = 1;
        BSO.scale[k] = -0x2000;
        BSO.offset_scale[k] = 1<<14;
        BSO.offset[k]       = 1;
        BSO.shift2[k] = 14;
    }
    

    nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                    (int32_t*) &BSO.B, 
                    (int16_t*) &BSO.shift1, 
                    (int16_t*) &BSO.scale, 
                    (int16_t*) &BSO.offset_scale,
                    (int16_t*) &BSO.offset,
                    (int16_t*) &BSO.shift2, 
                    NULL, C_out  );

    nn_fully_connected_plan_t plan;
    nn_fully_connected_job_t job;
    fully_connected_init(&plan, &job, C_in, C_out, NULL, 1);

    PRINTF("\t\t\tC...\n");
    memset(Y, 0xCC, sizeof(Y));
    fully_connected_16((int16_t*) Y, (int8_t*) W, (int8_t*) X,
                                (nn_bso_block_t*) &BSO, &plan, &job);

    PRINTF("\t\t\tChecking...\n");
    char str_buff[200] = {0};
    for(unsigned c = 0; c < C_out; c++){

        // X[] = [1,2,3,...,C_in] - 128
        // W[c][] = [c-24, c-24, c-24, ..., c-24]

        // X[]*W[c][] = [(c-24)*(0-128), (c-24)*(1-128), ..., (c-24)*(C_in-1-128)]
        //            = (c-24) * [(0-128), (1-128), (2-128), ..., (C_in-1-128)]

        // sum = (c-24) * ((0-128) + (1-128) + (2-128) + ... + (C_in-1-128))
        //     = (c-24) * (0+1+2+...+(C_in-1) - 128 * C_in)
        //     = (c-24) * ( (C_in-1)*(C_in/2) - 128 * C_in )
        //     = (c-24)*(C_in-1)*(C_in/2) - 128*(c-24)*C_in
        
        // -((sum >> 1) / 2)

        int16_t exp_val = (-( (c-24)*(C_in-1)*(C_in/2) - 64*(c-24)*C_in ) / 4) + 1;

        if(Y[c] != exp_val)
            sprintf(str_buff, "C failed. (index: %u)", c);

        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON


void test_fully_connected_16()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_fully_connected_16_case0);
    RUN_TEST(test_fully_connected_16_case1);
    RUN_TEST(test_fully_connected_16_case2);
    RUN_TEST(test_fully_connected_16_case3);
    RUN_TEST(test_fully_connected_16_case4);
    RUN_TEST(test_fully_connected_16_case5);
    RUN_TEST(test_fully_connected_16_case6);
    RUN_TEST(test_fully_connected_16_case7);
    RUN_TEST(test_fully_connected_16_case8);
    RUN_TEST(test_fully_connected_16_case9);
    RUN_TEST(test_fully_connected_16_case10);
    RUN_TEST(test_fully_connected_16_case11);
}