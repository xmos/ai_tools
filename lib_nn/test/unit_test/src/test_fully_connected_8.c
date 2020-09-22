
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


static const char TEST_TARGET[] = "fully_connected_8()";

#if CONFIG_SYMMETRIC_SATURATION_fully_connected_8
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 0 - Basic functionality w/  C_out = 16  and C_in = 32. No input or output tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case0()
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

    int8_t WORD_ALIGNED  Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000010,     0,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000010,     0,             -0x4000,        -0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x00,       0x00,      -0x00000080,     0,              0x4000,  NEG_SAT_VAL    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x20    },
        {   0x02,       0x04,       0x00000000,     0,              0x1000,         0x40    },
        {   0x04,       0x02,       0x00000000,     0,              0x1000,         0x40    },
        {   0x10,       0x08,       0x000001C0,     4,             -0x1000,        -0x47    },
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

        memset(Y, 0xCC, sizeof(Y));

        PRINTF("\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

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





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 1 - Functionality w/ > 1 C_in group. No tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (4 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case1()
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

    int8_t WORD_ALIGNED  Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x1000,         0x40    },
        {   0x00,       0x00,       0x00000100,     0,             -0x1000,        -0x40    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x2000,         0x40    },
        {   0x02,       0x04,       0x00000000,     0,              0x0400,         0x40    },
        {   0x04,       0x02,       0x00000000,     0,              0x0400,         0x40    },
        {   0x08,       0x02,       0x000000E0,     4,             -0x2000,        -0x47    },
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
        
        memset(Y, 0xCC, sizeof(Y));

        PRINTF("\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

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





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 2 - Functionality w/ > 1 C_out group. No tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (3 * VPU_INT8_ACC_PERIOD)
#define C_in            (VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case2()
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

    int8_t WORD_ALIGNED  Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x1000,         0x40    },
        {   0x00,       0x00,       0x00000100,     0,             -0x1000,        -0x40    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x20    },
        {   0x02,       0x04,       0x00000000,     0,              0x1000,         0x40    },
        {   0x04,       0x02,       0x00000000,     0,              0x0800,         0x20    },
        {   0x04,       0x04,       0x000000E0,     4,             -0x2000,        -0x17    },
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

        memset(Y, 0xCC, sizeof(Y));

        PRINTF("\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

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





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 3 - More than 1 C_out group and more than one C_in group. No Tails
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (3 * VPU_INT8_ACC_PERIOD)
#define C_in            (4 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case3()
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

    int8_t WORD_ALIGNED  Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x1000,         0x40    },
        {   0x00,       0x00,       0x00000100,     0,             -0x1000,        -0x40    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x2000,         0x40    },
        {   0x02,       0x04,       0x00000000,     0,              0x0400,         0x40    },
        {   0x04,       0x02,       0x00000000,     0,              0x0400,         0x40    },
        {   0x08,       0x02,       0x000000E0,     4,             -0x2000,        -0x47    },
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

        
        memset(Y, 0xCC, sizeof(Y));

        PRINTF("\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

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





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 4 - C_in < VPU_INT8_EPV
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (12)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case4()
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

    int8_t WORD_ALIGNED  Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },    // 0
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x1000,         0x40    },
        {   0x00,       0x00,       0x00000100,     0,             -0x0800,        -0x20    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },

        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },    // 7
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         C_in    },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         8*C_in  },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         8*C_in  },
        
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

        memset(Y, 0xCC, sizeof(Y));
        
        PRINTF("\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

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





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 5 - Multiple C_in groups with a tail
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (2 * VPU_INT8_EPV + 4)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case5()
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

    int8_t WORD_ALIGNED  Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x0400,         0x10    },
        {   0x00,       0x00,       0x00000100,     0,             -0x0400,        -0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x44    },
        {   0x02,       0x04,       0x00000000,     0,              0x0800,         0x44    },
        {   0x04,       0x02,       0x00000000,     0,              0x0800,         0x44    },
        {   0x10,       0x08,       0x000000E0,     1,              0x2000,         0x7F    },
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

        
        memset(Y, 0xCC, sizeof(Y));
        
        PRINTF("\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

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





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 6 -  C_out < 16  (tests both even and odd C_out). No input tail.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (12)
#define C_in            (2 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case6()
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

    int8_t WORD_ALIGNED Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x1000,         0x40    },
        {   0x00,       0x00,       0x00000100,     0,             -0x1000,        -0x40    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x40    },
        {   0x02,       0x04,       0x00000000,     0,              0x0400,         0x20    },
        {   0x04,       0x02,       0x00000000,     0,              0x0400,         0x20    },
        {   0x08,       0x04,       0x000000E0,     4,             -0x2000,        -0x47    },
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

            memset(Y, 0xCC, sizeof(Y));
        
            PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
            fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out_tmp);

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int8_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int8_t) 0xCC;

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




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 7 -  C_out > 16 but not a multiple of 16. No Input tail.  (Tests even AND odd C_out)
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (3 * VPU_INT8_ACC_PERIOD + 6)
#define C_in            (2 * VPU_INT8_EPV)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case7()
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

    int8_t WORD_ALIGNED Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x0080,         0x02    },
        {   0x00,       0x00,       0x00000100,     0,             -0x0080,        -0x02    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x40    },
        {   0x02,       0x04,       0x00000000,     0,              0x0200,         0x10    },
        {   0x04,       0x02,       0x00000000,     0,              0x0200,         0x10    },
        {   0x08,       0x04,       0x000000E0,     4,             -0x2000,        -0x47    },
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

            memset(Y, 0xCC, sizeof(Y));
        
            PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
            fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out_tmp);

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int8_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int8_t) 0xCC;

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






///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 8 -  C_out < 16 (Even and Odd) with input tail.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (12)
#define C_in            (24)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case8()
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

    int8_t WORD_ALIGNED Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,             -0x4000,        -0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x18    },
        {   0x02,       0x04,       0x00000000,     0,              0x2000,         0x60    },
        {   0x04,       0x02,       0x00000000,     0,              0x2000,         0x60    },
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -0x67    },
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

            memset(Y, 0xCC, sizeof(Y));
        
            PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
            fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out_tmp);

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int8_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int8_t) 0xCC;

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






///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 9 -  Multiple input and output groups, with both input and output tails.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)
#define C_in            (3 * VPU_INT8_EPV + 24)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case9()
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

    int8_t WORD_ALIGNED Y[C_out]        = { 0 };

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t w;
        int32_t bias;
        int16_t shift;
        int16_t scale;
        int8_t y;
    } case_t;

    case_t casses[] = {
            //X         //W         //Bias          //Shift         //Scale         //Y
        {   0x00,       0x00,       0x00000000,     0,              0x0000,         0x00    },
        {   0x00,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x00,       0x00000001,     0,              0x4000,         0x01    },
        {   0x00,       0x00,       0x00000100,     0,              0x1000,         0x40    },
        {   0x00,       0x00,       0x00000100,     0,             -0x1000,        -0x40    },
        {   0x00,       0x00,       0x00000100,     4,              0x4000,         0x10    },
        {   0x00,       0x00,       0x00000100,     4,              0x2000,         0x08    },
        {   0x01,       0x00,       0x00000000,     0,              0x4000,         0x00    },
        {   0x00,       0x01,       0x00000000,     0,              0x4000,         0x00    },
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         0x78    },
        {   0x02,       0x04,       0x00000000,     0,              0x0400,         0x3C    },
        {   0x04,       0x02,       0x00000000,     0,              0x0400,         0x3C    },
        {   0x04,       0x08,       0x000000E0,     4,             -0x2000,        -0x7F    },
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

        memset(Y, 0xCC, sizeof(Y));
        
        PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

        PRINTF("\t\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            int8_t exp_val = casse->y;

            if(Y[c] != exp_val)
                sprintf(str_buff, "(vector: %u) (index: %u)", v, c);

            TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);

        }


    }

}
#undef ceil_C_out
#undef C_in
#undef C_out





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 10 -  Weights aren't constant. (Inputs are)
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)
#define C_in            (2 * VPU_INT8_EPV + 24)
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case10()
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

    int8_t WORD_ALIGNED Y[C_out];

    PRINTF("%s...\n", __func__);

    for(int k = 0; k < C_in; k++){
        X[k] = 1;
    }

    for(int k = 0; k < C_out; k++){
        for(int j = 0; j < C_in; j++){
            W[k][j] = k + j - 22;
        }
    }

    for(int k = 0; k < C_out; k++){

        // sum = C_in * (k - 22) + (C_in-1)*(C_in/2)
        BSO.B[k] = 0x0000;
        BSO.shift1[k] = 1;

        // acc = C_in/2 * (k-22) + (C_in-1)*(C_in/4)
        BSO.scale[k] = -0x2000;

        // acc = -(2**13) * ( C_in/2 * (k-22) + (C_in-1)*(C_in/4) )
        BSO.offset_scale[k] = (1<<13);
        BSO.offset[k]       = (C_in-1)*(C_in/4);
        
        // acc = -(2**13) * C_in/2 * (k-22)
        BSO.shift2[k] = 17;
        
        // acc = - C_in/32.0 * (k-22)
    }
    
    nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                    (int32_t*) &BSO.B, 
                    (int16_t*) &BSO.shift1, 
                    (int16_t*) &BSO.scale, 
                    (int16_t*) &BSO.offset_scale,
                    (int16_t*) &BSO.offset,
                    (int16_t*) &BSO.shift2, 
                    NULL, C_out  );

        memset(Y, 0xCC, sizeof(Y));
        
        PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

        PRINTF("\t\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){
        
        int32_t acc = -(1<<13)*(C_in/2)*(c-22) + (1<<16);

        int8_t exp_val = acc >> 17;

        if(Y[c] != exp_val)
            sprintf(str_buff, "(index: %u)", c);

        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 11 -  Weights differ for each output channel. Inputs aren't constant.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)  // = 44
#define C_in            (3 * VPU_INT8_EPV + 24)         // = 120
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case11()
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

    int8_t WORD_ALIGNED Y[C_out];

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
        BSO.scale[k] = -0x2000; // - (2**13)
        BSO.offset_scale[k] = 1<<14;
        BSO.offset[k]       = 1;
        BSO.shift2[k] = 19;
    }
    

    nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                    (int32_t*) &BSO.B, 
                    (int16_t*) &BSO.shift1, 
                    (int16_t*) &BSO.scale, 
                    (int16_t*) &BSO.offset_scale,
                    (int16_t*) &BSO.offset,
                    (int16_t*) &BSO.shift2, 
                    NULL, C_out  );

    memset(Y, 0xCC, sizeof(Y));
        
    PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
    fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, C_out);

    PRINTF("\t\t\tChecking...\n");
    char str_buff[200] = {0};
    for(unsigned c = 0; c < C_out; c++){

        // X[] = [0,1,2,...,C_in-1] - 64
        // W[c][] = [c-24, c-24, c-24, ..., c-24]

        // X[]*W[c][] = [(c-24)*(0-64), (c-24)*(1-64), ..., (c-24)*(C_in-1-64)]
        //            = (c-24) * [(0-64), (1-64), (2-64), ..., (C_in-1-64)]
        //            = (c-24) * ([0, 1, 2, ..., C_in-1] - [64, 64, 64, ..., 64])

        // sum[c] = (c-24) * ((0-64) + (1-64) + (2-64) + ... + (C_in-1-64))
        //        = (c-24) * (0+1+2+...+(C_in-1) - 64 * C_in)
        //        = (c-24) * ( (C_in-1)*(C_in/2) - 64 * C_in )
        //        = (c-24)*(C_in-1)*(C_in/2) - 64*(c-24)*C_in

        // sum[0] = -24 * (C_in-1)*(C_in/2) - 64*(-24)*C_in = -24*(119)*(60) - 64*(-24)*120
        //        = -171360 - -(184320) = 12960

        // sum[43]= (43-24)*119*60 - 64*(43-24)*120 = -10260
        
        // after shift1: acc[c] = ((sum[c]+1) >> 1)   // +1 is rounding logic
        // after scale:  acc[c] = ((sum[c]+1) >> 1) * -(2**13)
        // after offset: acc[c] = ((sum[c]+1) >> 1) * -(2**13) + (2**14)
        // after shift2: acc[c] = (((sum[c]+1) >> 1) * -(2**13) + (2**14) + (1<<18)) >> 19   // +(1<<18) is rounding logic

        // final[0] = (((sum[0]+1) >> 1) * -(2**13) + (2**14) + (1<<18)) >> 19
        //          = 6480 * -2**13

        int32_t sum_c = (c-24)*(C_in-1)*(C_in/2) - 64*(c-24)*C_in;

        int8_t exp_val = (((sum_c+1) >> 1) * -(1<<13) + (1<<14) + (1<<18)) >> 19;

        // PRINTF("exp[%u] = %d\n", c, exp_val);

        if(Y[c] != exp_val)
            sprintf(str_buff, "(index: %u)", c);

        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out







///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 12 -  Check that things work correctly when out_chan_count == 0
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)  // = 44
#define C_in            (3 * VPU_INT8_EPV + 24)         // = 120
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case12()
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

    int8_t WORD_ALIGNED Y[C_out];

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
        BSO.scale[k] = -0x2000; // - (2**13)
        BSO.offset_scale[k] = 1<<14;
        BSO.offset[k]       = 1;
        BSO.shift2[k] = 19;
    }
    

    nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                    (int32_t*) &BSO.B, 
                    (int16_t*) &BSO.shift1, 
                    (int16_t*) &BSO.scale, 
                    (int16_t*) &BSO.offset_scale,
                    (int16_t*) &BSO.offset,
                    (int16_t*) &BSO.shift2, 
                    NULL, C_out  );

    memset(Y, 0xCC, sizeof(Y));
        
    PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
    fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, C_in, 0, 0);

    PRINTF("\t\t\tChecking...\n");
    char str_buff[200] = {0};
    for(unsigned c = 0; c < C_out; c++){

        int8_t exp_val = 0xCC;

        if(Y[c] != exp_val)
            sprintf(str_buff, "(index: %u)", c);

        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Case 13 -  Check that out_chan_start and out_chan_count are respected
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C_out           (2 * VPU_INT8_ACC_PERIOD + 12)  // = 44
#define C_in            (3 * VPU_INT8_EPV + 24)         // = 120
#define ceil_C_out      (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2)
void test_fully_connected_8_case13()
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

    int8_t WORD_ALIGNED Y[C_out];

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
        BSO.scale[k] = -0x2000; // - (2**13)
        BSO.offset_scale[k] = 1<<14;
        BSO.offset[k]       = 1;
        BSO.shift2[k] = 19;
    }

    nn_standard_BSO_layout(  (nn_bso_block_t*) &BSO, 
                    (int32_t*) &BSO.B, 
                    (int16_t*) &BSO.shift1, 
                    (int16_t*) &BSO.scale, 
                    (int16_t*) &BSO.offset_scale,
                    (int16_t*) &BSO.offset,
                    (int16_t*) &BSO.shift2, 
                    NULL, C_out  );



    typedef struct {
        channel_count_t out_chan_start;
        channel_count_t out_chan_count;
        unsigned line;
    } case_t;

    case_t casses[] = {
    //  {  start,      count,     line     }
        {      0,      C_out,     __LINE__ },
        {      0,          1,     __LINE__ },
        {      0,          2,     __LINE__ },
        {      0,          8,     __LINE__ },
        {      0,         17,     __LINE__ },
        {      0,         18,     __LINE__ },
        {      0,         24,     __LINE__ },
        {     16,          1,     __LINE__ },
        {     16,          2,     __LINE__ },
        {     16,          8,     __LINE__ },
        {     16,         17,     __LINE__ },
        {     16,         18,     __LINE__ },
        {     16,         24,     __LINE__ },
    };

    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(Y, 0xCC, sizeof(Y));
            
        PRINTF("\t\t\tCalling %s...\n", TEST_TARGET);
        fully_connected_8((int8_t*) Y, (int8_t*) W, (int8_t*) X, (nn_bso_block_t*) &BSO, 
                            C_in, casse->out_chan_start, casse->out_chan_count);

        PRINTF("\t\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            // exp[0] = (((sum[0]+1) >> 1) * -(2**13) + (2**14) + (1<<18)) >> 19
            //          = 6480 * -2**13

            int32_t sum_c = (c-24)*(C_in-1)*(C_in/2) - 64*(c-24)*C_in;

            int8_t exp_val = (((sum_c+1) >> 1) * -(1<<13) + (1<<14) + (1<<18)) >> 19;

            if( (c < casse->out_chan_start) || (c >= (casse->out_chan_start + casse->out_chan_count)))
                exp_val = 0xCC;

            if(Y[c] != exp_val)
                sprintf(str_buff, "(vector: %u) (index: %u) (line: %u)", v, c, casse->line);

            TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y[c], str_buff);
        }
    }
}
#undef ceil_C_out
#undef C_in
#undef C_out






void test_fully_connected_8()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_fully_connected_8_case0);
    RUN_TEST(test_fully_connected_8_case1);
    RUN_TEST(test_fully_connected_8_case2);
    RUN_TEST(test_fully_connected_8_case3);
    RUN_TEST(test_fully_connected_8_case4);
    RUN_TEST(test_fully_connected_8_case5);
    RUN_TEST(test_fully_connected_8_case6);
    RUN_TEST(test_fully_connected_8_case7);
    RUN_TEST(test_fully_connected_8_case8);
    RUN_TEST(test_fully_connected_8_case9);
    RUN_TEST(test_fully_connected_8_case10);
    RUN_TEST(test_fully_connected_8_case11);
    RUN_TEST(test_fully_connected_8_case12);
    RUN_TEST(test_fully_connected_8_case13);
}