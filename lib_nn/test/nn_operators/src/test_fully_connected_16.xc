
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

#if (defined(__XS3A__) && USE_ASM_fc_deepin_shallowout_16)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

// static unsigned seed = 4434;

// #define TEST_ASM ((HAS_ASM) && 1)
#define TEST_ASM (1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)





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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case0()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);
    
    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSS.B[k] = casse->bias;
            BSS.shift[k] = casse->shift;
            BSS.scale[k] = casse->scale;
        }

        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                     (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                      (data16_t*) &BSS, C_in, C_out);
#endif

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

#if TEST_C || TEST_ASM
  #if TEST_C
            if(Y_c[c] != casse->y)
                sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
            if(Y_asm[c] != casse->y)
                sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif



#if TEST_C
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_c[c], str_buff);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_asm[c], str_buff);

#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case1()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSS.B[k] = casse->bias;
            BSS.shift[k] = casse->shift;
            BSS.scale[k] = casse->scale;
        }

        
        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                     (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                      (data16_t*) &BSS, C_in, C_out);
#endif

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

#if TEST_C || TEST_ASM
  #if TEST_C
            if(Y_c[c] != casse->y)
                sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
            if(Y_asm[c] != casse->y)
                sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif



#if TEST_C
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_c[c], str_buff);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_asm[c], str_buff);

#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case2()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSS.B[k] = casse->bias;
            BSS.shift[k] = casse->shift;
            BSS.scale[k] = casse->scale;
        }


        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                     (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                      (data16_t*) &BSS, C_in, C_out);
#endif

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

#if TEST_C || TEST_ASM
  #if TEST_C
            if(Y_c[c] != casse->y)
                sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
            if(Y_asm[c] != casse->y)
                sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif



#if TEST_C
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_c[c], str_buff);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_asm[c], str_buff);

#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case3()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSS.B[k] = casse->bias;
            BSS.shift[k] = casse->shift;
            BSS.scale[k] = casse->scale;
        }


        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                     (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                      (data16_t*) &BSS, C_in, C_out);
#endif

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

#if TEST_C || TEST_ASM
  #if TEST_C
            if(Y_c[c] != casse->y)
                sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
            if(Y_asm[c] != casse->y)
                sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif



#if TEST_C
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_c[c], str_buff);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_asm[c], str_buff);

#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case4()...\n");

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
        {   0x01,       0x01,       0x00000000,     0,              0x4000,         C_in      },
        {   0x02,       0x04,       0x00000000,     0,              0x4000,         8*C_in    },
        {   0x04,       0x02,       0x00000000,     0,              0x4000,         8*C_in    },
        
        {   0x10,       0x08,       0x000000E0,     4,             -0x2000,        -(4*C_in+0x7)    },
    };


    const unsigned N_casses = sizeof(casses) / sizeof(case_t);

    const unsigned start_case = 0;
    const unsigned last_case = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSS.B[k] = casse->bias;
            BSS.shift[k] = casse->shift;
            BSS.scale[k] = casse->scale;
        }


        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                     (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                      (data16_t*) &BSS, C_in, C_out);
#endif

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

#if TEST_C || TEST_ASM
  #if TEST_C
            if(Y_c[c] != casse->y)
                sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
            if(Y_asm[c] != casse->y)
                sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif



#if TEST_C
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_c[c], str_buff);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_asm[c], str_buff);

#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case5()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSS.B[k] = casse->bias;
            BSS.shift[k] = casse->shift;
            BSS.scale[k] = casse->scale;
        }


        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                     (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                      (data16_t*) &BSS, C_in, C_out);
#endif

        PRINTF("\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

#if TEST_C || TEST_ASM
  #if TEST_C
            if(Y_c[c] != casse->y)
                sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
            if(Y_asm[c] != casse->y)
                sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif



#if TEST_C
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_c[c], str_buff);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL_MESSAGE(casse->y, Y_asm[c], str_buff);

#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case6()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

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
                BSS.B[k] = casse->bias;
                BSS.shift[k] = casse->shift;
                BSS.scale[k] = casse->scale;
            }


        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
            PRINTF("\t\t\tC...\n");
            memset(Y_c, 0xCC, sizeof(Y_c));
            fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                        (data16_t*) &BSS, C_in, C_out_tmp);
#endif
#if TEST_ASM
            PRINTF("\t\t\tASM...\n");
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                        (data16_t*) &BSS, C_in, C_out_tmp);
#endif

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int16_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int16_t) 0xCCCC;

#if TEST_C || TEST_ASM
  #if TEST_C
                if(Y_c[c] != exp_val)
                    sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
                if(Y_asm[c] != exp_val)
                    sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif

#if TEST_C
                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_c[c], str_buff);
#endif
#if TEST_ASM
                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_asm[c], str_buff);
#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case7()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

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
                BSS.B[k] = casse->bias;
                BSS.shift[k] = casse->shift;
                BSS.scale[k] = casse->scale;
            }


            fc_boggle_BSS(  (data16_t*) &BSS, 
                            (int32_t*) &BSS.B, 
                            (int16_t*) &BSS.shift, 
                            (int16_t*) &BSS.scale, 
                            NULL, C_out  );

#if TEST_C
            PRINTF("\t\t\tC...\n");
            memset(Y_c, 0xCC, sizeof(Y_c));
            fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                        (data16_t*) &BSS, C_in, C_out_tmp);
#endif
#if TEST_ASM
            PRINTF("\t\t\tASM...\n");
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                        (data16_t*) &BSS, C_in, C_out_tmp);
#endif

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int16_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int16_t) 0xCCCC;

#if TEST_C || TEST_ASM
  #if TEST_C
                if(Y_c[c] != exp_val)
                    sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
                if(Y_asm[c] != exp_val)
                    sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif

#if TEST_C
                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_c[c], str_buff);
#endif
#if TEST_ASM
                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_asm[c], str_buff);
#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case8()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

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
                BSS.B[k] = casse->bias;
                BSS.shift[k] = casse->shift;
                BSS.scale[k] = casse->scale;
            }


            fc_boggle_BSS(  (data16_t*) &BSS, 
                            (int32_t*) &BSS.B, 
                            (int16_t*) &BSS.shift, 
                            (int16_t*) &BSS.scale, 
                            NULL, C_out  );

#if TEST_C
            PRINTF("\t\t\tC...\n");
            memset(Y_c, 0xCC, sizeof(Y_c));
            fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                        (data16_t*) &BSS, C_in, C_out_tmp);
#endif
#if TEST_ASM
            PRINTF("\t\t\tASM...\n");
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                        (data16_t*) &BSS, C_in, C_out_tmp);
#endif

            PRINTF("\t\t\tChecking...\n");
            char str_buff[200] = {0};
            for(unsigned c = 0; c < C_out; c++){

                int16_t exp_val = casse->y;

                if(oddness && c == C_out_tmp)
                    exp_val = (int16_t) 0xCCCC;

#if TEST_C || TEST_ASM
  #if TEST_C
                if(Y_c[c] != exp_val)
                    sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
                if(Y_asm[c] != exp_val)
                    sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif

#if TEST_C
                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_c[c], str_buff);
#endif
#if TEST_ASM
                TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_asm[c], str_buff);
#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out]        = { 0 };
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out]        = { 0 };
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case9()...\n");

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

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= last_case; v++){
        PRINTF("\tvector %u...\n", v);

        case_t* casse = (case_t*) &casses[v];

        memset(X, casse->x, sizeof(X));
        memset(W, casse->w, sizeof(W));

        for(int k = 0; k < C_out; k++){
            BSS.B[k] = casse->bias;
            BSS.shift[k] = casse->shift;
            BSS.scale[k] = casse->scale;
        }

        fc_boggle_BSS(  (data16_t*) &BSS, 
                        (int32_t*) &BSS.B, 
                        (int16_t*) &BSS.shift, 
                        (int16_t*) &BSS.scale, 
                        NULL, C_out  );

#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                    (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                    (data16_t*) &BSS, C_in, C_out);
#endif

        PRINTF("\t\t\tChecking...\n");
        char str_buff[200] = {0};
        for(unsigned c = 0; c < C_out; c++){

            int16_t exp_val = casse->y;

#if TEST_C || TEST_ASM
  #if TEST_C
            if(Y_c[c] != exp_val)
                sprintf(str_buff, "C failed. (vector: %u) (index: %u)", v, c);
  #endif
  #if TEST_ASM
            if(Y_asm[c] != exp_val)
                sprintf(str_buff, "ASM failed. (vector: %u) (index: %u)", v, c);
  #endif
#endif

#if TEST_C
            TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_c[c], str_buff);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_asm[c], str_buff);
#endif

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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out];
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out];
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case10()...\n");

    print_warns(-1, TEST_C, TEST_ASM);

    for(int k = 0; k < C_in; k++){
        X[k] = 1;
    }

    for(int k = 0; k < C_out; k++){
        for(int j = 0; j < C_in; j++){
            W[k][j] = k + j - 64;
        }
    }

    for(int k = 0; k < C_out; k++){
        BSS.B[k] = 0x0000;
        BSS.shift[k] = 1;
        BSS.scale[k] = -0x2000;
    }
    

    fc_boggle_BSS(  (data16_t*) &BSS, 
                    (int32_t*) &BSS.B, 
                    (int16_t*) &BSS.shift, 
                    (int16_t*) &BSS.scale, 
                    NULL, C_out  );

#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y_c, 0xCC, sizeof(Y_c));
    fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                (data16_t*) &BSS, C_in, C_out);
#endif

    PRINTF("\t\t\tChecking...\n");
    char str_buff[200] = {0};
    for(unsigned c = 0; c < C_out; c++){

        // X[] = [1, 1, 1, ..., 1]
        // W[c][] = [c-64+0, c-64+1, c-64+2, ..., c-64+(C_in-1)]

        // X[]*W[c][] = [c-64+0, c-64+1, c-64+2, ..., c-64+(C_in-1)]

        // sum = -C_in*64 + C_in*c + (0 + 1 + 2 + ... + (C_in-1))
        //     = -C_in*64 + C_in*c + (C_in-1) * C_in / 2
        //     = C_in*(c-64) + (C_in/2)*(C_in-1)
        
        // -((sum >> 1) / 2)
        // -(C_in*(c-64) + (C_in/2)*(C_in-1))/4

        int16_t exp_val = -(C_in*(c-64) + (C_in/2)*(C_in-1))/4;

#if TEST_C || TEST_ASM
  #if TEST_C
        if(Y_c[c] != exp_val)
            sprintf(str_buff, "C failed. (index: %u)", c);
  #endif
  #if TEST_ASM
        if(Y_asm[c] != exp_val)
            sprintf(str_buff, "ASM failed. (index: %u)", c);
  #endif
#endif

#if TEST_C
        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_c[c], str_buff);
#endif
#if TEST_ASM
        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_asm[c], str_buff);
#endif


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
        int16_t shift[ceil_C_out];
        int16_t scale[ceil_C_out];
    } BSS;

#if TEST_C
    int16_t WORD_ALIGNED  Y_c[C_out];
#endif
#if TEST_ASM
    int16_t WORD_ALIGNED  Y_asm[C_out];
#endif

#if DEBUG_ON
    PRINTF("&W = 0x%08X\n", W);
    PRINTF("&X = 0x%08X\n", &X[0]);

    PRINTF("C_out = %u\n", C_out);
    PRINTF("ceil_C_out = %u\n", ceil_C_out);
    PRINTF("C_in = %u\n", C_in);
    PRINTF("\n\n");
#endif

    PRINTF("test_fully_connected_16_case11()...\n");
    
    print_warns(-1, TEST_C, TEST_ASM);

    for(int k = 0; k < C_in; k++){
        X[k] = k-64 ;
    }

    for(int k = 0; k < C_out; k++){
        for(int j = 0; j < C_in; j++){
            W[k][j] = k - 24;
        }
    }

    for(int k = 0; k < C_out; k++){
        BSS.B[k] = 0x0000;
        BSS.shift[k] = 1;
        BSS.scale[k] = -0x2000;
    }
    

    fc_boggle_BSS(  (data16_t*) &BSS, 
                    (int32_t*) &BSS.B, 
                    (int16_t*) &BSS.shift, 
                    (int16_t*) &BSS.scale, 
                    NULL, C_out  );

#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y_c, 0xCC, sizeof(Y_c));
    fully_connected_16_c((int16_t*) Y_c, (int8_t*) W, (int8_t*) X,
                                (data16_t*) &BSS, C_in, C_out);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    fully_connected_16_asm((int16_t*) Y_asm, (int8_t*) W, (int8_t*) X,
                                (data16_t*) &BSS, C_in, C_out);
#endif

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

        int16_t exp_val = -( (c-24)*(C_in-1)*(C_in/2) - 64*(c-24)*C_in ) / 4;

#if TEST_C || TEST_ASM
  #if TEST_C
        if(Y_c[c] != exp_val)
            sprintf(str_buff, "C failed. (index: %u)", c);
  #endif
  #if TEST_ASM
        if(Y_asm[c] != exp_val)
            sprintf(str_buff, "ASM failed. (index: %u)", c);
  #endif
#endif

#if TEST_C
        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_c[c], str_buff);
#endif
#if TEST_ASM
        TEST_ASSERT_EQUAL_MESSAGE(exp_val, Y_asm[c], str_buff);
#endif


    }

}
#undef ceil_C_out
#undef C_in
#undef C_out
#undef DEBUG_ON