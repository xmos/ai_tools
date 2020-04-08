
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "nn_types.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "unity.h"


#if USE_ASM(conv2d_shallowin_deepout_block)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (1)
#endif

#define TEST_ASM ((HAS_ASM) && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)






///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*

Test cases for a 1x1 Kernel in a 1x1 image, using minimum input and output
channels

*/
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
                        //top, left, rows, cols
#define REGION          {0,    0,    1,    1}
void test_conv2d_shallowin_deepout_1x1()
{
    const int8_t zero_point = 0;
    const unsigned Y_height = X_height;
    const unsigned Y_width = X_width;

    int8_t  WORD_ALIGNED    K[C_out][K_h][32/C_in][C_in]    = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    scales[2][C_out]           = {{ 0 }};

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height][X_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height][X_width][C_out]     = {{{ 0 }}};
#endif

    nn_conv2d_init_params_t init_params = { X_height, X_width, K_h, K_w, C_in, C_out, PADDING_SAME, zero_point };
    const nn_conv2d_region_params_t region_params = REGION;
    
    nn_conv2d_sido_params_t params;

    typedef struct {
        struct {    int8_t  scale;  int8_t  offset;               } input;
        struct {    int32_t scale;  int32_t offset;  int32_t exp; } bias;
        struct {    int8_t  scale;  int8_t  offset;               } kernel;
        struct {    int16_t scale;  int16_t offset;               } shift;
        struct {    int16_t scale;  int16_t offset;               } scale;
        struct {    int8_t  scale;  int8_t  offset;  int8_t exp;  } expected;
    } case1x1_params_t;

    case1x1_params_t casses[] = {
            //input             //bias                             //kernel           //shift            //scale                //expected

        // Vector 0
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x0000},    { 0x00,   0x00,   0}  },   //Most basic: 0
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   //0 with 1.0 scale
        {   { 0x00,  0x01},     { 0x00000000,  0x7FFFFFFF,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x0000},    { 0x00,   0x00,   0}  },   //nonzero sum with 0.0 scale
        {   { 0x00,  0x01},     { 0x00000000,  0x7FFFFFFF,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x7F,   0}  },   //positive saturating bias
        {   { 0x00,  0x01},     { 0x00000000, -0x7FFFFFFF,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,  -0x7F,   0}  },   //negative saturating bias
        {   { 0x00,  0x01},     { 0x00000000,  0x00000100,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   //Non saturating total
        {   { 0x00,  0x01},     { 0x00000100,  0x0000007F,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,   0x00,   0}  },   //VDEPTH8 rounds down when appropriate
        {   { 0x00,  0x01},     { 0x00000100,  0x00000080,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,   0x01,   0}  },   //VDEPTH8 rounds up when appropriate
        {   { 0x00,  0x01},     { 0x00000000, -0x00000081,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,  -0x01,   0}  },   //VDEPTH8 rounds down when appropriate (negative)
        {   { 0x00,  0x01},     { 0x00000000, -0x00000080,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   //VDEPTH8 rounds up when appropriate (negative)

        // Vector 10   -- Scale tests
        {   { 0x00,  0x01},     { 0x00000200,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x02,   0x00,   0}  },   //
        {   { 0x00,  0x01},     { 0x00000200,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x2000},    { 0x01,   0x00,   0}  },   // Down scale
        {   { 0x00,  0x01},     { 0x00000200,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000, -0x2000},    {-0x01,   0x00,   0}  },   // negative scale
        {   { 0x00,  0x01},     { 0x00000000,  0x00001000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0400,  0x4000},    { 0x01,   0x10,   0}  },   // non-constant scale (out = 16 * (1 + cout/16))
        {   { 0x00,  0x01},     { 0x00000000,  0x00000100,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x1FDF},    { 0x00,   0x00,   0}  },   // scale rounds down
        {   { 0x00,  0x01},     { 0x00000000,  0x00000200,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x2FEF},    { 0x00,   0x01,   0}  },   // scale rounds down 2
        {   { 0x00,  0x01},     { 0x00000000,  0x00000100,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x1FE0},    { 0x00,   0x01,   0}  },   // scale rounds up
        {   { 0x00,  0x01},     { 0x00000000,  0x00000200,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x2FF0},    { 0x00,   0x02,   0}  },   // scale rounds up 2
        
        // Vector 18 -- Shift tests
        {   { 0x00,  0x01},     { 0x00000000,  0x00004000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x40,   0}  },   // 
        {   { 0x00,  0x01},     { 0x00000000,  0x00004000,  0},    { 0x00,  0x00},     { 0x00, 0x01},     { 0x0000,  0x4000},    { 0x00,   0x20,   0}  },   // Constant shift 1   
        {   { 0x00,  0x01},     { 0x00000000,  0x00004000,  0},    { 0x00,  0x00},     { 0x00, 0x02},     { 0x0000,  0x4000},    { 0x00,   0x10,   0}  },   // Constant shift 2   
        {   { 0x00,  0x01},     { 0x00000000,  0x01000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x7F,   0}  },   //
        {   { 0x00,  0x01},     { 0x00000000,  0x01000000,  0},    { 0x00,  0x00},     { 0x00, 0x0C},     { 0x0000,  0x4000},    { 0x00,   0x10,   0}  },   // Constant shift 3   
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  9},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   1}  },   // 
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  9},    { 0x00,  0x00},     { 0x01, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x02,   0}  },   // non-constant shift 1
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000, 11},    { 0x00,  0x00},     { 0x01, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x08,   0}  },   // non-constant shift 2

        // Vector 26 -- Kernel tests
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   // 
        {   { 0x00,  0x04},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // 
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x04},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // 
        {   { 0x00,  0x08},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x08},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // 
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x04,   0}  },   // 

        // Vector 31 -- Kernel tests (input channels differ)

        {   { 0x08,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x03,   0}  },  //
        {   { 0x08,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x0C,   0}  },  //
        {   { 0x20,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x30,   0}  },  //

        // Vector 34 -- Kernel tests (output channels differ)

        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x04,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,   0x00,   0}  },   //
        {   { 0x00,  0x20},     { 0x00000000,  0x00000000,  0},    { 0x04,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x02,   0x00,   0}  },   //
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x04,  0x20},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,   0x08,   0}  },   //
        {   { 0x00,  0x20},     { 0x00000000,  0x00000000,  0},    { 0x04, -0x20},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x02,  -0x10,   0}  },   //

        // Vector 38 -- Kernel tests (input channels and output channels differ)

        {   { 0x10,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x08,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x03,   0x00,   0}  },   //

    };

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("%s...\n", __func__);

    for(int v = START_ON_CASE; v < casse_count && v < STOP_ON_CASE; v++){
        PRINTF("\tVector %u...\n", v);

        for(int p = 0; p < 2; p++){

            PRINTF("\t\tPadding mode: %s...\n", (p == PADDING_VALID)? "PADDING_VALID" : "PADDING_SAME");

            case1x1_params_t* casse = &casses[v];

            init_params.pad_mode = (padding_mode_t) p;

            //Set biases, shifts and scales
            for(int cout = 0; cout < C_out; cout++){
                
                B[cout]      = casse->bias.scale * cout + casse->bias.offset;
                if( casse->bias.exp != 0)
                    B[cout] += (1 << (cout + casse->bias.exp));

                scales[0][cout] = casse->shift.scale * cout + casse->shift.offset;
                scales[1][cout] = casse->scale.scale * cout + casse->scale.offset;
            }

            //Set kernel
            memset(K, 0, sizeof(K));
            for(int cout = 0; cout < C_out; cout++){

                int8_t value = casse->kernel.scale * cout + casse->kernel.offset;

                for(int krow = 0; krow < K_h; krow++){
                    for(int kcol = 0; kcol < K_w; kcol++){
                        for(int cin = 0; cin < C_in; cin++){
                            K[cout][krow][kcol][cin] = value;
                        }
                    }
                }
            }

            //Set input image
            for(int cin = 0; cin < C_in; cin++){
                
                int8_t value = casse->input.scale * cin + casse->input.offset;

                for(int xr = 0; xr < X_height; xr++){
                    for(int xc = 0; xc < X_width; xc++){
                        X[xr][xc][cin] = value;
                    }
                }
            }

            conv2d_sido_boggle_K((int8_t*) K, K_h, VPU_INT8_EPV / C_in, C_in, C_out);
            conv2d_boggle_B(B, C_out);
            conv2d_boggle_shift_scale((int16_t*)scales, C_out, NULL);
            conv2d_shallowin_deepout_init(&params, &init_params, &region_params, (int8_t*) K, (data16_t*) B);


            //There should always be exactly one block in this test.  
            TEST_ASSERT_EQUAL(1, params.block_count);

            //Perform the actual convolution(s)   (run both C and ASM before checking either)
    #if TEST_C
            memset(Y_c, 0xCC, sizeof(Y_c));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t* blk = &params.blocks[block];
                conv2d_shallowin_deepout_block_c(  (int8_t*)Y_c,     &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
    #endif

    #if TEST_ASM
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t* blk = &params.blocks[block];
                conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm, &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
    #endif


            char str_buff[2000];

            //Check that all outputs are what they should be
            for(int co = 0; co < C_out; co++){
                
                int32_t exp_out32 = casse->expected.scale * ((int32_t)co) + casse->expected.offset;
                if(casse->expected.exp != 0)
                    exp_out32 += (1 << (co + casse->expected.exp));

                int8_t exp_out = (exp_out32 > VPU_INT8_MAX)?    0x7F
                               : (exp_out32 < VPU_INT8_MIN)?   -0x7F
                               : exp_out32;
                // int8_t exp_out = casse->exp.scale * co + casse->exp.offset;
                
                for(int row = 0; row < Y_height; row++){
                    for(int col = 0; col < Y_width; col++){
    #if TEST_C 
                        int8_t c_val = Y_c[row][col][co];
    #endif
    #if TEST_ASM
                        int8_t asm_val = Y_asm[row][col][co];
    #endif

    #if (TEST_C && TEST_ASM)
                        //First thing to check is whether they match one another (if both are being tested)
                        //  Also report the actual values, so we know which (if either) is correct
                        if(c_val != asm_val){
                            sprintf(str_buff, 
                                "     C and ASM implementations gave different results for Y[%u][%u][%u] on vector %u. C: %d      ASM: %d    Expected: %d", 
                                row, col, co, v, c_val, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(c_val, asm_val, str_buff);
    #endif

    #if TEST_C
                        if(c_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "      C failed.  Y_c[%u][%u][%u] = %d. Expected %d.", row, col, co, c_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, c_val, str_buff);
    #endif
    #if TEST_ASM
                        if(asm_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "       ASM failed.  Y_asm[%u][%u][%u] = %d. Expected %d.", row, col, co, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, asm_val, str_buff);
    #endif
                    }
                }
            }


            conv2d_shallowin_deepout_deinit(&params);
        }
    }
}
#undef REGION
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON

















///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*

1x1 Kernel in 1x1 image with more than minimum channel counts

These tests mostly just want to make sure that the function is iterating
over all the input and output channel groups.

*/
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (4 * VPU_INT8_ACC_PERIOD )
#define C_in            (4)
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
                        //top, left, rows, cols
#define REGION          {0,    0,    1,    1}
void test_conv2d_shallowin_deepout_1x1_chans()
{
    const int8_t zero_point = 0;
    const unsigned Y_height = X_height;
    const unsigned Y_width = X_width;

    int8_t  WORD_ALIGNED    K[C_out][K_h][32/C_in][C_in]    = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    scales[2][C_out]           = {{ 0 }};

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height][X_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height][X_width][C_out]     = {{{ 0 }}};
#endif

    nn_conv2d_init_params_t init_params = { X_height, X_width, K_h, K_w, C_in, C_out, PADDING_SAME, zero_point };
    const nn_conv2d_region_params_t region_params = REGION;
    
    nn_conv2d_sido_params_t params;

    typedef struct {
        struct {    int8_t  scale;  int8_t  offset;               } input;
        struct {    int32_t scale;  int32_t offset;  int32_t exp; } bias;
        struct {    int32_t scale;  int32_t offset;               } kernel;
        struct {    int16_t scale;  int16_t offset;               } shift;
        struct {    int16_t scale;  int16_t offset;               } scale;
        struct {    int8_t  scale;  int8_t  offset;  int8_t exp;  } expected;
    } case1x1_params_t;

    case1x1_params_t casses[] = {
            //input             //bias                             //kernel           //shift            //scale                //expected

        // Vector 0
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x0000},    { 0x00,   0x00,   0}  },   //
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   //
        {   { 0x00,  0x01},     { 0x00000000,  0x00000100,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x0000},    { 0x00,   0x00,   0}  },   //
        {   { 0x00,  0x01},     { 0x00000000,  0x00000100,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // All output channels expect nonzero values
        {   { 0x00,  0x01},     { 0x00000100,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,   0x00,   0}  },   // Each output channel expects a different value

        // Vector 5 -- Scales
        {   { 0x00,  0x01},     { 0x00000200,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x02,   0x00,   0}  },   //
        {   { 0x00,  0x01},     { 0x00000200,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x2000},    { 0x01,   0x00,   0}  },   // Down scale
        {   { 0x00,  0x01},     { 0x00000200,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000, -0x2000},    {-0x01,   0x00,   0}  },   // negative scale
        {   { 0x00,  0x01},     { 0x00000000,  0x00004000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0100,  0x4000},    { 0x01,   0x40,   0}  },   // non-constant scale (out = 64 * (1 + cout/64))

        // Vector 9 -- Shift tests
        {   { 0x00,  0x01},     { 0x00000000,  0x00004000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x40,   0}  },   // 
        {   { 0x00,  0x01},     { 0x00000000,  0x00004000,  0},    { 0x00,  0x00},     { 0x00, 0x01},     { 0x0000,  0x4000},    { 0x00,   0x20,   0}  },   // Constant shift 1   
        {   { 0x00,  0x01},     { 0x00000000,  0x00004000,  0},    { 0x00,  0x00},     { 0x00, 0x02},     { 0x0000,  0x4000},    { 0x00,   0x10,   0}  },   // Constant shift 2   
        {   { 0x00,  0x01},     { 0x00000000,  0x01000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x7F,   0}  },   //
        {   { 0x00,  0x01},     { 0x00000000,  0x01000000,  0},    { 0x00,  0x00},     { 0x00, 0x0C},     { 0x0000,  0x4000},    { 0x00,   0x10,   0}  },   // Constant shift 3   
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  9},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   1}  },   // 
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  9},    { 0x00,  0x00},     { 0x01, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x02,   0}  },   // non-constant shift 1
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000, 11},    { 0x00,  0x00},     { 0x01, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x08,   0}  },   // non-constant shift 2

        // Vector 17 -- Kernel tests  (output channels same, input channels same)
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   // 
        {   { 0x00,  0x04},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // 
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x04},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // 
        {   { 0x00,  0x08},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x08},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // 
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x04,   0}  },   // 

        // Vector 22 -- Kernel tests (input channels differ)        
        {   { 0x08,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x03,   0}  },  //
        {   { 0x08,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x0C,   0}  },  //
        {   { 0x20,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x30,   0}  },  //

        // Vector 25 -- Kernel tests (output channels differ)
        {   { 0x00,  0x20},     { 0x00000000,  0x00000000,  0},    { 0x02,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,   0x00,   0}  },   //
        {   { 0x00,  0x40},     { 0x00000000,  0x00000000,  0},    { 0x02,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x02,   0x00,   0}  },   //
        {   { 0x00,  0x40},     { 0x00000000,  0x00000000,  0},    { 0x01, -0x20},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,  -0x20,   0}  },   //
        {   { 0x00,  0x40},     { 0x00000000,  0x00000000,  0},    {-0x01,  0x20},     { 0x00, 0x00},     { 0x0000,  0x4000},    {-0x01,   0x20,   0}  },   //
        {   { 0x00, -0x80},     { 0x00000000,  0x00000000,  0},    {-0x02,  0x40},     { 0x00, 0x02},     { 0x0000,  0x4000},    { 0x01,  -0x20,   0}  },   //

        // Vector 30 -- Kernel tests (input/output both differ)
        {   { 0x40, -0x80},     { 0x00000000,  0x00000000,  0},    {-0x02,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x01,   0x00,   0}  },   //
        
    };

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("%s...\n", __func__);

    for(int v = START_ON_CASE; v < casse_count && v < STOP_ON_CASE; v++){
        PRINTF("\tVector %u...\n", v);

        for(int p = 1; p >= 0; p--){

            PRINTF("\t\tPadding mode: %s...\n", (p == PADDING_VALID)? "PADDING_VALID" : "PADDING_SAME");

            case1x1_params_t* casse = &casses[v];

            init_params.pad_mode = (padding_mode_t) p;

            //Set biases, shifts and scales
            for(int cout = 0; cout < C_out; cout++){
                
                B[cout]      = casse->bias.scale * cout + casse->bias.offset;
                if( casse->bias.exp != 0)
                    B[cout] += (1 << (cout/4 + casse->bias.exp));

                scales[0][cout] = casse->shift.scale * cout/4 + casse->shift.offset;
                scales[1][cout] = casse->scale.scale * cout + casse->scale.offset;
            }

            //Set kernel
            memset(K, 0, sizeof(K));
            for(int cout = 0; cout < C_out; cout++){

                int8_t value = casse->kernel.scale * cout + casse->kernel.offset;

                for(int krow = 0; krow < K_h; krow++){
                    for(int kcol = 0; kcol < K_w; kcol++){
                        for(int cin = 0; cin < C_in; cin++){
                            K[cout][krow][kcol][cin] = value;
                        }
                    }
                }
            }
            
            //Set input image
            for(int cin = 0; cin < C_in; cin++){
                
                int8_t value = (casse->input.scale * cin + casse->input.offset);

                for(int xr = 0; xr < X_height; xr++){
                    for(int xc = 0; xc < X_width; xc++){
                        X[xr][xc][cin] = value;
                    }
                }
            }
            
            conv2d_sido_boggle_K((int8_t*) K, K_h, 32/C_in, C_in, C_out);
            conv2d_boggle_B(B, C_out);
            conv2d_boggle_shift_scale((int16_t*)scales, C_out, NULL);
            conv2d_shallowin_deepout_init(&params, &init_params, &region_params, (int8_t*) K, (data16_t*) B);


            //There should always be exactly one block in this test.  
            TEST_ASSERT_EQUAL(1, params.block_count);

            //Perform the actual convolution(s)   (run both C and ASM before checking either)
    #if TEST_C
            memset(Y_c, 0xCC, sizeof(Y_c));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t* blk = &params.blocks[block];
                conv2d_shallowin_deepout_block_c(  (int8_t*)Y_c,     &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
    #endif

    #if TEST_ASM
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t* blk = &params.blocks[block];
                conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm, &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
    #endif


            char str_buff[2000];

            //Check that all outputs are what they should be
            for(int co = 0; co < C_out; co++){
                
                int32_t exp_out32 = casse->expected.scale * ((int32_t)co) + casse->expected.offset;
                if(casse->expected.exp != 0)
                    exp_out32 += (1 << (co/4 + casse->expected.exp));

                int8_t exp_out = (exp_out32 > VPU_INT8_MAX)?    0x7F
                               : (exp_out32 < VPU_INT8_MIN)?   -0x7F
                               : exp_out32;
                // int8_t exp_out = casse->exp.scale * co + casse->exp.offset;
                
                for(int row = 0; row < Y_height; row++){
                    for(int col = 0; col < Y_width; col++){
    #if TEST_C 
                        int8_t c_val = Y_c[row][col][co];
    #endif
    #if TEST_ASM
                        int8_t asm_val = Y_asm[row][col][co];
    #endif

    #if (TEST_C && TEST_ASM)
                        //First thing to check is whether they match one another (if both are being tested)
                        //  Also report the actual values, so we know which (if either) is correct
                        if(c_val != asm_val){
                            sprintf(str_buff, 
                                "     C and ASM implementations gave different results for Y[%u][%u][%u] on vector %u. C: %d      ASM: %d    Expected: %d", 
                                row, col, co, v, c_val, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(c_val, asm_val, str_buff);
    #endif

    #if TEST_C
                        if(c_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "      C failed.  Y_c[%u][%u][%u] = %d. Expected %d.", row, col, co, c_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, c_val, str_buff);
    #endif
    #if TEST_ASM
                        if(asm_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "       ASM failed.  Y_asm[%u][%u][%u] = %d. Expected %d.", row, col, co, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, asm_val, str_buff);
    #endif
                    }
                }
            }


            conv2d_shallowin_deepout_deinit(&params);
        }
    }
}
#undef REGION
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON
















///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*

Test cases for a 1x1 Kernel and larger images (up to 3x3)



*/
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (1)
#define K_w             (1)
#define X_height_max    (3)
#define X_width_max     (3)
void test_conv2d_shallowin_deepout_1x1_xsize()
{
    const int8_t zero_point = 12;

    int8_t  WORD_ALIGNED    K[C_out][K_h][32/C_in][C_in]        = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height_max][X_width_max][C_in]  = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                            = { 0 };
    int16_t WORD_ALIGNED    scales[2][C_out]           = {{ 0 }};

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height_max][X_width_max][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height_max][X_width_max][C_out]     = {{{ 0 }}};
#endif

    
    nn_conv2d_sido_params_t params;

    typedef struct {
        
        int8_t X[X_height_max][X_width_max];

        struct {    unsigned X_height; unsigned X_width;  } input;
        struct {    int32_t scale;     int32_t offset;    } bias;
        struct {    int8_t  scale;     int8_t  offset;    } kernel;
        struct {    int16_t scale;     int16_t offset;    } shift;
        struct {    int16_t scale;     int16_t offset;    } scale;

        int8_t expected[X_height_max][X_width_max];

    } case1x1_params_t;

    int8_t OxXX = 0xCC;

    case1x1_params_t casses[] = {
            // X            

        {  {{ 0,  0,  0},
            { 0,  0,  0},   // X Dims   //bias                         //kernel           //shift            //scale                //expected
            { 0,  0,  0}},  {1, 1},     { 0x00000000,  0x00000000},    { 0x00,  0x00},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x00,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },

        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {1, 1},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {1, 2},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x03, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {2, 1},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  OxXX, OxXX},
                                                                                                                                     { 0x03,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {1, 3},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x03, 0x03},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {2, 2},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x03, OxXX},
                                                                                                                                     { 0x03,  0x03, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {2, 3},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x03, 0x03},
                                                                                                                                     { 0x03,  0x03, 0x03},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {3, 2},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x03, OxXX},
                                                                                                                                     { 0x03,  0x03, OxXX},
                                                                                                                                     { 0x03,  0x03, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {3, 3},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x03, 0x03},
                                                                                                                                     { 0x03,  0x03, 0x03},
                                                                                                                                     { 0x03,  0x03, 0x03}} },
        // Vector 9
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {1, 1},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {1, 3},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x05, 0x07},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {3, 1},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  OxXX, OxXX},
                                                                                                                                     { 0x09,  OxXX, OxXX},
                                                                                                                                     { 0x0F,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {2, 2},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x05, OxXX},
                                                                                                                                     { 0x09,  0x0B, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {3, 3},     { 0x00000000,  0x00000000},    { 0x00, -0x80},    { 0x00, 0x00},     { 0x0000, -0x4000},    {{ 0x03,  0x05, 0x07},
                                                                                                                                     { 0x09,  0x0B, 0x0D},
                                                                                                                                     { 0x0F,  0x11, 0x13}} },



    };

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("%s...\n", __func__);

    for(int v = START_ON_CASE; v < casse_count && v < STOP_ON_CASE; v++){
        PRINTF("\tVector %u...\n", v);
        
        case1x1_params_t* casse = &casses[v];

        for(int p = 0; p < 2; p++){

            PRINTF("\t\tPadding mode: %s...\n", (p == PADDING_VALID)? "PADDING_VALID" : "PADDING_SAME");

            int X_height = casse->input.X_height;
            int X_width  = casse->input.X_width;


            const unsigned Y_height = X_height;
            const unsigned Y_width = X_width;

            nn_conv2d_init_params_t init_params = { X_height, X_width, K_h, K_w, C_in, C_out, (padding_mode_t) p, zero_point };
            

            //Set input image
            //  
            // Pixels for test case will be numbered 0 to X_height * X_width - 1, row by row
            // Channel k's value for each pixel will be  pxl_dex * k
            for(int xr = 0; xr < X_height; xr++){
                for(int xc = 0; xc < X_width; xc++){
                    
                    unsigned pxl_index = xr * X_width + xc;
                    unsigned base_val = casse->X[xr][xc];
                    
                    for(int cin = 0; cin < C_in; cin++){
                        unsigned chan_index = pxl_index * C_in + cin;
                        ((int8_t*)X)[chan_index] = base_val + cin;
                    }
                }
            }

            //Set biases, shifts and scales
            for(int cout = 0; cout < C_out; cout++){
                
                B[cout]      = casse->bias.scale * cout + casse->bias.offset;

                scales[0][cout] = casse->shift.scale * cout + casse->shift.offset;
                scales[1][cout] = casse->scale.scale * cout + casse->scale.offset;
            }

            //Set kernel
            for(int cout = 0; cout < C_out; cout++){

                int8_t value = casse->kernel.scale * cout + casse->kernel.offset;

                // PRINTF("K[%d] -> %d\n", cout, value);
                // PRINTF("B[%d] -> %d\n", cout, B[cout]);
                memset(&K[cout], value, sizeof(int8_t) * K_h * K_w * C_in);

                // for(int krow = 0; krow < K_h; krow++){
                //     for(int kcol = 0; kcol < K_w; kcol++){
                //         for(int cin = 0; cin < C_in; cin++){
                //             K[cout][krow][kcol][cin] = value;
                //         }
                //     }
                // }
            }
            
            conv2d_sido_boggle_K((int8_t*) K, K_h, VPU_INT8_EPV/C_in, C_in, C_out);
            conv2d_boggle_B(B, C_out);
            conv2d_boggle_shift_scale((int16_t*)scales, C_out, NULL);
            conv2d_shallowin_deepout_init(&params, &init_params, NULL, (int8_t*) K, (data16_t* ) B);


            //There should always be exactly one block in this test.  
            TEST_ASSERT_EQUAL(1, params.block_count);

            //Perform the actual convolution(s)   (run both C and ASM before checking either)
#if TEST_C
            memset(Y_c, OxXX, sizeof(Y_c));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t*  blk = &params.blocks[block];
                conv2d_shallowin_deepout_block_c(  (int8_t*)Y_c,     &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
#endif

#if TEST_ASM
            memset(Y_asm, OxXX, sizeof(Y_asm));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t*  blk = &params.blocks[block];
                conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm, &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
#endif

            char str_buff[2000];

            //Check that all outputs are what they should be
            for(int co = 0; co < C_out; co++){
                
                for(int row = 0; row < Y_height; row++){
                    for(int col = 0; col < Y_width; col++){
                        
                        int8_t exp_out = casse->expected[row][col];

                        unsigned ydex = row * Y_width * C_out + col * C_out + co;

#if TEST_C 
                        int8_t c_val = ((int8_t*)Y_c)[ydex];
#endif
#if TEST_ASM
                        int8_t asm_val = ((int8_t*)Y_asm)[ydex];
#endif

#if (TEST_C && TEST_ASM)
                        //First thing to check is whether they match one another (if both are being tested)
                        //  Also report the actual values, so we know which (if either) is correct
                        if(c_val != asm_val){
                            sprintf(str_buff, 
                                "     C and ASM implementations gave different results for Y[%u][%u][%u] on vector %u. C: %d      ASM: %d    Expected: %d", 
                                row, col, co, v, c_val, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(c_val, asm_val, str_buff);
#endif

#if TEST_C
                        if(c_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "      C failed.  Y_c[%u][%u][%u] = %d. Expected %d.", row, col, co, c_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, c_val, str_buff);
#endif
#if TEST_ASM
                        if(asm_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "       ASM failed.  Y_asm[%u][%u][%u] = %d. Expected %d.", row, col, co, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, asm_val, str_buff);
#endif
                    }
                }
            }


            //Check that everything else is OxXX

            for(int i = Y_height * Y_width * C_out; i < X_height_max * X_width_max * C_out; i++){

#if TEST_C
                TEST_ASSERT_EQUAL(OxXX, ((int8_t*)Y_c)[i]);
#endif
#if TEST_ASM
                TEST_ASSERT_EQUAL(OxXX, ((int8_t*)Y_asm)[i]);
#endif
            }


            conv2d_shallowin_deepout_deinit(&params);
        }
    }
}
#undef REGION
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON
















///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*

Test cases for a 1x1 Kernel and larger images (up to 3x3)



*/
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (3)
#define K_w             (3)
#define X_height        (3)
#define X_width         (3)
void test_conv2d_shallowin_deepout_3x3()
{
    const unsigned Y_height = X_height;
    const unsigned Y_width = X_width;

    int8_t  WORD_ALIGNED    K[C_out][K_h][32/C_in][C_in]    = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    scales[2][C_out]           = {{ 0 }};

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height][X_width][C_out]   = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height][X_width][C_out] = {{{ 0 }}};
#endif

    
    nn_conv2d_sido_params_t params;

    typedef struct {
        
        int8_t X_row1[X_width];

        unsigned incr_X;

        int32_t bias;

        int8_t K_row1[K_w];

        int16_t shift;

        int8_t zero_point;

        int8_t Y_row1[X_width];

        int8_t X_row2[X_width];
        int8_t K_row2[K_w];
        int8_t Y_row2[X_width];

        int8_t X_row3[X_width];
        int8_t K_row3[K_w];
        int8_t Y_row3[X_width];

    } case1x1_params_t;

    case1x1_params_t casses[] = {

            // X                 //incr X   // bias       // K                       //shift    // zero     // Y
        {   {  0x00,  0x00,  0x00}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    0,         0x00,       {  0x00,  0x00,  0x00},
            {  0x00,  0x00,  0x00},                       {  0x00,  0x00,  0x00},                           {  0x00,  0x00,  0x00},
            {  0x00,  0x00,  0x00},                       {  0x00,  0x00,  0x00},                           {  0x00,  0x00,  0x00}},
            
        {   {  0x00,  0x00,  0x00}, 0,      0x00001300,   {  0x00,  0x00,  0x00},    0,         0x00,       {  0x13,  0x13,  0x13},
            {  0x00,  0x00,  0x00},                       {  0x00,  0x00,  0x00},                           {  0x13,  0x13,  0x13},
            {  0x00,  0x00,  0x00},                       {  0x00,  0x00,  0x00},                           {  0x13,  0x13,  0x13}},

        {   {  0x40,  0x40,  0x40}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    0,         0x00,       {  0x00,  0x00,  0x00},
            {  0x40,  0x40,  0x40},                       {  0x00,  0x00,  0x00},                           {  0x00,  0x00,  0x00},
            {  0x40,  0x40,  0x40},                       {  0x00,  0x00,  0x00},                           {  0x00,  0x00,  0x00}},
        
        {   {  0x40,  0x40,  0x40}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    0,         0x40,       {  0x00,  0x00,  0x00},
            {  0x40,  0x40,  0x40},                       {  0x00,  0x00,  0x00},                           {  0x00,  0x00,  0x00},
            {  0x40,  0x40,  0x40},                       {  0x00,  0x00,  0x00},                           {  0x00,  0x00,  0x00}},
        
        {   {  0x00,  0x00,  0x00}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    0,         0x40,       {  0x00,  0x00,  0x00},
            {  0x00,  0x00,  0x00},                       {  0x00,  0x40,  0x00},                           {  0x00,  0x00,  0x00},
            {  0x00,  0x00,  0x00},                       {  0x00,  0x00,  0x00},                           {  0x00,  0x00,  0x00}},
        
        // Vector 5
        {   {  0x01,  0x02,  0x03}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    0,         0x00,       {  0x01,  0x02,  0x03},
            {  0x04,  0x05,  0x06},                       {  0x00,  0x40,  0x00},                           {  0x04,  0x05,  0x06},
            {  0x07,  0x08,  0x09},                       {  0x00,  0x00,  0x00},                           {  0x07,  0x08,  0x09}},
            
        {   {  0x08,  0x10,  0x18}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    3,         0x00,       {  0x01,  0x02,  0x03},
            {  0x20,  0x28,  0x30},                       {  0x00,  0x40,  0x00},                           {  0x04,  0x05,  0x06},
            {  0x38,  0x40,  0x48},                       {  0x00,  0x00,  0x00},                           {  0x07,  0x08,  0x09}},
            
        {   {  0x01,  0x02,  0x03}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    0,         0x00,       {  0x03,  0x05,  0x03},
            {  0x04,  0x05,  0x06},                       {  0x00,  0x40,  0x40},                           {  0x09,  0x0B,  0x06},
            {  0x07,  0x08,  0x09},                       {  0x00,  0x00,  0x00},                           {  0x0F,  0x11,  0x09}},
            
        {   {  0x01,  0x02,  0x03}, 0,      0x00000000,   {  0x00,  0x00,  0x00},    0,         0x00,       {  0x03,  0x06,  0x05},
            {  0x04,  0x05,  0x06},                       {  0x40,  0x40,  0x40},                           {  0x09,  0x0F,  0x0B},
            {  0x07,  0x08,  0x09},                       {  0x00,  0x00,  0x00},                           {  0x0F,  0x18,  0x11}},
            
        {   {  0x01,  0x02,  0x03}, 0,      0x00000000,   {  0x00,  0x40,  0x00},    0,         0x00,       {  0x07,  0x0B,  0x0B},
            {  0x04,  0x05,  0x06},                       {  0x40,  0x40,  0x40},                           {  0x11,  0x19,  0x17},
            {  0x07,  0x08,  0x09},                       {  0x00,  0x40,  0x00},                           {  0x13,  0x1D,  0x17}},
        
        // Vector 10    
        {   {  0x01,  0x02,  0x03}, 0,      0x00000000,   {  0x40,  0x40,  0x40},    0,         0x00,       {  0x0C,  0x15,  0x10},
            {  0x04,  0x05,  0x06},                       {  0x40,  0x40,  0x40},                           {  0x1B,  0x2D,  0x21},
            {  0x07,  0x08,  0x09},                       {  0x40,  0x40,  0x40},                           {  0x18,  0x27,  0x1C}},
        
        {   { -0x08, -0x08, -0x08}, 0,      0x00004800,   {  0x40,  0x40,  0x40},    3,         0x00,       {  0x05,  0x03,  0x05},
            { -0x08, -0x08, -0x08},                       {  0x40,  0x40,  0x40},                           {  0x03,  0x00,  0x03},
            { -0x08, -0x08, -0x08},                       {  0x40,  0x40,  0x40},                           {  0x05,  0x03,  0x05}},
            
        {   {  0x00,  0x00,  0x00}, 0,      0x00000000,   {  0x40,  0x40,  0x40},    3,         0x08,       {  0x05,  0x03,  0x05},
            {  0x00,  0x00,  0x00},                       {  0x40,  0x40,  0x40},                           {  0x03,  0x00,  0x03},
            {  0x00,  0x00,  0x00},                       {  0x40,  0x40,  0x40},                           {  0x05,  0x03,  0x05}},
            
        {   {  0x01,  0x02,  0x03}, 0,      0x00000000,   {  0x40,  0x40,  0x40},    0,         0x10,       {  0x5C,  0x45,  0x60},
            {  0x04,  0x05,  0x06},                       {  0x40,  0x40,  0x40},                           {  0x4B,  0x2D,  0x51},
            {  0x07,  0x08,  0x09},                       {  0x40,  0x40,  0x40},                           {  0x68,  0x57,  0x6C}},

        {   {  0x01,  0x02,  0x03}, 0,     -0x00008000,   {  0x08,  0x10,  0x18},    0,         0x01,       { -0x72, -0x6C, -0x70},
            {  0x04,  0x05,  0x06},                       {  0x20,  0x28,  0x30},                           { -0x67, -0x5C, -0x66},
            {  0x07,  0x08,  0x09},                       {  0x38,  0x40,  0x48},                           { -0x6F, -0x6A, -0x70}},

        // Vector 15
        {   {  0x01,  0x02,  0x03}, 1,      0x00000000,   {  0x01,  0x02,  0x03},    0,         0x04,       {  0x03,  0x04,  0x04},
            {  0x04,  0x05,  0x06},                       {  0x04,  0x05,  0x06},                           {  0x04,  0x06,  0x05},
            {  0x07,  0x08,  0x09},                       {  0x07,  0x08,  0x09},                           {  0x04,  0x04,  0x04}},
            
    };

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("%s...\n", __func__);

    for(int v = START_ON_CASE; v < casse_count && v < STOP_ON_CASE; v++){
        PRINTF("\tVector %u...\n", v);
        
        case1x1_params_t* casse = &casses[v];
        
        int8_t*  casse_X[X_height] = {(int8_t* ) &casse->X_row1,(int8_t* )  &casse->X_row2,(int8_t* )  &casse->X_row3};
        int8_t*  casse_K[K_h]      = {(int8_t* ) &casse->K_row1,(int8_t* )  &casse->K_row2,(int8_t* )  &casse->K_row3};
        int8_t*  casse_Y[X_height] = {(int8_t* ) &casse->Y_row1,(int8_t* )  &casse->Y_row2,(int8_t* )  &casse->Y_row3};
        
        padding_mode_t pmodes[] = {PADDING_SAME, PADDING_VALID};
        for(int p = 0; p < sizeof(pmodes)/sizeof(padding_mode_t); p++){

            PRINTF("\t\tPadding mode: %s...\n", (pmodes[p] == PADDING_VALID)? "PADDING_VALID" : "PADDING_SAME");
            
            nn_conv2d_init_params_t init_params = { X_height, X_width, K_h, K_w, C_in, C_out, pmodes[p], casse->zero_point };

            //Set input image
            for(int xr = 0; xr < X_height; xr++){
                for(int xc = 0; xc < X_width; xc++){
                    for(int cin = 0; cin < C_in; cin++){
                        X[xr][xc][cin] = casse_X[xr][xc] + cin * casse->incr_X;
                    }
                }
            }

#if DEBUG_ON
            {
                unsigned debug_chan_in = 0;
                PRINTF("X[:,:,%u] =", debug_chan_in);
                for(int xr = 0; xr < X_height; xr++){
                    PRINTF(xr? "\t\t" : "\t");
                    for(int xc = 0; xc < X_width; xc++){
                        PRINTF("%d  ", X[xr][xc][debug_chan_in]);
                    }
                    PRINTF("\n");
                }
                PRINTF("\n");
            }
#endif

            //Set biases, shifts and scales
            for(int cout = 0; cout < C_out; cout++){
                B[cout]      = casse->bias;
                scales[0][cout] = casse->shift;
                scales[1][cout] = 0x4000;
            }

#if DEBUG_ON
            unsigned debug_chan_out = 0;
            PRINTF("B[%u] = %ld\n", debug_chan_out, B[debug_chan_out]);
            PRINTF("shifts[%u] = %d\n", debug_chan_out, shifts[debug_chan_out]);
            PRINTF("scales[%u] = 0x%04X\n\n", debug_chan_out, scales[debug_chan_out]);
#endif

            //Set kernel
            memset(K, 0, sizeof(K));
            for(int cout = 0; cout < C_out; cout++){
                for(int krow = 0; krow < K_h; krow++){
                    for(int kcol = 0; kcol < K_w; kcol++){
                        for(int cin = 0; cin < C_in; cin++){
                            K[cout][krow][kcol][cin] = casse_K[krow][kcol];
                        }
                    }
                }
            }

#if DEBUG_ON
            {
                PRINTF("K[%u,:,:,0] =", debug_chan_out);
                for(int kr = 0; kr < K_h; kr++){
                    PRINTF(kr? "\t\t" : "\t");
                    for(int kc = 0; kc < K_h; kc++){
                        PRINTF("%d  ", K[debug_chan_out][kr][kc][0]);
                    }
                    PRINTF("\n");
                }
                PRINTF("\n");
            }
#endif
            
            conv2d_sido_boggle_K((int8_t*) K, K_h, VPU_INT8_EPV/C_in, C_in, C_out);
            conv2d_boggle_B(B, C_out);
            conv2d_boggle_shift_scale((int16_t*)scales, C_out, NULL);
            conv2d_shallowin_deepout_init(&params, &init_params, NULL, (int8_t*) K, (data16_t* ) B);

            //Padding mode SAME should have 1 block, mode VALID should have K_h*K_w
            TEST_ASSERT_EQUAL_MESSAGE((init_params.pad_mode == PADDING_VALID)? 1 : K_h*K_w, params.block_count, "Wrong number of convolution blocks.");

            //Perform the actual convolution(s)   (run both C and ASM before checking either)
#if TEST_C
            memset(Y_c, 0xCC, sizeof(Y_c));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t*  blk = &params.blocks[block];
                int8_t* Y_targ = (init_params.pad_mode == PADDING_SAME)? (int8_t*)Y_c : (int8_t*) &Y_c[X_height>>1][X_width>>1];
                conv2d_shallowin_deepout_block_c(  Y_targ,     &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
#endif

#if TEST_ASM
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_sido_block_params_t*  blk = &params.blocks[block];
                int8_t* Y_targ = (init_params.pad_mode == PADDING_SAME)? (int8_t*)Y_asm : (int8_t*) &Y_asm[X_height>>1][X_width>>1];
                conv2d_shallowin_deepout_block_asm(Y_targ, &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
            }
#endif


#if DEBUG_ON
                {
                    PRINTF("Y_exp[:,:] =");
                    for(int yr = 0; yr < Y_height; yr++){
                        PRINTF(yr? "\t\t" : "\t");
                        for(int yc = 0; yc < Y_width; yc++){
                            int8_t exp = casse_Y[yr][yc];
                            if( (init_params.pad_mode == PADDING_VALID) 
                                && ((yr != (X_height>>1)) 
                                 || (yc != (X_width>>1))))
                                exp = 0xCC;
                            PRINTF("%d  ", exp);
                        }
                        PRINTF("\n");
                    }
                    PRINTF("\n");
                    
                    PRINTF("Y_act[:,:] =");
                    for(int yr = 0; yr < Y_height; yr++){
                        PRINTF(yr? "\t\t" : "\t");
                        for(int yc = 0; yc < Y_width; yc++){
                            PRINTF("%d  ", Y_c[yr][yc][debug_chan_out]);
                        }
                        PRINTF("\n");
                    }
                    PRINTF("\n");
                }
#endif

            char str_buff[2000];

            //Check that all outputs are what they should be

            for(int co = 0; co < C_out; co++){
                
                for(int row = 0; row < Y_height; row++){
                    for(int col = 0; col < Y_width; col++){
                        
                        int8_t exp_out = casse_Y[row][col];

                        if(init_params.pad_mode == PADDING_VALID){
                            if(row != (X_height>>1) || col != (X_width>>1))
                                exp_out = 0xCC;
                        }


                        unsigned ydex = row * Y_width * C_out + col * C_out + co;

#if TEST_C 
                        int8_t c_val = Y_c[row][col][co];
#endif
#if TEST_ASM
                        int8_t asm_val = Y_asm[row][col][co];
#endif

#if (TEST_C && TEST_ASM)
                        //First thing to check is whether they match one another (if both are being tested)
                        //  Also report the actual values, so we know which (if either) is correct
                        if(c_val != asm_val){
                            sprintf(str_buff, 
                                "     C and ASM implementations gave different results for Y[%u][%u][%u] on vector %u. C: %d      ASM: %d    Expected: %d", 
                                row, col, co, v, c_val, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(c_val, asm_val, str_buff);
#endif

#if TEST_C
                        if(c_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "      C failed.  Y_c[%u][%u][%u] = %d. Expected %d.", row, col, co, c_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, c_val, str_buff);
#endif
#if TEST_ASM
                        if(asm_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                            sprintf(str_buff, "       ASM failed.  Y_asm[%u][%u][%u] = %d. Expected %d.", row, col, co, asm_val, exp_out);
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(exp_out, asm_val, str_buff);
#endif
                    }
                }
            }

            conv2d_shallowin_deepout_deinit(&params);
        }
    }
}
#undef REGION
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON

















///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*

Test cases for limiting convolutions to specific regions of the output image



*/
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( 3 * VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (5)
#define K_w             (5)
#define X_height        (8)
#define X_width         (8)
void test_conv2d_shallowin_deepout_regions()
{
    const unsigned Y_height = X_height;
    const unsigned Y_width = X_width;

    int8_t  WORD_ALIGNED    K[C_out][K_h][VPU_INT8_EPV/C_in][C_in]        = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    scales[2][C_out]           = {{ 0 }};

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height][X_width][C_out]   = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height][X_width][C_out] = {{{ 0 }}};
#endif

    nn_conv2d_sido_params_t params;

    typedef struct {
        unsigned top;
        unsigned left;
        unsigned rows;
        unsigned cols;
    } case1x1_params_t;

    case1x1_params_t casses[] = {

            //top       //left      //rows      //cols
        {   0,          0,          8,          8           },
        {   0,          0,          1,          1           },
        {   0,          0,          1,          2           },
        {   0,          0,          2,          1           },
        {   0,          0,          2,          2           },
        {   0,          0,          3,          3           },
        {   0,          0,          3,          5           },
        {   0,          0,          5,          3           },
        {   0,          0,          8,          2           },
        {   0,          0,          2,          8           },

        {   1,          1,          1,          1           },
        {   3,          2,          1,          2           },
        {   5,          5,          2,          1           },
        {   0,          6,          2,          2           },
        {   5,          5,          3,          3           },
        {   1,          2,          3,          5           },
            
            
    };

    // PRINTF("&K = 0x%08X\n", K);

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("%s...\n", __func__);

    for(int v = START_ON_CASE; v < casse_count && v < STOP_ON_CASE; v++){
        
        case1x1_params_t* casse = &casses[v];
        PRINTF("\tVector %u (%u, %u, %u, %u)...\n", v, casse->top, casse->left, casse->rows, casse->cols);
                
        nn_conv2d_init_params_t init_params = { X_height, X_width, K_h, K_w, C_in, C_out, PADDING_SAME, 0x00 };


        //Set input image
        // memset(X, 0x00, sizeof(X));

        //Set biases, shifts and scales
        for(int cout = 0; cout < C_out; cout++){
            B[cout]      = 0x0100;
            scales[0][cout] = 0;
            scales[1][cout] = 0x4000;
        }

        //Set kernel
        // memset(K, 0x00, sizeof(K));

        nn_conv2d_region_params_t reg = {casse->top, casse->left, casse->rows, casse->cols};
        
        //No need to boggle K, all values are zero.
        // conv2d_sido_boggle_K((int8_t*) K, K_h, K_w, C_in, C_out);
        conv2d_boggle_B(B, C_out);
        conv2d_boggle_shift_scale((int16_t*)scales, C_out, NULL);
        conv2d_shallowin_deepout_init(&params, &init_params, &reg, (int8_t*) K, (data16_t* ) B);

        //Perform the actual convolution(s)   (run both C and ASM before checking either)
#if TEST_C
        // PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        for(int block = 0; block < params.block_count; block++){
            // PRINTF("\t\t\tblock %d...\n", block);
            const nn_conv2d_sido_block_params_t*  blk = &params.blocks[block];
            conv2d_shallowin_deepout_block_c( (int8_t*) Y_c, &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
        }
#endif

#if TEST_ASM
        // PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        for(int block = 0; block < params.block_count; block++){
            // PRINTF("\t\t\tblock %d...\n", block);
            const nn_conv2d_sido_block_params_t*  blk = &params.blocks[block];
            conv2d_shallowin_deepout_block_asm( (int8_t*) Y_asm, &params, blk, (int8_t*)X, (int8_t*)K, (int16_t*) scales);
        }
#endif


        char str_buff[2000];

        //Check that all outputs are what they should be
        for(int co = 0; co < C_out; co++){
            for(int row = 0; row < Y_height; row++){
                for(int col = 0; col < Y_width; col++){
                    
                    unsigned in_region =  (row >= casse->top)
                                       && (col >= casse->left)
                                       && (row  < casse->top + casse->rows)
                                       && (col  < casse->left + casse->cols);

                    int8_t exp_out = in_region? 0x01 : 0xCC;

#if TEST_C 
                    int8_t c_val = Y_c[row][col][co];
#endif
#if TEST_ASM
                    int8_t asm_val = Y_asm[row][col][co];
#endif

#if (TEST_C && TEST_ASM)
                    //First thing to check is whether they match one another (if both are being tested)
                    //  Also report the actual values, so we know which (if either) is correct
                    if(c_val != asm_val){
                        sprintf(str_buff, 
                            "     C and ASM implementations gave different results for Y[%u][%u][%u] on vector %u. C: %d      ASM: %d    Expected: %d", 
                            row, col, co, v, c_val, asm_val, exp_out);
                    }
                    TEST_ASSERT_EQUAL_MESSAGE(c_val, asm_val, str_buff);
#endif

#if TEST_C
                    if(c_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                        sprintf(str_buff, "      C failed.  Y_c[%u][%u][%u] = %d. Expected %d.", row, col, co, c_val, exp_out);
                    }
                    TEST_ASSERT_EQUAL_MESSAGE(exp_out, c_val, str_buff);
#endif
#if TEST_ASM
                    if(asm_val != exp_out){   //just so we don't have to do the sprintf() unless it's wrong. Speeds things up immensely
                        sprintf(str_buff, "       ASM failed.  Y_asm[%u][%u][%u] = %d. Expected %d.", row, col, co, asm_val, exp_out);
                    }
                    TEST_ASSERT_EQUAL_MESSAGE(exp_out, asm_val, str_buff);
#endif
                }
            }
        }

        conv2d_shallowin_deepout_deinit(&params);
    }
}
#undef REGION
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON



void test_conv2d_shallowin_deepout()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_conv2d_shallowin_deepout_1x1);
    RUN_TEST(test_conv2d_shallowin_deepout_1x1_chans);
    RUN_TEST(test_conv2d_shallowin_deepout_1x1_xsize);
    RUN_TEST(test_conv2d_shallowin_deepout_3x3);
    RUN_TEST(test_conv2d_shallowin_deepout_regions);
}