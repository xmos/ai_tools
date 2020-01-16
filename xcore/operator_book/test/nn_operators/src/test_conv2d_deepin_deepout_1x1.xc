
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
#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#if (defined(__XS3A__) && USE_ASM_conv2d_deepin_deepout_block)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (1)
#endif

#define TEST_ASM ((HAS_ASM) && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)


unsafe {










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
#define C_in            ( VPU_INT8_EPV )
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
                        //top, left, rows, cols
#define REGION          {0,    0,    1,    1}
void test_conv2d_deepin_deepout_1x1()
{
    const int8_t zero_point = 0;
    const unsigned Y_height = X_height;
    const unsigned Y_width = X_width;

    int8_t  WORD_ALIGNED    K[C_out][K_h][K_w][C_in]        = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height][X_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height][X_width][C_out]     = {{{ 0 }}};
#endif

    nn_conv2d_init_params_t init_params = { X_height, X_width, K_h, K_w, C_in, C_out, PADDING_SAME, zero_point };
    const nn_conv2d_region_params_t region_params = REGION;
    
    nn_conv2d_dido_params_t params;

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
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   // (2^5 * (0   * 2^0) ) / 2^8 = 0     (out chans same)
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x08},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // (2^5 * (2^3 * 2^0) ) / 2^8 = 2^0   (out chans same)
        {   { 0x00,  0x08},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x01},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // (2^5 * (2^0 * 2^3) ) / 2^8 = 2^0   (out chans same)
        {   { 0x00,  0x02},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x04},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x01,   0}  },   // (2^5 * (2^2 * 2^1) ) / 2^8 = 2^0   (out chans same)
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x20,   0}  },   // (2^5 * (2^2 * 2^1) ) / 2^8 = 2^5   (out chans same)

        // Vector 31 -- Kernel tests (input channels differ)
        {   { 0x04,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x7F,   0}  },  // Input channels differ (saturation)
        {   { 0x04,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x7C,   0}  },  // Input channels differ (no saturation)
        {   { 0x04,  0x00},     { 0x00000000, -0x00024080,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,  -0x50,   0}  },  // Input channels differ (bias prevents saturation)
        {   { 0x04,  0x00},     { 0x00000000, -0x00024081,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,  -0x51,   0}  },  // Input channels differ (bias prevents saturation)

        // Vector 35 -- Kernel tests (output channels differ)
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x04,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x08,   0x00,   0}  },   // Output channels differ (no saturation)
        {   { 0x00,  0x20},     { 0x00000000,  0x00000000,  0},    { 0x04,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x10,   0x00,   0}  },   // Output channels differ (saturation on some)
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x04,  0x20},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x08,   0x40,   0}  },   // Output channels differ
        {   { 0x00,  0x20},     { 0x00000000,  0x00000000,  0},    { 0x04, -0x20},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x10,  -0x80,   0}  },   // Output channels differ 

        // Vector 39 -- Kernel tests (input channels and output channels differ)

        // 0 --  0
        // 1 --  4  * (0 + 4 + 8 + 12 + ...)  = 4  * 4 * (0 + 1 + 2 + ... + 31) = 4  * 4 * 16 * 31 = 4  * 2^6 * 31 = 1 * 2^8 * 31
        // 2 --  8  * (0 + 4 + 8 + 12 + ...)  = 8  * 4 * (0 + 1 + 2 + ... + 31) = 8  * 4 * 16 * 31 = 8  * 2^6 * 31 = 2 * 2^8 * 31
        // 3 --  12 * (0 + 4 + 8 + 12 + ...)  = 12 * 4 * (0 + 1 + 2 + ... + 31) = 12 * 4 * 16 * 31 = 12 * 2^6 * 31 = 3 * 2^8 * 31

        // cout --  cout * 2^8 * 31
        // after vdepth8 ->   cout * 31
        
        {   { 0x04,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x04,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x1F,   0x00,   0}  },   //

        // {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   //
    };

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("test_conv2d_deepin_deepout_1x1()...\n");

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

                shifts[cout] = casse->shift.scale * cout + casse->shift.offset;
                scales[cout] = casse->scale.scale * cout + casse->scale.offset;
            }

            //Set kernel
            for(int cout = 0; cout < C_out; cout++){

                int8_t value = casse->kernel.scale * cout + casse->kernel.offset;

                // printf("K[%d] -> %d\n", cout, value);
                // printf("B[%d] -> %d\n", cout, B[cout]);
                memset(&K[cout], value, sizeof(int8_t) * K_h * K_w * C_in);

                // for(int krow = 0; krow < K_h; krow++){
                //     for(int kcol = 0; kcol < K_w; kcol++){
                //         for(int cin = 0; cin < C_in; cin++){
                //             K[cout][krow][kcol][cin] = value;
                //         }
                //     }
                // }
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
            
            conv2d_dido_boggle_K((int8_t*) K, K_h, K_w, C_in, C_out);


            conv2d_boggle_B(B, C_out);

            
            conv2d_deepin_deepout_init(&params, &init_params, &region_params, (int8_t*) K, (data16_t* unsafe) B);


            //There should always be exactly one block in this test.  
            TEST_ASSERT_EQUAL(1, params.block_count);

            //Perform the actual convolution(s)   (run both C and ASM before checking either)
    #if TEST_C
            memset(Y_c, 0xCC, sizeof(Y_c));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_dido_block_params_t* unsafe blk = &params.blocks[block];
                conv2d_deepin_deepout_block_c(  (int8_t*)Y_c,     &params, blk, (int8_t*)X, (int8_t*)K, shifts, scales);
            }
    #endif

    #if TEST_ASM
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_dido_block_params_t* unsafe blk = &params.blocks[block];
                conv2d_deepin_deepout_block_asm((int8_t*)Y_asm, &params, blk, (int8_t*)X, (int8_t*)K, shifts, scales);
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


            conv2d_deepin_deepout_deinit(&params);
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
#define C_in            (2 * VPU_INT8_EPV)
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
                        //top, left, rows, cols
#define REGION          {0,    0,    1,    1}
void test_conv2d_deepin_deepout_1x1_chans()
{
    const int8_t zero_point = 0;
    const unsigned Y_height = X_height;
    const unsigned Y_width = X_width;

    int8_t  WORD_ALIGNED    K[C_out][K_h][K_w][C_in]        = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height][X_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height][X_width][C_out]     = {{{ 0 }}};
#endif

    nn_conv2d_init_params_t init_params = { X_height, X_width, K_h, K_w, C_in, C_out, PADDING_SAME, zero_point };
    const nn_conv2d_region_params_t region_params = REGION;
    
    nn_conv2d_dido_params_t params;

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
        {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x08},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x02,   0}  },   // (2^6 * (2^3 * 2^0) ) / 2^8 = 2^1  
        {   { 0x00,  0x08},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x01},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x02,   0}  },   // (2^6 * (2^0 * 2^3) ) / 2^8 = 2^1  
        {   { 0x00,  0x02},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x04},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x02,   0}  },   // (2^6 * (2^2 * 2^1) ) / 2^8 = 2^1
        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x10},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x40,   0}  },   // (2^6 * (2^2 * 2^1) ) / 2^8 = 2^6  

        // Vector 22 -- Kernel tests (input channels differ)        
        {   { 0x02,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x7F,   0}  },  // Input channels differ (saturation)
        {   { 0x02,  0x00},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x08},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x7E,   0}  },  // Input channels differ (no saturation)
        {   { 0x02,  0x00},     { 0x00000000, -0x0003f000,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },  // Input channels differ (bias prevents saturation)
        {   { 0x02,  0x00},     { 0x00000000, -0x0003f200,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,  -0x02,   0}  },  // Input channels differ (bias prevents saturation)
        {   { 0x02,  0x00},     { 0x00000000, -0x00041100,  0},    { 0x00,  0x40},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,  -0x21,   0}  },  // Input channels differ (bias prevents saturation)

        // Vector 27 -- Kernel tests (output channels differ)

        {   { 0x00,  0x10},     { 0x00000000,  0x00000000,  0},    { 0x02,  0x00},     { 0x00, 0x03},     { 0x0000,  0x4000},    { 0x01,   0x00,   0}  },   // Output channels differ (no saturation)
        {   { 0x00,  0x20},     { 0x00000000,  0x00000000,  0},    { 0x02,  0x00},     { 0x00, 0x03},     { 0x0000,  0x4000},    { 0x02,   0x00,   0}  },   // Output channels differ (no saturation)
        {   { 0x00,  0x40},     { 0x00000000,  0x00000000,  0},    { 0x02,  0x00},     { 0x00, 0x03},     { 0x0000,  0x4000},    { 0x04,   0x00,   0}  },   // Output channels differ (saturation on some)

        // Vector 30 -- Kernel tests (input/output both differ)
        {   { 0x04, -0x80},     { 0x00000000,  0x00000000,  0},    { 0x02,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    {-0x01,   0x00,   0}  },   //
        {   { 0x02,  0x00},     { 0x00000100, -0x00002000,  0},    { 0x04, -0x80},     { 0x00, 0x06},     { 0x0000,  0x4000},    { 0x01,  -0x20,   0}  },   //
        {   { 0x04, -0x80},     { 0x00000000,  0x00000000,  0},    { 0x04, -0x80},     { 0x00, 0x00},     { 0x0000,  0x4000},    {-0x02,   0x40,   0}  },   //

        
        // {   { 0x00,  0x01},     { 0x00000000,  0x00000000,  0},    { 0x00,  0x00},     { 0x00, 0x00},     { 0x0000,  0x4000},    { 0x00,   0x00,   0}  },   //
    };

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("test_conv2d_deepin_deepout_1x1_chans()...\n");

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
                    B[cout] += (1 << (cout/4 + casse->bias.exp));

                shifts[cout] = casse->shift.scale * cout/4 + casse->shift.offset;
                scales[cout] = casse->scale.scale * cout + casse->scale.offset;
            }

            //Set kernel
            for(int cout = 0; cout < C_out; cout++){

                int8_t value = casse->kernel.scale * cout + casse->kernel.offset;

                // printf("K[%d] -> %d\n", cout, value);
                // printf("B[%d] -> %d\n", cout, B[cout]);
                memset(&K[cout], value, sizeof(int8_t) * K_h * K_w * C_in);

                // for(int krow = 0; krow < K_h; krow++){
                //     for(int kcol = 0; kcol < K_w; kcol++){
                //         for(int cin = 0; cin < C_in; cin++){
                //             K[cout][krow][kcol][cin] = value;
                //         }
                //     }
                // }
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
            
            conv2d_dido_boggle_K((int8_t*) K, K_h, K_w, C_in, C_out);


            conv2d_boggle_B(B, C_out);


            conv2d_deepin_deepout_init(&params, &init_params, &region_params, (int8_t*) K, (data16_t* unsafe) B);


            //There should always be exactly one block in this test.  
            TEST_ASSERT_EQUAL(1, params.block_count);

            //Perform the actual convolution(s)   (run both C and ASM before checking either)
    #if TEST_C
            memset(Y_c, 0xCC, sizeof(Y_c));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_dido_block_params_t* unsafe blk = &params.blocks[block];
                conv2d_deepin_deepout_block_c(  (int8_t*)Y_c,     &params, blk, (int8_t*)X, (int8_t*)K, shifts, scales);
            }
    #endif

    #if TEST_ASM
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_dido_block_params_t* unsafe blk = &params.blocks[block];
                conv2d_deepin_deepout_block_asm((int8_t*)Y_asm, &params, blk, (int8_t*)X, (int8_t*)K, shifts, scales);
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


            conv2d_deepin_deepout_deinit(&params);
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
#define C_in            ( VPU_INT8_EPV )
#define K_h             (1)
#define K_w             (1)
#define X_height_max    (3)
#define X_width_max     (3)
void test_conv2d_deepin_deepout_1x1_xsize()
{
    const int8_t zero_point = 12;

    int8_t  WORD_ALIGNED    K[C_out][K_h][K_w][C_in]            = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height_max][X_width_max][C_in]  = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                            = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                       = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                       = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[X_height_max][X_width_max][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[X_height_max][X_width_max][C_out]     = {{{ 0 }}};
#endif

    
    nn_conv2d_dido_params_t params;

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
            { 0,  0,  0}},  {1, 1},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {1, 2},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x1F, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {2, 1},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  OxXX, OxXX},
                                                                                                                                     { 0x1F,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {1, 3},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x1F, 0x1F},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {2, 2},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x1F, OxXX},
                                                                                                                                     { 0x1F,  0x1F, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {2, 3},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x1F, 0x1F},
                                                                                                                                     { 0x1F,  0x1F, 0x1F},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {3, 2},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x1F, OxXX},
                                                                                                                                     { 0x1F,  0x1F, OxXX},
                                                                                                                                     { 0x1F,  0x1F, OxXX}} },
        {  {{ 0,  0,  0},
            { 0,  0,  0},   
            { 0,  0,  0}},  {3, 3},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x1F, 0x1F},
                                                                                                                                     { 0x1F,  0x1F, 0x1F},
                                                                                                                                     { 0x1F,  0x1F, 0x1F}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {1, 1},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {1, 3},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x21, 0x23},
                                                                                                                                     { OxXX,  OxXX, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {3, 1},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  OxXX, OxXX},
                                                                                                                                     { 0x25,  OxXX, OxXX},
                                                                                                                                     { 0x2B,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {2, 2},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x21, OxXX},
                                                                                                                                     { 0x25,  0x27, OxXX},
                                                                                                                                     { OxXX,  OxXX, OxXX}} },
        {  {{ 0,  1,  2},
            { 3,  4,  5},   
            { 6,  7,  8}},  {3, 3},     { 0x00000000,  0x00000000},    { 0x00,  0x10},    { 0x00, 0x00},     { 0x0000,  0x4000},    {{ 0x1F,  0x21, 0x23},
                                                                                                                                     { 0x25,  0x27, 0x29},
                                                                                                                                     { 0x2B,  0x2D, 0x2F}} },



    };

    const unsigned START_ON_CASE = 0;
    const unsigned STOP_ON_CASE = (unsigned) -1;

    const unsigned casse_count = sizeof(casses) / sizeof(case1x1_params_t);

    PRINTF("test_conv2d_deepin_deepout_1x1_xsize()...\n");

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

                shifts[cout] = casse->shift.scale * cout + casse->shift.offset;
                scales[cout] = casse->scale.scale * cout + casse->scale.offset;
            }

            //Set kernel
            for(int cout = 0; cout < C_out; cout++){

                int8_t value = casse->kernel.scale * cout + casse->kernel.offset;

                // printf("K[%d] -> %d\n", cout, value);
                // printf("B[%d] -> %d\n", cout, B[cout]);
                memset(&K[cout], value, sizeof(int8_t) * K_h * K_w * C_in);

                // for(int krow = 0; krow < K_h; krow++){
                //     for(int kcol = 0; kcol < K_w; kcol++){
                //         for(int cin = 0; cin < C_in; cin++){
                //             K[cout][krow][kcol][cin] = value;
                //         }
                //     }
                // }
            }
            
            conv2d_dido_boggle_K((int8_t*) K, K_h, K_w, C_in, C_out);


            conv2d_boggle_B(B, C_out);

            
            conv2d_deepin_deepout_init(&params, &init_params, NULL, (int8_t*) K, (data16_t* unsafe) B);


            //There should always be exactly one block in this test.  
            TEST_ASSERT_EQUAL(1, params.block_count);

            //Perform the actual convolution(s)   (run both C and ASM before checking either)
#if TEST_C
            memset(Y_c, OxXX, sizeof(Y_c));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_dido_block_params_t* unsafe blk = &params.blocks[block];
                conv2d_deepin_deepout_block_c(  (int8_t*)Y_c,     &params, blk, (int8_t*)X, (int8_t*)K, shifts, scales);
            }
#endif

#if TEST_ASM
            memset(Y_asm, OxXX, sizeof(Y_asm));
            for(int block = 0; block < params.block_count; block++){
                const nn_conv2d_dido_block_params_t* unsafe blk = &params.blocks[block];
                conv2d_deepin_deepout_block_asm((int8_t*)Y_asm, &params, blk, (int8_t*)X, (int8_t*)K, shifts, scales);
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


            conv2d_deepin_deepout_deinit(&params);
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




}
