
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

#if (defined(__XS3A__) && USE_ASM_conv2d_shallowin_deepout_block)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM) && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)

// static unsigned seed = 4412311;





unsafe {





#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
#define PAD_MODE_VALID  (0)
#define region_top      (0)
#define region_left     (0)
#define region_rows     (1)
#define region_cols     (1)
void test_conv2d_shallowin_deepout_case1()
{
#define Y_height (X_height)
#define Y_width (X_width)

    PRINTF("test_conv2d_shallowin_deepout_case1()...\n");

    int8_t  WORD_ALIGNED    K[C_out][K_h][8][C_in]          = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif

    {   //Initialize stuff
        for(int i = 0; i < C_out; i++){
            B[i] = i << 8;
            scales[i] = 0x4000;
        }
    }

    conv2d_sido_boggle_K((int8_t*) K, K_h, 8, C_in, C_out);

    data16_t* unsafe B_boggled = conv2d_boggle_B(B, C_out);

    nn_conv2d_sido_params_t params;

    padding_mode_t pad_mode = PAD_MODE_VALID? PADDING_VALID : PADDING_SAME;

    conv2d_shallowin_deepout_init(
        &params, X_height, X_width, K_h, K_w, C_in, C_out, 
        pad_mode, 0, region_top, region_left, region_rows, region_cols);

    TEST_ASSERT_EQUAL(1, params.block_count);

    for(int block = 0; block < params.block_count; block++){

#if TEST_C
        conv2d_shallowin_deepout_block_c((int8_t*)Y_c,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif

#if TEST_ASM
        conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif


    }

    for(int co = 0; co < C_out; co++){
#if TEST_C
        TEST_ASSERT_EQUAL(co, Y_c[0][0][co]);
#endif
#if TEST_ASM
        TEST_ASSERT_EQUAL(co, Y_asm[0][0][co]);
#endif
    }

    free(params.blocks);

#undef Y_width
#undef Y_height

}
#undef region_top
#undef region_left
#undef region_rows
#undef region_cols
#undef PAD_MODE_VALID
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON




















#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
#define PAD_MODE_VALID  (0)
#define region_top      (0)
#define region_left     (0)
#define region_rows     (1)
#define region_cols     (1)
void test_conv2d_shallowin_deepout_case2()
{
#define Y_height (X_height)
#define Y_width (X_width)

    PRINTF("test_conv2d_shallowin_deepout_case2()...\n");

    int8_t  WORD_ALIGNED    K[C_out][K_h][8][C_in]          = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif

    {   //Initialize stuff
        for(int i = 0; i < C_out; i++){
            B[i] = i << 9;
            shifts[i] = 1;
            scales[i] = 0x4000;
        }
    }

    conv2d_sido_boggle_K((int8_t*) K, K_h, 8, C_in, C_out);

    data16_t* unsafe B_boggled = conv2d_boggle_B(B, C_out);

    nn_conv2d_sido_params_t params;

    padding_mode_t pad_mode = PAD_MODE_VALID? PADDING_VALID : PADDING_SAME;

    conv2d_shallowin_deepout_init(
        &params, X_height, X_width, K_h, K_w, C_in, C_out, 
        pad_mode, 0, region_top, region_left, region_rows, region_cols);

    TEST_ASSERT_EQUAL(1, params.block_count);

    for(int block = 0; block < params.block_count; block++){

#if TEST_C
        conv2d_shallowin_deepout_block_c((int8_t*)Y_c,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif

#if TEST_ASM
        conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif
    }

    for(int co = 0; co < C_out; co++){
#if TEST_C
        TEST_ASSERT_EQUAL(co, Y_c[0][0][co]);
#endif
#if TEST_ASM
        TEST_ASSERT_EQUAL(co, Y_asm[0][0][co]);
#endif
    }

    free(params.blocks);

#undef Y_width
#undef Y_height

}
#undef region_top
#undef region_left
#undef region_rows
#undef region_cols
#undef PAD_MODE_VALID
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON




















#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
#define PAD_MODE_VALID  (0)
#define region_top      (0)
#define region_left     (0)
#define region_rows     (1)
#define region_cols     (1)
void test_conv2d_shallowin_deepout_case3()
{
#define Y_height (X_height)
#define Y_width (X_width)

    PRINTF("test_conv2d_shallowin_deepout_case3()...\n");

    int8_t  WORD_ALIGNED    K[C_out][K_h][8][C_in]          = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif

    {   //Initialize stuff
        for(int i = 0; i < C_out; i++){
            B[i] = i << 11;
            shifts[i] = 2;
            scales[i] = 0x2000;
        }
    }

    conv2d_sido_boggle_K((int8_t*) K, K_h, 8, C_in, C_out);

    data16_t* unsafe B_boggled = conv2d_boggle_B(B, C_out);

    nn_conv2d_sido_params_t params;

    padding_mode_t pad_mode = PAD_MODE_VALID? PADDING_VALID : PADDING_SAME;

    conv2d_shallowin_deepout_init(
        &params, X_height, X_width, K_h, K_w, C_in, C_out, 
        pad_mode, 0, region_top, region_left, region_rows, region_cols);

    TEST_ASSERT_EQUAL(1, params.block_count);

    for(int block = 0; block < params.block_count; block++){

#if TEST_C
        conv2d_shallowin_deepout_block_c((int8_t*)Y_c,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif

#if TEST_ASM
        conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif
    }

    for(int co = 0; co < C_out; co++){
#if TEST_C
        TEST_ASSERT_EQUAL(co, Y_c[0][0][co]);
#endif
#if TEST_ASM
        TEST_ASSERT_EQUAL(co, Y_asm[0][0][co]);
#endif
    }

    free(params.blocks);

#undef Y_width
#undef Y_height

}
#undef region_top
#undef region_left
#undef region_rows
#undef region_cols
#undef PAD_MODE_VALID
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON




















#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             (1)
#define K_w             (1)
#define X_height        (1)
#define X_width         (1)
#define PAD_MODE_VALID  (0)
#define region_top      (0)
#define region_left     (0)
#define region_rows     (1)
#define region_cols     (1)
void test_conv2d_shallowin_deepout_case4()
{
#define Y_height (X_height)
#define Y_width (X_width)

    PRINTF("test_conv2d_shallowin_deepout_case4()...\n");

    int8_t  WORD_ALIGNED    K[C_out][K_h][8][C_in]          = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif

    const int8_t X_val = 1 << 5;
    const int8_t K_val = 1 << 5;

    {   //Initialize stuff

        memset(X, X_val, sizeof(X));

        // memset(K, K_val, sizeof(K));

        for(int co = 0; co < C_out; co++){
            for(int kr = 0; kr < K_h; kr++){
                for(int kc = 0; kc < K_w; kc++){
                    for(int ci = 0; ci < C_in; ci++){
                        K[co][kr][kc][ci] = K_val;
                    }
                }
            }
        }

        for(int i = 0; i < C_out; i++){
            B[i] = i << 11;
            shifts[i] = 2;
            scales[i] = 0x2000;
        }
    }

    conv2d_sido_boggle_K((int8_t*) K, K_h, 8, C_in, C_out);

    data16_t* unsafe B_boggled = conv2d_boggle_B(B, C_out);

    nn_conv2d_sido_params_t params;

    padding_mode_t pad_mode = PAD_MODE_VALID? PADDING_VALID : PADDING_SAME;

    conv2d_shallowin_deepout_init(
        &params, X_height, X_width, K_h, K_w, C_in, C_out, 
        pad_mode, 0, region_top, region_left, region_rows, region_cols);

    TEST_ASSERT_EQUAL(1, params.block_count);

    for(int block = 0; block < params.block_count; block++){

#if TEST_C
        conv2d_shallowin_deepout_block_c((int8_t*)Y_c,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif

#if TEST_ASM
        conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif
    }

    for(int co = 0; co < C_out; co++){
#if TEST_C
        TEST_ASSERT_EQUAL(co+2, Y_c[0][0][co]);
#endif
#if TEST_ASM
        TEST_ASSERT_EQUAL(co+2, Y_asm[0][0][co]);
#endif
    }

    free(params.blocks);

#undef Y_width
#undef Y_height

}
#undef region_top
#undef region_left
#undef region_rows
#undef region_cols
#undef PAD_MODE_VALID
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON




















#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( VPU_INT8_ACC_PERIOD )
#define C_in            ( 4 )
#define K_h             ( 3 )
#define K_w             ( 3 )
#define X_height        ( 3 )
#define X_width         ( 3 )
#define PAD_MODE_VALID  ( 0 )
#define region_top      ( 0 )
#define region_left     ( 0 )
#define region_rows     ( 3 )
#define region_cols     ( 3 )
void test_conv2d_shallowin_deepout_case5()
{
#define Y_height (X_height)
#define Y_width (X_width)

    PRINTF("test_conv2d_shallowin_deepout_case5()...\n");

    int8_t  WORD_ALIGNED    K[C_out][K_h][8][C_in]          = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };

#if TEST_C
    int8_t  WORD_ALIGNED    Y_c[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif
#if TEST_ASM
    int8_t  WORD_ALIGNED    Y_asm[Y_height][Y_width][C_out]     = {{{ 0 }}};
#endif

    // const int8_t X_val = 1 << 5;
    // const int8_t K_val = 1 << 5;

    {   //Initialize stuff

        // memset(X, X_val, sizeof(X));

        // // memset(K, K_val, sizeof(K));

        // for(int co = 0; co < C_out; co++){
        //     for(int kr = 0; kr < K_h; kr++){
        //         for(int kc = 0; kc < K_w; kc++){
        //             for(int ci = 0; ci < C_in; ci++){
        //                 K[co][kr][kc][ci] = K_val;
        //             }
        //         }
        //     }
        // }

        for(int i = 0; i < C_out; i++){
            B[i] = i << 8;
            shifts[i] = 0;
            scales[i] = 0x4000;
        }
    }

    conv2d_sido_boggle_K((int8_t*) K, K_h, 8, C_in, C_out);

    data16_t* unsafe B_boggled = conv2d_boggle_B(B, C_out);

    nn_conv2d_sido_params_t params;

    padding_mode_t pad_mode = PAD_MODE_VALID? PADDING_VALID : PADDING_SAME;

    conv2d_shallowin_deepout_init(
        &params, X_height, X_width, K_h, K_w, C_in, C_out, 
        pad_mode, 64, region_top, region_left, region_rows, region_cols);

    //final 32-bit accumulator should be   (co<<8) + (2**6 * C_in * padding_cells)
    //                                     co * 2**8  + 2**8 * padding_cells
    //                                     (co + padding_cells) * 2**8
    // with a shift of 0 and scale of 0x4000, that makes the final value  (co + padding_cells)

    TEST_ASSERT_EQUAL(9, params.block_count);

    // PRINTF("Blocks: %d\n", params.block_count);
    for(int block = 0; block < params.block_count; block++){

#if TEST_C
        conv2d_shallowin_deepout_block_c((int8_t*)Y_c,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif

#if TEST_ASM
        conv2d_shallowin_deepout_block_asm((int8_t*)Y_asm,  
                                             &params, 
                                             (nn_conv2d_sido_block_params_t*) &params.blocks[block], 
                                             (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);
#endif
    }

    unsigned table[K_h][K_w] = {
        {5, 3, 5},
        {3, 0, 3},
        {5, 3, 5},
    };

    for(int kr = 0; kr < K_h; kr++){
        for(int kc = 0; kc < K_w; kc++){
            for(int co = 0; co < C_out; co++){

                int pad_cells = table[kr][kc];

#if TEST_C
                TEST_ASSERT_EQUAL(co+pad_cells, Y_c[kr][kc][co]);
                // printf("%d\t%d\t%d\n", kr, kc, co);
#endif
#if TEST_ASM
                TEST_ASSERT_EQUAL(co+pad_cells, Y_asm[kr][kc][co]);
#endif
            }
        }
    }

    free(params.blocks);

#undef Y_width
#undef Y_height

}
#undef region_top
#undef region_left
#undef region_rows
#undef region_cols
#undef PAD_MODE_VALID
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON






















// void test_conv2d_shallowin_deepout()
// {
//     test_conv2d_shallowin_deepout_case1();
//     test_conv2d_shallowin_deepout_case2();
//     test_conv2d_shallowin_deepout_case3();
//     test_conv2d_shallowin_deepout_case4();
// }

}
