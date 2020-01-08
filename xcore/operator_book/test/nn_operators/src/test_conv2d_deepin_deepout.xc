
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
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM) && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)


unsafe {



#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           ( 2 * VPU_INT8_ACC_PERIOD )
#define C_in            ( 2 * VPU_INT8_EPV )
#define K_h             (5)
#define K_w             (5)
#define X_height        (8)
#define X_width         (8)
#define PAD_MODE_VALID  (0)
#define region_top      (0)
#define region_left     (0)
#define region_rows     (8)
#define region_cols     (8)
void test_conv2d_deepin_deepout_case123()
{

#if (PAD_MODE_VALID)
  #define Y_height (X_height- 2*(K_h>>1))
  #define Y_width (X_width - 2*(K_w>>1))
#else
  #define Y_height (X_height)
  #define Y_width (X_width)
#endif

    PRINTF("test_conv2d_deepin_deepout()...\n");

    int8_t  WORD_ALIGNED    K[C_out][K_h][K_w][C_in]        = {{{{ 0 }}}};
    int8_t  WORD_ALIGNED    X[X_height][X_width][C_in]      = {{{ 0 }}};
    int32_t WORD_ALIGNED    B[C_out]                        = { 0 };
    int16_t WORD_ALIGNED    shifts[C_out]                   = { 0 };
    int16_t WORD_ALIGNED    scales[C_out]                   = { 0 };
    int8_t  WORD_ALIGNED    Y[Y_height][Y_width][C_out]     = {{{ 0 }}};

    {   //Initialize stuff
        for(int i = 0; i < C_out; i++){
            B[i] = i << 8;
            scales[i] = 0x4000;
        }
    }

    conv2d_dido_boggle_K((int8_t*) K, K_h, K_w, C_in, C_out);

    data16_t* unsafe B_boggled = conv2d_boggle_B(B, C_out);

    nn_conv2d_dido_params_t params;

    padding_mode_t pad_mode = PAD_MODE_VALID? PADDING_VALID : PADDING_SAME;

    conv2d_deepin_deepout_init(&params, 
                                    X_height, X_width, K_h, K_w, C_in, C_out, 
                                    pad_mode, 0, 
                                    region_top, region_left, region_rows, region_cols);

    for(int block = 0; block < params.block_count; block++){

        memset(Y, 0, sizeof(Y));

        conv2d_deepin_deepout_block_asm((int8_t*)Y,  
                                                &params, 
                                                (nn_conv2d_dido_block_params_t*) &params.blocks[block], 
                                                (int8_t*)X, (int8_t*)K, B_boggled, shifts, scales);

        int print_chan = 1;
        printf("\n");
        for(int row = 0; row < Y_height; row++){
            for(int col = 0; col < Y_width; col++){
                printf("% 2d", Y[row][col][print_chan]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

#undef Y_width
#undef Y_height

}
#undef PAD_MODE
#undef X_width
#undef X_height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON





void test_conv2d_deepin_deepout()
{
    // test_conv2d_deepin_deepout_case1();
    // test_conv2d_deepin_deepout_case2();
    // test_conv2d_deepin_deepout_case3();
    // test_conv2d_deepin_deepout_case4();
}


}