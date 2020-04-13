
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "xs3_vpu.h"

#include "unity.h"



#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

#define MIN_CHAN_OUT_GROUPS(CHAN_COUNT) (((CHAN_COUNT+(VPU_INT8_VLMACC_ELMS-1))>>VPU_INT8_VLMACC_ELMS_LOG2)<<VPU_INT8_VLMACC_ELMS_LOG2)



static void check_Y(
    const int8_t y_exp, 
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line,
    const int8_t* Y,
    const nn_image_params_t* y_params)
{
    char str_buff[200];

    unsigned y_offset = IMG_ADDRESS_VECT(y_params, row, col, chn);

    int8_t y = Y[y_offset];

    //Only sprintf-ing if the test will fail saves a ton of time.
    if(y != y_exp){
        sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)  [test vector @ line %u]", 
                row, col, chn, line);
    }

    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
}







#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANS_IN        (VPU_INT8_VLMACC_ELMS)
#define CHANS_OUT       (CHANS_IN)
#define CHANS_OUT_CEIL  MIN_CHAN_OUT_GROUPS(CHANS_OUT)
#define X_HEIGHT        (3)
#define X_WIDTH         (3)
#define Y_HEIGHT        (X_HEIGHT)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (1)
#define K_w             (1)
#define K_vstride       (1)
#define K_hstride       (1)
void test_conv2d_depthwise_case0()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];

    int8_t WORD_ALIGNED  K[K_h][K_w][CHANS_OUT];

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

    PRINTF("%s...\n", __func__);

    typedef struct {
        int8_t x;
        int8_t k;
        int32_t bias;
        int16_t shift1;
        int16_t scale;
        int16_t shift2;

        int8_t y;
        unsigned line;
    } test_case_t;


    //   Y[i] = C_in * (x * k)

    const test_case_t casses[] = {
        //  X       K           bias            shift1      scale       shift2      Y
        {   0x00,   0x00,       0x00000000,     0,          0x0000,     0,          0x00,       __LINE__}, 
        {   0x00,   0x00,       0x00000000,     0,          0x0001,     0,          0x00,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     0,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     0,          0x0002,     0,          0x02,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     0,         -0x0001,     0,         -0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     0,          0x0004,     1,          0x02,       __LINE__}, 
        {   0x00,   0x00,       0x00000004,     0,          0x0001,     2,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x0000007f,     0,          0x0001,     0,          0x7f,       __LINE__}, 
        {   0x00,   0x00,       0x00000080,     0,          0x0001,     0,          0x7f,       __LINE__}, 
        {   0x00,   0x00,       0x00007f00,     0,          0x0001,     0,          0x7f,       __LINE__}, 
        {   0x00,   0x00,       0x00007f00,     0,          0x0001,     8,          0x7f,       __LINE__}, 
        {   0x00,   0x00,       0x00007f00,     0,          0x0001,     9,          0x40,       __LINE__}, 
        {   0x00,   0x00,       0x00010000,     0,          0x0001,     0,          0x7f,       __LINE__}, 
        {   0x00,   0x00,       0x00020000,     0,          0x0001,     0,          0x7f,       __LINE__}, 
        {   0x00,   0x00,       0x00020000,     3,          0x0001,     0,          0x7f,       __LINE__}, 
        {   0x00,   0x00,       0x00020000,     3,          0x0001,     8,          0x40,       __LINE__}, 
        {   0x01,   0x00,       0x00000000,     0,          0x0001,     0,          0x00,       __LINE__}, 
        {   0x00,   0x01,       0x00000000,     0,          0x0001,     0,          0x00,       __LINE__}, 
        {   0x01,   0x01,       0x00000000,     0,          0x0001,     0,          0x01,       __LINE__}, 
        {  -0x01,   0x01,       0x00000000,     0,          0x0001,     0,         -0x01,       __LINE__}, 
        {   0x01,  -0x01,       0x00000000,     0,          0x0001,     0,         -0x01,       __LINE__}, 
        {   0x02,   0x02,       0x00000010,     0,          0x0001,     0,          0x14,       __LINE__}, 
        {   0x40,   0x40,       0x00001000,     2,         -0x0080,    12,         -0x40,       __LINE__}, 

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, 1, 1);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){

        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

        memset(X, casse->x, sizeof(X));
        memset(K, casse->k, sizeof(K));

        for(int k = 0; k < CHANS_OUT; k++){
            BSS.bias[k]     = casse->bias;
            BSS.shift1[k]   = casse->shift1;
            BSS.scale[k]    = casse->scale;
            BSS.shift2[k]   = casse->shift2;
        }


        nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);

        nn_conv2d_depthwise_plan_t plan;
        nn_conv2d_depthwise_job_t job;

        conv2d_depthwise_init(&plan, &job, &x_params, &y_params, NULL, 0, 0, K_h, K_w, K_vstride, K_hstride, 12, 1);


#if (DEBUG_ON || 0)

#endif //DEBUG_ON

        PRINTF("\t\t\tC...\n");
        memset(Y, 0xCC, sizeof(Y)); 
        conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) &bss, &plan, &job);

        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = casse->y;

                    check_Y(y_exp, row, col, chn, casse->line, (int8_t*) Y, &y_params);
                }
            }
        }

    }

}
#undef DEBUG_ON
#undef CHANS_IN
#undef CHANS_OUT
#undef CHANS_OUT_CEIL
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_h
#undef K_w
#undef K_vstride
#undef K_hstride











#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANS_IN_MAX    (5*VPU_INT8_VLMACC_ELMS)
#define CHANS_OUT_MAX   (CHANS_IN_MAX)
#define X_HEIGHT        (2)
#define X_WIDTH         (2)
#define Y_HEIGHT        (X_HEIGHT)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (1)
#define K_w             (1)
#define K_vstride       (1)
#define K_hstride       (1)
void test_conv2d_depthwise_case1()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN_MAX];

    int8_t WORD_ALIGNED  K[K_h][K_w][CHANS_OUT_MAX];

    struct {
        int32_t bias[CHANS_OUT_MAX];
        int16_t shift1[CHANS_OUT_MAX];
        int16_t scale[CHANS_OUT_MAX];
        int16_t shift2[CHANS_OUT_MAX];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT_MAX)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT_MAX];

    PRINTF( "test_conv2d_depthwise_case1()...\n");

    typedef struct {
        int8_t x;
        int8_t k;
        int8_t y;
        unsigned line;
    } test_case_t;


    //   Y[i] = C_in * (x * k)

    const test_case_t casses[] = {
        //  X       K       Y
        {   0x00,   0x00,   0x00,       __LINE__}, 
        {   0x01,   0x00,   0x00,       __LINE__}, 
        {   0x00,   0x01,   0x00,       __LINE__}, 
        {   0x01,   0x01,   0x01,       __LINE__}, 
        {  -0x01,   0x01,  -0x01,       __LINE__}, 
        {   0x01,  -0x01,  -0x01,       __LINE__}, 
        {   0x02,   0x02,   0x04,       __LINE__}, 
        {   0x08,   0x04,   0x20,       __LINE__}, 
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, 1, 1);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){

        const test_case_t* casse = (const test_case_t*) &casses[v];

        unsigned chan_counts[] = {16, 32, 64, 4, 8, 12, 24, 36};

        for(int p = 0; p < sizeof(chan_counts)/sizeof(unsigned); p++){
            unsigned channel_count = chan_counts[p];
            
            PRINTF("\ttest vector %u...(%u channels)\n", v, channel_count);
            
            nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, channel_count };
            nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, channel_count };

            memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
            memset(K, casse->k, K_h * K_w * y_params.channels * sizeof(int8_t));

            for(int k = 0; k < y_params.channels; k++){
                BSS.bias[k]     = 0;
                BSS.shift1[k]   = 0;
                BSS.scale[k]    = 1;
                BSS.shift2[k]   = 0;
            }

            nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                    (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, channel_count);

            nn_conv2d_depthwise_plan_t plan;
            nn_conv2d_depthwise_job_t job;

            conv2d_depthwise_init(&plan, &job, &x_params, &y_params, NULL, 0, 0, K_h, K_w, K_vstride, K_hstride, 12, 1);

#if (DEBUG_ON || 0)

#endif //DEBUG_ON



#if TEST_C
            PRINTF("\t\t\tC...\n");
            memset(Y, 0xCC, sizeof(Y)); 
            conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif
#if TEST_ASM
            PRINTF("\t\t\tASM...\n");
            memset(Y_asm, 0xCC,  sizeof(Y_asm));
            conv2d_depthwise_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif

            PRINTF("\t\t\tChecking...\n");
            for(unsigned row = 0; row < y_params.height; row++){
                for(unsigned col = 0; col < y_params.width; col++){
                    for(unsigned chn = 0; chn < y_params.channels; chn++){
                        
                        int8_t y_exp = casse->y;

                        check_Y(y_exp, row, col, chn, casse->line, (int8_t*) Y, &y_params);
                    }
                }
            }

#if TEST_C
            TEST_ASSERT_EQUAL((int8_t)0xCC, ((int8_t*)Y)[IMG_ADDRESS_VECT(&y_params, y_params.height-1, y_params.width-1, y_params.channels-1)+1]);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL((int8_t)0xCC, ((int8_t*)Y_asm)[IMG_ADDRESS_VECT(&y_params, y_params.height-1, y_params.width-1, y_params.channels-1)+1]);
#endif
        }

    }

}
#undef DEBUG_ON
#undef CHANS_IN_MAX
#undef CHANS_OUT_MAX
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_h
#undef K_w
#undef K_vstride
#undef K_hstride














#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANNELS        (2*VPU_INT8_VLMACC_ELMS)
#define X_HEIGHT        (6)
#define X_WIDTH         (6)
#define Y_HEIGHT_MAX    (X_HEIGHT)
#define Y_WIDTH_MAX     (X_WIDTH)
void test_conv2d_depthwise_case2()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANNELS];

    int8_t WORD_ALIGNED  K[X_HEIGHT][X_WIDTH][CHANNELS];

    struct {
        int32_t bias[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift1[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t scale[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift2[MIN_CHAN_OUT_GROUPS(CHANNELS)];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANNELS)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT_MAX][Y_WIDTH_MAX][CHANNELS+1];

    PRINTF( "test_conv2d_depthwise_case2()...\n");

    typedef struct {
        int8_t x;
        int8_t k;
        unsigned K_h;
        unsigned K_w;
        unsigned v_stride;
        unsigned h_stride;
        unsigned line;
    } test_case_t;


    const test_case_t casses[] = {
        //     X       K    K_h     K_w     v_stride    h_stride     
        {   0x01,   0x01,   1,      1,      1,          1,              __LINE__}, 
        {   0x01,   0x01,   1,      2,      1,          2,              __LINE__}, 
        {   0x01,   0x01,   1,      2,      1,          2,              __LINE__}, 
        {   0x01,   0x01,   2,      2,      2,          2,              __LINE__}, 
        {   0x01,   0x01,   1,      3,      1,          3,              __LINE__}, 
        {   0x01,   0x01,   2,      3,      2,          3,              __LINE__}, 
        {   0x01,   0x01,   3,      3,      3,          3,              __LINE__}, 
        {   0x01,   0x01,   2,      6,      2,          6,              __LINE__}, 
        {   0x01,   0x01,   6,      3,      6,          3,              __LINE__}, 
        {   0x01,   0x01,   6,      6,      6,          6,              __LINE__}, 
        {   0x01,   0x01,   4,      4,      4,          4,              __LINE__}, 
        {   0x01,   0x01,   4,      4,      1,          1,              __LINE__},
        {   0x01,   0x01,   4,      4,      2,          2,              __LINE__}, 
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, 1, 1);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){

        const test_case_t* casse = (const test_case_t*) &casses[v];

        unsigned y_height = 1 + (X_HEIGHT - casse->K_h) / casse->v_stride;
        unsigned y_width  = 1 + (X_WIDTH  - casse->K_w)  / casse->h_stride;
        PRINTF("\ttest vector %u... (%d, %d)\n", v, y_height, y_width);

        
        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANNELS };
        nn_image_params_t y_params = { y_height, y_width, CHANNELS };

        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, casse->k, casse->K_h * casse->K_w * y_params.channels * sizeof(int8_t));

        for(int k = 0; k < y_params.channels; k++){
            BSS.bias[k]     = 0;
            BSS.shift1[k]   = 0;
            BSS.scale[k]    = 1;
            BSS.shift2[k]   = 0;
        }

        nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANNELS);

        nn_conv2d_depthwise_plan_t plan;
        nn_conv2d_depthwise_job_t job;

        conv2d_depthwise_init(&plan, &job, &x_params, &y_params, NULL, 0, 0, casse->K_h, casse->K_w, casse->v_stride, casse->h_stride, 12, 1);

#if (DEBUG_ON || 0)

#endif //DEBUG_ON



#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y, 0xCC, sizeof(Y)); 
        conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC,  sizeof(Y_asm));
        conv2d_depthwise_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = casse->x * casse->k * casse->K_h * casse->K_w;

                    check_Y(y_exp, row, col, chn, casse->line, (int8_t*) Y, &y_params);
                }
            }
        }

#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE((int8_t)0xCC, ((int8_t*)Y)[IMG_ADDRESS_VECT(&y_params, y_params.height-1, y_params.width-1, y_params.channels-1)+1], str_buff);
#endif
#if TEST_ASM
                    TEST_ASSERT_EQUAL_MESSAGE((int8_t)0xCC, ((int8_t*)Y_asm)[IMG_ADDRESS_VECT(&y_params, y_params.height-1, y_params.width-1, y_params.channels-1)+1], str_buff);
#endif
    }


}
#undef DEBUG_ON
#undef CHANNELS
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT_MAX
#undef Y_WIDTH_MAX














#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANNELS        (2*VPU_INT8_VLMACC_ELMS)
#define X_HEIGHT        (5)
#define X_WIDTH         (3)
#define Y_HEIGHT        (X_HEIGHT)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (5)
#define K_w             (3)
#define v_stride        (1)
#define h_stride        (1)
#define ZERO_POINT      (5)
void test_conv2d_depthwise_case3()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANNELS];

    int8_t WORD_ALIGNED  K[K_h][K_w][CHANNELS];

    struct {
        int32_t bias[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift1[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t scale[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift2[MIN_CHAN_OUT_GROUPS(CHANNELS)];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANNELS)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANNELS];

    PRINTF( "test_conv2d_depthwise_case3()...\n");

    print_warns(-1, 1, 1);

    
    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANNELS };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANNELS };

    memset(X, 1, sizeof(X));
    memset(K, 1, sizeof(K));

    for(int k = 0; k < y_params.channels; k++){
        BSS.bias[k]     = k;
        BSS.shift1[k]   = 0;
        BSS.scale[k]    = 1;
        BSS.shift2[k]   = 0;
    }

    nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANNELS);

    nn_conv2d_depthwise_plan_t plan;
    nn_conv2d_depthwise_job_t job;

    conv2d_depthwise_init(&plan, &job, &x_params, &y_params, NULL, -(K_h/2), -(K_w/2), K_h, K_w, v_stride, h_stride, ZERO_POINT, 1);

#if (DEBUG_ON || 0)

#endif //DEBUG_ON



#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y, 0xCC, sizeof(Y)); 
    conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC,  sizeof(Y_asm));
    conv2d_depthwise_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif
    // int8_t Y_exp[Y_HEIGHT][Y_WIDTH] = { {  41  } };

    int8_t Y_exp[Y_HEIGHT][Y_WIDTH] = {
        {   0x33,  0x27,  0x33,  },
        {   0x2B,  0x1B,  0x2B,  },
        {   0x23,  0x0F,  0x23,  },
        {   0x2B,  0x1B,  0x2B,  },
        {   0x33,  0x27,  0x33,  },
    };

    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int8_t y_exp = Y_exp[row][col] + chn;

                check_Y(y_exp, row, col, chn, 0, (int8_t*) Y, &y_params);
            }
        }

    }


}
#undef DEBUG_ON         
#undef CHANNELS         
#undef X_HEIGHT         
#undef X_WIDTH          
#undef Y_HEIGHT         
#undef Y_WIDTH          
#undef K_h          
#undef K_w          
#undef v_stride         
#undef h_stride         
#undef ZERO_POINT        














#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANNELS        (2*VPU_INT8_VLMACC_ELMS)
#define X_HEIGHT        (5)
#define X_WIDTH         (3)
#define Y_HEIGHT        (X_HEIGHT)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (5)
#define K_w             (3)
#define v_stride        (1)
#define h_stride        (1)
#define ZERO_POINT      (5)
void test_conv2d_depthwise_case4()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANNELS];

    int8_t WORD_ALIGNED  K[K_h][K_w][CHANNELS];

    struct {
        int32_t bias[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift1[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t scale[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift2[MIN_CHAN_OUT_GROUPS(CHANNELS)];
    } BSS;

    nn_bss_block_t bss[MIN_CHAN_OUT_GROUPS(CHANNELS)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANNELS];

    PRINTF( "test_conv2d_depthwise_case4()...\n");

    typedef struct {
        struct {
            int rows;
            int cols;
            int channels;
        } output;

        struct {
            int row;
            int col;
            int channel;
        } Y_start;

        unsigned line;
    } test_case_t;


    //   Y[i] = C_in * (x * k)

    const test_case_t casses[] = {
        //  out{ rows    cols    channels }     Y_start{ row    col     chan }       
        {      { 5,      3,      32       },           { 0,     0,      0,   },     __LINE__}, 
        {      { 5,      3,      16       },           { 0,     0,      0,   },     __LINE__}, 
        {      { 5,      2,      32       },           { 0,     0,      0,   },     __LINE__}, 
        {      { 3,      3,      32       },           { 0,     0,      0,   },     __LINE__}, 
        {      { 5,      3,      16       },           { 0,     0,     16,   },     __LINE__}, 
        {      { 5,      2,      32       },           { 0,     1,      0,   },     __LINE__}, 
        {      { 3,      3,      32       },           { 2,     0,      0,   },     __LINE__}, 
        {      { 4,      2,      16       },           { 1,     1,     16,   },     __LINE__}, 
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, 1, 1);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){

        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANNELS };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANNELS };

        memset(X, 1, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, 1, K_h * K_w * y_params.channels * sizeof(int8_t));

        for(int k = 0; k < y_params.channels; k++){
            BSS.bias[k]     = k;
            BSS.shift1[k]   = 0;
            BSS.scale[k]    = 1;
            BSS.shift2[k]   = 0;
        }

        nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANNELS);

        nn_conv2d_depthwise_plan_t plan;
        nn_conv2d_depthwise_job_t job;

        nn_conv2d_job_params_t job_params = { {casse->Y_start.row, casse->Y_start.col, casse->Y_start.channel}, 
                                              {casse->output.rows, casse->output.cols, casse->output.channels} };

        conv2d_depthwise_init(&plan, &job, &x_params, &y_params, &job_params, -(K_h/2), -(K_w/2), K_h, K_w, v_stride, h_stride, ZERO_POINT, 1);

#if (DEBUG_ON || 0)

#endif //DEBUG_ON

        PRINTF("\t\t\tC...\n");
        memset(Y, 0xCC, sizeof(Y)); 
        conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);

        int8_t Y_exp[Y_HEIGHT][Y_WIDTH] = {
            {   0x33,  0x27,  0x33,  },
            {   0x2B,  0x1B,  0x2B,  },
            {   0x23,  0x0F,  0x23,  },
            {   0x2B,  0x1B,  0x2B,  },
            {   0x33,  0x27,  0x33,  },
        };

        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                    int8_t y_exp = Y_exp[row][col] + chn;

                    if(  row < casse->Y_start.row || row >= casse->Y_start.row + casse->output.rows
                      || col < casse->Y_start.col || col >= casse->Y_start.col + casse->output.cols
                      || chn < casse->Y_start.channel || chn >= casse->Y_start.channel + casse->output.channels){
                        y_exp = 0xCC;
                    }

                    check_Y(y_exp, row, col, chn, casse->line, (int8_t*) Y, &y_params);
                }
            }
        }
    }
}
#undef DEBUG_ON         
#undef CHANNELS         
#undef X_HEIGHT         
#undef X_WIDTH          
#undef Y_HEIGHT         
#undef Y_WIDTH          
#undef K_h          
#undef K_w          
#undef v_stride         
#undef h_stride         
#undef ZERO_POINT           














#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANNELS        (2*VPU_INT8_VLMACC_ELMS + 4)
#define X_HEIGHT        (5)
#define X_WIDTH         (3)
#define Y_HEIGHT        (X_HEIGHT)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (5)
#define K_w             (3)
#define v_stride        (1)
#define h_stride        (1)
#define ZERO_POINT      (5)
void test_conv2d_depthwise_case5()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANNELS];

    int8_t WORD_ALIGNED  K[K_h][K_w][CHANNELS];

    struct {
        int32_t bias[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift1[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t scale[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift2[MIN_CHAN_OUT_GROUPS(CHANNELS)];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANNELS)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANNELS];

    PRINTF( "test_conv2d_depthwise_case5()...\n");

    print_warns(-1, 1, 1);

    
    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANNELS };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANNELS };

    memset(X, 1, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
    memset(K, 1, K_h * K_w * y_params.channels * sizeof(int8_t));

    for(int k = 0; k < y_params.channels; k++){
        BSS.bias[k]     = k;
        BSS.shift1[k]   = 0;
        BSS.scale[k]    = 1;
        BSS.shift2[k]   = 0;
    }

    nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANNELS);

#define JOB_COUNT 9
    nn_conv2d_depthwise_plan_t plan;
    nn_conv2d_depthwise_job_t job[JOB_COUNT];

    nn_conv2d_job_params_t job_params[JOB_COUNT] = {
        {   {  0,  0,  0},  {2,  1,  16}},
        {   {  0,  1,  0},  {2,  2,  16}},
        {   {  2,  0,  0},  {2,  3,  16}},
        {   {  4,  0,  0},  {1,  2,  16}},
        {   {  0,  0, 16},  {4,  3,  16}},
        {   {  4,  0, 16},  {1,  1,  16}},
        {   {  0,  0, 32},  {5,  1,   4}},
        {   {  0,  1, 32},  {4,  2,   4}},
        {   {  4,  2,  0},  {1,  1,  36}},  // last block remains uncomputed, and should be 0xCC
        // {   {  4,  1, 16}, {1,  1,  20}}, // Y[4,1,16:] = CC
    };
    assert(sizeof(job_params)/sizeof(nn_conv2d_job_params_t) == JOB_COUNT);

    conv2d_depthwise_init(&plan, job, &x_params, &y_params, job_params, -(K_h/2), -(K_w/2), K_h, K_w, v_stride, h_stride, ZERO_POINT, JOB_COUNT);



#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y, 0xCC, sizeof(Y)); 

    for(int i = 0; i < JOB_COUNT; i++)
        conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job[i]);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC,  sizeof(Y_asm));

    for(int i = 0; i < JOB_COUNT; i++)
        conv2d_depthwise_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job[i]);
#endif

    int8_t Y_exp[Y_HEIGHT][Y_WIDTH] = {
        {   0x33,  0x27,  0x33,  },
        {   0x2B,  0x1B,  0x2B,  },
        {   0x23,  0x0F,  0x23,  },
        {   0x2B,  0x1B,  0x2B,  },
        {   0x33,  0x27,  0x33,  },
    };

    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int8_t y_exp = Y_exp[row][col] + chn;

                if( (row == 4 && col == 1 && chn >= 16))
                    y_exp = 0xCC;

                check_Y(y_exp, row, col, chn, 0, (int8_t*) Y, &y_params);
            }
        }

    }


}
#undef DEBUG_ON         
#undef CHANNELS         
#undef X_HEIGHT         
#undef X_WIDTH          
#undef Y_HEIGHT         
#undef Y_WIDTH          
#undef K_h          
#undef K_w          
#undef v_stride         
#undef h_stride         
#undef ZERO_POINT        





#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANNELS        (4)
#define X_HEIGHT        (10)
#define X_WIDTH         (10)
#define Y_HEIGHT        (5)
#define Y_WIDTH         (5)
#define K_h             (3)
#define K_w             (3)
#define v_stride        (2)
#define h_stride        (2)
#define ZERO_POINT      (10)
void test_conv2d_depthwise_case6_()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANNELS];

    int8_t WORD_ALIGNED  K[K_h][K_w][CHANNELS];

    struct {
        int32_t bias[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift1[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t scale[MIN_CHAN_OUT_GROUPS(CHANNELS)];
        int16_t shift2[MIN_CHAN_OUT_GROUPS(CHANNELS)];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANNELS)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANNELS];

    PRINTF( "test_conv2d_depthwise_case6()...\n");

    print_warns(-1, 1, 1);

    
    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANNELS };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANNELS };

    memset(X, 2, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
    memset(K, 2, K_h * K_w * y_params.channels * sizeof(int8_t));

    for(int k = 0; k < y_params.channels; k++){
        BSS.bias[k]     = 4*k;
        BSS.shift1[k]   = 1;
        BSS.scale[k]    = 2;
        BSS.shift2[k]   = 2;
    }

    nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANNELS);

    nn_conv2d_depthwise_plan_t plan;
    nn_conv2d_depthwise_job_t job;

    conv2d_depthwise_init(&plan, &job, &x_params, &y_params, NULL, -(K_h/2), -(K_w/2), K_h, K_w, v_stride, h_stride, ZERO_POINT, 1);


#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y, 0xCC, sizeof(Y)); 
    conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC,  sizeof(Y_asm));
    conv2d_depthwise_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif

/*
    _____
   |5 5 5|5 5 5 5 5 5 5 5 5
   |5 1 1|1 1 1 1 1 1 1 1 5
   |5_1_1|1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 1 1 1 1 1 1 1 1 1 1 5
    5 5 5 5 5 5 5 5 5 5 5 5

*/

    int8_t Y_exp[Y_HEIGHT][Y_WIDTH] = {
        {   0x1D, 0x15, 0x15, 0x15, 0x15 },
        {   0x15, 0x09, 0x09, 0x09, 0x09 },
        {   0x15, 0x09, 0x09, 0x09, 0x09 },
        {   0x15, 0x09, 0x09, 0x09, 0x09 },
        {   0x15, 0x09, 0x09, 0x09, 0x09 },
    };

    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int8_t y_exp = Y_exp[row][col] + chn;

                check_Y(y_exp, row, col, chn, __LINE__, (int8_t*) Y, &y_params);
            }
        }

    }


}
#undef DEBUG_ON         
#undef CHANNELS         
#undef X_HEIGHT         
#undef X_WIDTH          
#undef Y_HEIGHT         
#undef Y_WIDTH          
#undef K_h          
#undef K_w          
#undef v_stride         
#undef h_stride         
#undef ZERO_POINT        




#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANNELS        (4)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (1)
#define K_h             (2)
#define K_w             (2)
#define v_stride        (1)
#define h_stride        (1)
#define ZERO_POINT      (-1)
void test_conv2d_depthwise_case6()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANNELS] = {{{99,-98, 69, 47}}};

    int8_t WORD_ALIGNED  K[K_h][K_w][CHANNELS] = {
        {  {-127,   24,  -69,   23}, { 75,   97,  -66,   -9}  },
        {  {  52, -127,  -47, -127}, {-93,    7,  127, -106}  }
    };

    struct {
        int32_t bias[CHANNELS];
        int16_t shift1[CHANNELS];
        int16_t scale[CHANNELS];
        int16_t shift2[CHANNELS];
    } BSS = {
        {  -93,     1,   -55,  -219, },
        {    0,     0,     0,     0, },
        {16520, 16748, 32565, 22546, },
        {   21,    21,    22,    22, },
    };

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANNELS)];

    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANNELS];

    PRINTF( "test_conv2d_depthwise_case6()...\n");

    print_warns(-1, 1, 1);
    
    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANNELS };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANNELS };


    nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANNELS);

    nn_conv2d_depthwise_plan_t plan;
    nn_conv2d_depthwise_job_t job;



    conv2d_depthwise_init(&plan, &job, &x_params, &y_params, NULL, 0, 0, K_h, K_w, v_stride, h_stride, ZERO_POINT, 1);

#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y, 0xCC, sizeof(Y)); 
    conv2d_depthwise((int8_t*)Y, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC,  sizeof(Y_asm));
    conv2d_depthwise_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (nn_bss_block_t*) bss, &plan, &job);
#endif

    int8_t Y_exp[Y_HEIGHT][Y_WIDTH][CHANNELS] = {
        {{ -100, -19, -38, 6}}
    };

    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int8_t y_exp = Y_exp[row][col][chn];

                check_Y(y_exp, row, col, chn, __LINE__, (int8_t*) Y, &y_params);
            }
        }

    }


}
#undef DEBUG_ON         
#undef CHANNELS         
#undef X_HEIGHT         
#undef X_WIDTH          
#undef Y_HEIGHT         
#undef Y_WIDTH          
#undef K_h          
#undef K_w          
#undef v_stride         
#undef h_stride         
#undef ZERO_POINT        






void test_conv2d_depthwise()
{
    UNITY_SET_FILE();

    RUN_TEST(test_conv2d_depthwise_case0);
    RUN_TEST(test_conv2d_depthwise_case1);
    RUN_TEST(test_conv2d_depthwise_case2);
    RUN_TEST(test_conv2d_depthwise_case3);
    RUN_TEST(test_conv2d_depthwise_case4);
    RUN_TEST(test_conv2d_depthwise_case5);
    RUN_TEST(test_conv2d_depthwise_case6);
}