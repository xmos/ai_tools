
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "../tst_common.h"

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "xs3_vpu.h"

#include "unity.h"


#if USE_ASM(nn_compute_hstrip_tail_deep)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM)     && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#if TEST_C && TEST_ASM
  #define Y_C_ASM  (int8_t*)Y_c, (int8_t*)Y_asm
#elif TEST_C && !TEST_ASM
  #define Y_C_ASM (int8_t*)Y_c
#elif !TEST_C && TEST_ASM
  #define Y_C_ASM (int8_t*)Y_asm
#else
  #error Neither TEST_C nor TEST_ASM is specified.
#endif

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)





static void check_Y(
    const int8_t y_exp, 
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line,
#if TEST_C
    const nn_image_t* Y_c,
#endif
#if TEST_ASM
    const nn_image_t* Y_asm,
#endif
    const nn_image_params_t* y_params)
{
    char str_buff[200];

    unsigned y_offset = IMG_ADDRESS_VECT(y_params, row, col, chn);

    int flg = 0;

    //Only sprintf-ing if the test will fail saves a ton of time.
#if TEST_C
    int8_t y_c = Y_c[y_offset];
    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
    int8_t y_asm = Y_asm[y_offset];
    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif

    if(flg){
        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)  [test vector @ %u]", 
                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                row, col, chn, line);
    }

#if TEST_C
    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM
    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
}














#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT_MAX   (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (1)
#define K_w             (1)
#define K_hstride       (1)
void test_nn_compute_hstrip_tail_deep_case0()
{
    PRINTF("%s...\n", __func__);

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];

    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT_MAX][K_h][K_w][CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT_MAX];
        int16_t shift1[CHANS_OUT_MAX];
        int16_t scale[CHANS_OUT_MAX];
        int16_t shift2[CHANS_OUT_MAX];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT_MAX)];

    nn_image_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS_OUT_MAX];
    nn_image_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS_OUT_MAX];
    
    typedef struct {
        int8_t x;
        int8_t k;
        int8_t expected;
        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  x,  k,        expected,   line
        {   0,  0,               0,   __LINE__ },
        {   1,  0,               0,   __LINE__ },
        {   0,  1,               0,   __LINE__ },
        {   1,  1,        CHANS_IN,   __LINE__ },
        {   2,  1,      2*CHANS_IN,   __LINE__ },
        {   1,  2,      2*CHANS_IN,   __LINE__ },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);

    const unsigned start_case = 0;
    const unsigned stop_case = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= stop_case; v++){
        PRINTF("\tvector %d..\n", v);

        test_case_t* casse = &casses[v];

        for(int C_out = 4; C_out < CHANS_OUT_MAX; C_out += 4){

            PRINTF("\t\tC_out = %d\n", C_out);

            nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
            nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, C_out };


            memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
            memset(K, casse->k, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));


            for(int k = 0; k < y_params.channels; k++){
                BSS.bias[k]     = k;
                BSS.shift1[k]   = 0;
                BSS.scale[k]    = 1;
                BSS.shift2[k]   = 0;
            }

            nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                    (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, C_out);

            mem_stride_t k_cout_stride = -K_h*K_w*x_params.channels;
            nn_tensor_t* K_init = &K[C_out-1][0][0][0];

#if TEST_C
            PRINTF("\t\t\tC...\n");
            memset(Y_c, 0xCC, sizeof(Y_c));
            nn_compute_hstrip_tail_deep_c((nn_image_t*) Y_c, (nn_image_t*) X, K_init, 
                                            (nn_bss_block_t*) &bss, K_h, K_w, K_hstride, x_params.channels,
                                            (x_params.width-K_w)*x_params.channels, k_cout_stride,
                                            y_params.channels, 1, C_out);
#endif
#if TEST_ASM
            PRINTF("\t\t\tASM...\n");
            memset(Y_asm, 0xCC, sizeof(Y_asm));
            nn_compute_hstrip_tail_deep_asm((nn_image_t*) Y_asm, (nn_image_t*) X, K_init, 
                                            (nn_bss_block_t*) &bss, K_h, K_w, K_hstride, x_params.channels,
                                            (x_params.width-K_w)*x_params.channels, k_cout_stride,
                                            y_params.channels, 1, C_out);
#endif

    
            PRINTF("\t\t\tChecking...\n");
            for(unsigned row = 0; row < y_params.height; row++){
                for(unsigned col = 0; col < y_params.width; col++){
                    for(unsigned chn = 0; chn < y_params.channels; chn++){
                        
                        int8_t y_exp = casse->expected + chn;

                        check_Y(y_exp, row, col, chn, casse->line, Y_C_ASM, &y_params);
                    }
                }
            }
        }
    }
}
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT_MAX 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 












#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT_MAX   (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (3)
#define X_WIDTH         (8)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (4)
#define K_h             (3)
#define K_w             (2)
#define K_hstride       (2)
void test_nn_compute_hstrip_tail_deep_case1()
{
    PRINTF("%s...\n", __func__);

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT_MAX][K_h][K_w][CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT_MAX];
        int16_t shift1[CHANS_OUT_MAX];
        int16_t scale[CHANS_OUT_MAX];
        int16_t shift2[CHANS_OUT_MAX];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT_MAX)];

    nn_image_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS_OUT_MAX];
    nn_image_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS_OUT_MAX];

    print_warns(0, TEST_C, TEST_ASM);

    for(int C_out = 4; C_out < CHANS_OUT_MAX; C_out += 4){

        PRINTF("\t\tC_out = %d\n", C_out);

        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, C_out };


        const int8_t x_vals[] = {1,3,5,7};
        const int8_t k_vals[] = {2,4,10};

        const int8_t y_exp[] = {18,54,90,126}; // minus the channel index

        for(int i = 0; i < x_params.height; i++){
            for(int j = 0; j < x_params.width; j++){
                for(int k = 0; k < x_params.channels; k++){
                    X[i][j][k] = x_vals[j/K_w];
                }
            }
        }    
        
        for(int h = 0; h < y_params.channels; h++){
            for(int i = 0; i < K_h; i++){
                for(int j = 0; j < K_w; j++){
                    for(int k = 0; k < x_params.channels; k++){
                        K[h][i][j][k] = k_vals[i];
                    }
                }
            }
        }

        for(int k = 0; k < y_params.channels; k++){
            BSS.shift1[k]   = 6;
            BSS.bias[k]     = -(1<<BSS.shift1[k]) * k;
            BSS.scale[k]    = 64;
            BSS.shift2[k]   = 6;
        }

        nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, C_out);

        mem_stride_t k_cout_stride = -K_h*K_w*x_params.channels;
        nn_tensor_t* K_init = &K[C_out-1][0][0][0];

#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        nn_compute_hstrip_tail_deep_c((nn_image_t*) Y_c, (nn_image_t*) X, K_init, 
                                        (nn_bss_block_t*) &bss, K_h, K_w, K_hstride, x_params.channels,
                                        (x_params.width-K_w)*x_params.channels, k_cout_stride,
                                        y_params.channels, Y_WIDTH, C_out);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        nn_compute_hstrip_tail_deep_asm((nn_image_t*) Y_asm, (nn_image_t*) X, K_init, 
                                        (nn_bss_block_t*) &bss, K_h, K_w, K_hstride, x_params.channels,
                                        (x_params.width-K_w)*x_params.channels, k_cout_stride,
                                        y_params.channels, Y_WIDTH, C_out);
#endif

        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_expected = y_exp[col] - chn;

                    check_Y(y_expected, row, col, chn, __LINE__, Y_C_ASM, &y_params);
                }
            }
        }
    }
}
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT_MAX 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride




void test_nn_compute_hstrip_tail_deep()
{
    UNITY_SET_FILE();

    RUN_TEST(test_nn_compute_hstrip_tail_deep_case0);
    RUN_TEST(test_nn_compute_hstrip_tail_deep_case1);
}