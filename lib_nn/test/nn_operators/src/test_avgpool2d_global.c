
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "unity.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)





#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define MAX_CHANS   (4*VPU_INT8_ACC_PERIOD)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
void test_avgpool2d_global_case1()
{
    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y[MAX_CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        uint32_t height;    
        uint32_t width;
        uint32_t channels;
        int32_t bias;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               //Chans     //Bias
        {   1,  1,          16,         0          },  // 0
        {   2,  2,          16,         0          },  
        {   4,  4,          16,         0          },
        {   6,  6,          16,         0          },  
        {   8,  8,          16,         0          },
        {   4,  2,          16,         0          },  // 5
        {   8,  6,          16,         0          },
        {   2, 16,          16,         0          },
        {  32, 16,          16,         0          },
        {  32, 32,          16,         0          },
        {   2,  2,          32,         0          },  // 10
        {   2,  2,          48,         0          },
        {   2,  2,          64,         0          },
        {   2,  2,           4,         0          },
        {   2,  2,           8,         0          },
        {   2,  2,          12,         0          },  // 15
        {   2,  2,          20,         0          },
        {   2,  2,          24,         0          },
        {   2,  2,          28,         0          },
        {   2,  2,          36,         0          },
        {   4,  8,          40,         0          },  // 20
        {  12,  6,          12,         0          },
        {  16,  2,          40,         0          },
        {   4, 24,          36,         0          },
        {  32, 32,          60,         0          },
        {   2,  2,          12,         1          },  // 25
        {   2,  2,          20,       -10          },
        {   2,  2,          24,        22          },
        {   2,  2,          28,       -13          },
        {   2,  2,          36,        40          },
        {   4,  8,          40,       -77          },  // 30
        {  12,  6,          12,         3          },
        {  16,  2,          40,       -20          },
        {   4, 24,          36,        44          },
        {  32, 32,          60,      -100          },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, 1, 1);

    memset(X, 120, sizeof(X));

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->height, casse->width, casse->channels };
        nn_image_params_t y_params = { 1, 1, casse->channels };

        for(int r = 0; r < x_params.height; r++){
            for(int c = 0; c < x_params.width; c++){
                for(int ch = 0; ch < x_params.channels; ch++){
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, r, c, ch)] = 24 + ch;
                }
            }
        }

        uint32_t shift;
        uint32_t scale;

        avgpool2d_global_init(&shift, &scale, x_params.height, x_params.width);
        
        int32_t bias = casse->bias * x_params.height * x_params.width * scale;

        PRINTF("\t\tC...\n");
        memset(Y, 0xCC, sizeof(Y));
        avgpool2d_global((int8_t*)Y, (int8_t*)X, casse->height, casse->width, casse->channels, bias, shift, scale);

        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){

                int32_t y_base = IMG_ADDRESS_VECT(&y_params, row, col, 0);

                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = 24 + chn + casse->bias;

                    int8_t y = ((int8_t*)Y)[y_base + chn];

                    if(y != y_exp){
                        sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)", row, col, chn);
                    }

#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM        
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
                }
            }
        }

    }

}
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS
#undef DEBUG_ON

void test_avgpool2d_global()
{
    UNITY_SET_FILE();

    RUN_TEST(test_avgpool2d_global_case1);
}