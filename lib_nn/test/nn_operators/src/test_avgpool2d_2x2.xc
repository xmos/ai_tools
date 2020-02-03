
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#if (defined(__XS3A__) && USE_ASM_avgpool2d_2x2)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM)     && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)



static unsigned seed = 4321234;







#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define MAX_CHANS   (4*VPU_INT8_ACC_PERIOD)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
void test_avgpool2d_2x2_case1()
{
    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
#if TEST_C
    int8_t WORD_ALIGNED  Y_c[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];
#endif

    PRINTF("test_avgpool2d_2x2_case1()...\n");

    
    typedef struct {
        uint32_t height;    
        uint32_t width;
        uint32_t channels;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               //Chans
        {   2,  2,          16          },  // 0
        {   4,  4,          16          },
        {   6,  6,          16          },
        {   8,  8,          16          },
        {   2,  4,          16          },
        {   4,  2,          16          },  // 5
        {   8,  6,          16          },
        {   2, 16,          16          },
        {  32, 16,          16          },
        {  32, 32,          16          },
        
        {   2,  2,          32          },  // 10
        {   2,  2,          48          },
        {   2,  2,          64          },
        {   2,  2,           4          },
        {   2,  2,           8          },
        {   2,  2,          12          },  // 15
        {   2,  2,          20          },
        {   2,  2,          24          },
        {   2,  2,          28          },
        {   2,  2,          36          },

        {   4,  8,          40          },  // 20
        {  12,  6,          12          },
        {  16,  2,          40          },
        {   4, 24,          36          },
        {  32, 32,          60          },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    memset(X, 120, sizeof(X));

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        printf("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->height, casse->width, casse->channels };
        nn_image_params_t y_params = { casse->height/2, casse->width/2, casse->channels };

        for(int r = 0; r < y_params.height; r++){
            for(int c = 0; c < y_params.width; c++){
                for(int ch = 0; ch < y_params.channels; ch++){
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+0, 2*c+0, ch)] = (ch&1)? 110 : -110;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+1, 2*c+0, ch)] = (ch&1)?  90 : -90;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+0, 2*c+1, ch)] = (ch&1)? 120 : -120;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+1, 2*c+1, ch)] = (ch&1)?  80 : -80;
                }
            }
        }

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, casse->height * casse->width * casse->channels / 4);
        avgpool2d_2x2_c((int8_t*)Y_c, (int8_t*)X, casse->height, casse->width, casse->channels);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC,  casse->height * casse->width * casse->channels / 4);
        avgpool2d_2x2_asm((int8_t*)Y_asm, (int8_t*)X, casse->height, casse->width, casse->channels);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){

                int32_t y_base = IMG_ADDRESS_VECT(&y_params, row, col, 0);

                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = (chn&1)? 100 : -100;

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = ((int8_t*)Y_c)[y_base + chn];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = ((int8_t*)Y_asm)[y_base + chn];
                    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif
                    if(flg){
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, chn);
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




